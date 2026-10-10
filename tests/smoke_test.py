#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end smoke test for jitsi-audio-bridge.

Brings up the same environment as ``python3 -m tools.testenv`` — stub Whisper,
Ollama and SMTP, plus the real daemon — drives it with the sender simulator
from ``tools.send_meeting``, and asserts on what came out the far end.

Not part of the pytest suite: it binds sockets and spawns a subprocess, so it
is run explicitly:

    python3 tests/smoke_test.py

Exit status is 0 when every check passes.
"""

from __future__ import annotations

import asyncio
import contextlib
import email
import email.message
import json
import os
import re
import sys
import time
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.send_meeting import Participant  # noqa: E402
from tools.send_meeting import main as send_main  # noqa: E402
from tools.stubs import (  # noqa: E402
    OllamaStub,
    S3Stub,
    SmtpStub,
    WhisperStub,
    free_port,
    write_config,
)
from tools.testenv import Bridge  # noqa: E402
from tools.verify_jitsi import main as verify_main  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def settle(smtp: SmtpStub, quiet: float = 3.0, timeout: float = 60.0) -> None:
    """Wait until no new mail has arrived for *quiet* seconds.

    Several checks count messages — "exactly one email", "no email at all" —
    and a count is only meaningful when nothing else is in flight.  Sections
    here run back to back while earlier sessions are still being transcribed
    and mailed: their mail arrives a grace period after their last connection,
    which on a fast machine lands in the middle of the next section's window
    and turns a correct count into a failure.  Waiting for the mailbox to go
    quiet first is what makes those counts assertions about the daemon rather
    than about the machine's speed.
    """
    deadline = time.monotonic() + timeout
    seen = len(smtp.messages)
    changed = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(0.2)
        if len(smtp.messages) != seen:
            seen = len(smtp.messages)
            changed = time.monotonic()
        elif time.monotonic() - changed >= quiet:
            return
    print(f"    (the mailbox never went quiet for {quiet}s; counting anyway)")


def wait_for(predicate, timeout: float, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def send_meeting(url: str, **options: object) -> None:
    """Drive the simulative sender with a list of CLI arguments."""
    argv = ["--url", url]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            for item in value:
                argv += [flag, str(item)]
        else:
            argv += [flag, str(value)]
    with contextlib.suppress(SystemExit):
        send_main(argv)


def check_wrong_path_is_refused(port: int) -> int | None:
    """Connect to a path the bridge does not serve; return the close code."""
    import websockets

    async def attempt() -> int | None:
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/somewhere-else") as ws:
                await asyncio.sleep(0.3)
                await ws.recv()
        except websockets.ConnectionClosed as exc:
            return exc.code
        return None

    return asyncio.run(attempt())


def send_junk_then_audio(url: str, session_id: str) -> bool:
    """Send malformed control frames, then a valid one and real audio.

    Exercises the bridge's control-frame validation: a bad frame is dropped and
    counted, not allowed to tear the session down. Returns whether the
    connection stayed open and the audio was delivered.
    """
    import websockets

    from jitsi_audio_bridge.audio import OpusEncoder
    from tools.send_meeting import frame_samples, tone_frame

    async def attempt() -> bool:
        uri = f"{url}?sessionId={session_id}"
        try:
            async with websockets.connect(uri) as ws:
                for junk in ("not json at all", "[1,2,3]", '"a bare string"', "{}"):
                    await ws.send(junk)
                await ws.send(
                    json.dumps(
                        {"room_name": "Junk", "participants": [{"id": "j1", "name": "Jo",
                                                                 "email": "jo@example.com"}]}
                    )
                )
                encoder = OpusEncoder(48000, 1, "audio")
                try:
                    for index in range(20):
                        packet = encoder.encode(
                            tone_frame(440.0, index, 48000), frame_samples(48000)
                        )
                        await ws.send(b"j1".ljust(16, b"\x00") + packet)
                finally:
                    encoder.close()
                await asyncio.sleep(0.3)
            return True
        except Exception as exc:  # noqa: BLE001 - the point is that it does not happen
            print(f"    session died on a junk frame: {type(exc).__name__}: {exc}")
            return False

    return asyncio.run(attempt())


def send_media_json_events(url: str, session_id: str) -> bool:
    """Send media-json events that must not abort the session, and check the pong.

    Exercises the dispatch and validation rules: junk events, a ping the bridge
    must answer with a matching pong, and media that arrives before any start
    event. Returns whether the connection stayed open and the pong arrived.
    """
    import websockets

    from jitsi_audio_bridge.audio import OpusEncoder
    from tools.send_meeting import (
        frame_samples,
        media_json_media,
        media_json_ping,
        media_json_session_end,
        tone_frame,
    )

    async def attempt() -> bool:
        uri = f"{url}?sessionId={session_id}"
        participant = Participant(identifier="edge", name="Edge", email="edge@example.com")
        try:
            async with websockets.connect(uri) as ws:
                for junk in (
                    "not json at all",
                    "[1,2,3]",
                    '{"event": 5}',
                    '{"event": "media"}',
                    json.dumps({"event": "future-event"}),
                    json.dumps({"event": "ping"}),  # no id: cannot be answered
                ):
                    await ws.send(junk)

                await ws.send(json.dumps(media_json_ping(7)))
                reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if reply != {"event": "pong", "id": 7}:
                    print(f"    expected a pong for id 7, got {reply!r}")
                    return False

                # No start event: the bridge must key the recording by tag anyway.
                encoder = OpusEncoder(48000, 1, "audio")
                try:
                    for index in range(20):
                        packet = encoder.encode(
                            tone_frame(440.0, index, 48000), frame_samples(48000)
                        )
                        await ws.send(
                            json.dumps(media_json_media(participant, index + 1, index,
                                                        index * 960, packet))
                        )
                finally:
                    encoder.close()
                await ws.send(json.dumps(media_json_session_end()))
                await asyncio.sleep(0.3)
            return True
        except Exception as exc:  # noqa: BLE001 - the point is that it does not happen
            print(f"    media-json session died: {type(exc).__name__}: {exc}")
            return False

    return asyncio.run(attempt())


def run_process_dir(config_path: Path, meeting_dir: Path) -> tuple[int, str]:
    """Run the daemon in batch mode over an existing meeting directory."""
    import os
    import subprocess

    from tools import SRC

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jitsi_audio_bridge.daemon",
            "--config",
            str(config_path),
            "--process-dir",
            str(meeting_dir),
        ],
        env=dict(os.environ, PYTHONPATH=str(SRC)),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return result.returncode, result.stdout + result.stderr


def _write_meeting_wavs(meeting_dir: Path, speakers: list[str]) -> None:
    """Write per-speaker WAVs, as Jitsi's own recording would leave behind.

    Named after the address, which is how attribution resolves a speaker when
    the filename carries no participant id.
    """
    from tools.send_meeting import tone_frame

    for index, address in enumerate(speakers):
        path = meeting_dir / f"{address}_audio.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframes(
                b"".join(tone_frame(440.0 + 110 * index, i, 48000) for i in range(25))
            )


def check_directory_mode(
    config_path: Path, workdir: Path, smtp: SmtpStub, ollama: OllamaStub
) -> None:
    """Exercise the batch path over directories that already exist on disk."""
    # --- per-speaker recordings, named after the address -------------------
    jitsi_dir = workdir / "batch-jitsi"
    jitsi_dir.mkdir(exist_ok=True)
    _write_meeting_wavs(jitsi_dir, ["ada@example.com", "grace@example.com"])
    (jitsi_dir / "metadata.json").write_text(
        json.dumps(
            {
                "meeting_url": "https://meet.example.com/Batch-Review",
                "participants": [
                    {"user": {"id": "u1", "name": "Ada", "email": "ada@example.com"}},
                    {"user": {"id": "u2", "name": "Grace", "email": "grace@example.com"}},
                ],
            }
        )
    )

    code, output = run_process_dir(config_path, jitsi_dir)
    check("--process-dir exits successfully on a Jitsi-shaped directory", code == 0,
          f"exit {code}")
    check("a transcript was written", (jitsi_dir / "transcript.txt").is_file())
    check("a summary was written", (jitsi_dir / "summary.md").is_file())
    if (jitsi_dir / "transcript.txt").is_file():
        # Attribution must resolve through the email -> name mapping.
        body = (jitsi_dir / "transcript.txt").read_text()
        check("speakers were attributed from the address mapping",
              "Ada" in body and "Grace" in body, body.strip()[:80])
    # Its own mail, found by the room it is about rather than by being the
    # newest: a session from earlier can still be finishing and deliver after
    # this one.
    check("directory mode sent an email", bool(mailed_for(smtp, "Batch-Review")))
    check("the room came from meeting_url",
          ollama.summary_prompts and "Batch-Review" in ollama.summary_prompts[-1])

    # --- a single master recording, needing ffmpeg extraction --------------
    master_dir = workdir / "batch-master"
    master_dir.mkdir(exist_ok=True)
    (master_dir / "metadata.json").write_text(
        json.dumps({"meeting_url": "https://meet.example.com/Master-Recording"})
    )
    extracted = build_master_recording(master_dir)
    if extracted is None:
        print("    (ffmpeg or flite unavailable; skipping the master-track check)")
        return

    code, output = run_process_dir(config_path, master_dir)
    check("--process-dir handles a single master recording", code == 0, f"exit {code}")
    check("the master track was extracted to 16 kHz",
          (master_dir / "extracted_audio.wav").is_file())
    check("a transcript was written from the extracted audio",
          (master_dir / "transcript.txt").is_file())
    check("the master-recording session sent an email",
          bool(mailed_for(smtp, "Master-Recording")))


def build_master_recording(meeting_dir: Path) -> Path | None:
    """Put a container file in the directory that only ffmpeg can read.

    Uses Matroska because it accepts PCM directly, so this needs no audio
    encoder beyond what a stock ffmpeg build has.
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return None

    source = meeting_dir / "source.wav"
    synthesised = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "flite=text='The master recording is being extracted.'",
            "-ar", "48000", "-ac", "1", "-y", str(source),
        ],
        capture_output=True,
        check=False,
    )
    if synthesised.returncode != 0:
        return None

    master = meeting_dir / "room-recording.mkv"
    muxed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
         "-c:a", "copy", "-y", str(master)],
        capture_output=True,
        check=False,
    )
    source.unlink(missing_ok=True)
    return master if muxed.returncode == 0 else None


def mailed_body(raw: str) -> str:
    """What a reader of the message sees, which is not what was transmitted.

    The message is multipart once anything is attached, so the body is the
    text/plain part — and a body carrying a long URL is quoted-printable: the
    URL is soft-wrapped with "=\\r\\n" and its own "=" characters arrive as
    "=3D", so a substring check on the raw payload fails on exactly the line it
    is looking for.
    """
    message = email.message_from_string(raw)
    for part in message.walk():
        if part.get_content_type() == "text/plain" and not part.get_filename():
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                return payload.decode("utf-8", "replace")
    return raw


def mailed_html(raw: str) -> str:
    """The text/html part, or "" when the message has none."""
    message = email.message_from_string(raw)
    for part in message.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                return payload.decode("utf-8", "replace")
    return ""


def recording_link(body: str) -> str | None:
    """The first URL in *body*, which the recording paragraph puts on its own."""
    found = re.search(r"https?://\S+", body)
    return found.group(0) if found else None


def mailed_for(smtp: SmtpStub, needle: str, timeout: float = 30.0) -> str:
    """The body of the mail that mentions *needle*, waiting for it to arrive.

    Sessions are post-processed independently, so a mail for one can land
    between another's upload and the check that follows it: "the newest
    message" and "one more message than before" are both assertions about
    timing rather than about the mail.  Asking for the one that names what the
    check is about is the only one of the three that stays true on a machine
    that runs the smoke test faster than this one.
    """
    deadline = time.time() + timeout
    while True:
        for raw in smtp.messages:
            body = mailed_body(raw)
            if needle in body:
                return body
        if time.time() >= deadline:
            return ""
        time.sleep(0.2)


def plant_jibri_recording(
    jibri_dir: Path, room: str, age_seconds: float = 60, size: int = 9 << 20
) -> Path:
    """A recording Jibri has already finished, waiting to be picked up.

    Jibri's own layout, down to the layout inside the directory: a session
    directory holding the recording — named after the room, stamped with the
    moment it stopped — and the metadata Jibri leaves beside it.

    The size is above boto3's 8 MiB multipart threshold by default, because
    that is the path every real recording takes; a smaller one would test a
    branch no meeting ever reaches.
    """
    session = jibri_dir / "jibri-session-smoke"
    session.mkdir(parents=True, exist_ok=True)
    stopped = time.time() - age_seconds
    recording = session / f"{room}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime(stopped))}.mp4"
    recording.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * (size - 8))
    os.utime(recording, (stopped, stopped))
    (session / "metadata.json").write_text(
        json.dumps({"meeting_url": f"https://jitsi.example.com/{room}", "participants": []}),
        encoding="utf-8",
    )
    return recording


def main() -> int:
    whisper, ollama, smtp, s3 = WhisperStub(), OllamaStub(), SmtpStub(), S3Stub()
    whisper.start()
    ollama.start()
    smtp.start()
    s3.start()

    workdir = Path("/tmp") / f"jitsi-bridge-smoke-{int(time.time())}"
    recordings = workdir / "recordings"
    mail_dir = workdir / "mail"
    jibri_dir = workdir / "jibri-recordings"
    recordings.mkdir(parents=True)
    jibri_dir.mkdir(parents=True)
    smtp.mail_dir = mail_dir  # keep the delivered message for inspection
    mail_dir.mkdir(exist_ok=True)
    port = free_port()

    config_path = write_config(
        workdir / "config.ini",
        whisper=whisper,
        ollama=ollama,
        smtp=smtp,
        recordings_dir=recordings,
        bridge_port=port,
        s3=s3,
        jibri_dir=jibri_dir,
    )
    bridge = Bridge(config_path, port, workdir).start()
    url = f"ws://127.0.0.1:{port}/transcribe"

    print(f"workdir: {workdir}\nbridge:  {url}\n")

    try:
        print("1. startup")
        started = bridge.wait_until_listening()
        check("the daemon binds and accepts connections", started)
        if not started:
            print(bridge.log())
            return 1

        print("\n2. audio capture")
        send_meeting(
            url,
            session_id="smoketest",
            participants=2,
            duration=4,
            room_name="Smoke Test Room",
            fast=True,
            participant=["alice:Alice:alice@example.com", "bob:Bob:bob@example.com"],
        )

        session = recordings / "smoketest"
        wait_for(lambda: (session / "participant-alice.wav").exists(), timeout=15)

        check("session directory created", session.is_dir())
        check("metadata.json written", (session / "metadata.json").is_file())
        check("alice's recording exists", (session / "participant-alice.wav").is_file())
        check("bob's recording exists", (session / "participant-bob.wav").is_file())

        alice = session / "participant-alice.wav"
        if alice.is_file():
            with wave.open(str(alice), "rb") as handle:
                shape = (handle.getnchannels(), handle.getframerate(), handle.getsampwidth())
                check("WAV is 16 kHz mono 16-bit", shape == (1, 16000, 2), f"{shape}")
                expected = 4 * 16000
                check(
                    "WAV holds the audio that was sent",
                    abs(handle.getnframes() - expected) <= 1600,
                    f"{handle.getnframes()} frames, expected about {expected}",
                )

        print("\n3. post-processing")
        wait_for(lambda: bool(smtp.messages), timeout=60)

        transcript = session / "transcript.txt"
        check("transcript.txt written", transcript.is_file())
        if transcript.is_file():
            body = transcript.read_text()
            check("both speakers are attributed", "Alice" in body and "Bob" in body)
            check("whisper output reached the transcript", "stub transcript" in body)

        audio = whisper.last_audio
        check(
            "whisper received a parseable WAV",
            audio.get("framerate") == 16000 and audio.get("channels") == 1,
            json.dumps(audio),
        )
        check("whisper was called once per participant", len(whisper.requests) == 2,
              f"{len(whisper.requests)} calls")
        check(
            "ollama was asked to detect the language",
            len(ollama.language_prompts) == 1,
            f"{len(ollama.language_prompts)} language probe(s)",
        )
        check(
            "the detected language was carried into the summary prompt",
            any("strictly in English" in p for p in ollama.summary_prompts),
        )
        # The sender emits a meeting_url, so the room name is the URL's last
        # path segment — this is what the real metadata provides.
        check(
            "the room name was derived from meeting_url",
            any("Smoke-Test-Room" in p for p in ollama.summary_prompts),
            "expected 'Smoke-Test-Room' in the prompt",
        )
        check(
            "the summary prompt asks for speaker attribution",
            any("STRICT RULES FOR SPEAKER ANNOTATION" in p for p in ollama.summary_prompts),
        )

        check("exactly one email was sent", len(smtp.messages) == 1,
              f"{len(smtp.messages)} message(s)")
        if smtp.messages:
            mail = smtp.messages[0]
            check("the subject names the room", "Smoke-Test-Room" in mail)
            check(
                "the mail went to both participants",
                "alice@example.com" in mail and "bob@example.com" in mail,
            )
            check("the summary is in the body", "STUB SUMMARY" in mail)
            check("the transcript is attached", "transcript.txt" in mail)
            check("the message was written to disk", any(mail_dir.glob("*.eml")))

            # The HTML part is built and inlined at send time, so this is the
            # only place the whole path — template, stylesheet, inliner and
            # MIME assembly — is exercised together.
            html = mailed_html(mail)
            check("the mail carries an HTML part", html.startswith("<!DOCTYPE html>"),
                  html[:60])
            check(
                "its styles are inlined, not left in a linked stylesheet",
                'style="' in html and "<link" not in html,
            )
            check("the HTML part shows the summary", "STUB SUMMARY" in html)
            check(
                "the plain-text body is still there as the fallback",
                "STUB SUMMARY" in mailed_body(mail),
            )

        print("\n4. security guards")
        send_meeting(url, session_id="../../tmp/pwned", participants=1, duration=0.5, fast=True)
        wait_for(lambda: (recordings / "tmp_pwned").is_dir(), timeout=10)

        check("path traversal did not escape the recordings directory",
              not Path("/tmp/pwned").exists())
        check(
            "the traversal sessionId was rewritten to a safe name",
            (recordings / "tmp_pwned").is_dir(),
            f"contents: {sorted(p.name for p in recordings.iterdir())}",
        )
        check("a connection to a wrong path is refused",
              check_wrong_path_is_refused(port) == 1008)

        print("\n5. empty session handling")
        # The previous session's mail arrives a grace period after its last
        # connection, so let it — and every other session still finishing —
        # land before counting what this one sends.
        settle(smtp)
        before = len(smtp.messages)
        send_meeting(url, session_id="noaudio", participants=1, audio="none", fast=True)
        time.sleep(2)
        empty = recordings / "noaudio"
        check("an audio-less session still records its metadata",
              (empty / "metadata.json").is_file())
        check("an audio-less session produces no recording",
              not list(empty.glob("participant-*.wav")))
        check("an audio-less session produces no transcript",
              not (empty / "transcript.txt").exists())
        check("an audio-less session sends no email", len(smtp.messages) == before)

        print("\n6. malformed control frames")
        # A junk control frame must not abort the session: the audio that
        # follows it still has to be captured and processed.
        settle(smtp)
        before_junk = len(smtp.messages)
        check("a malformed control frame does not kill the session",
              send_junk_then_audio(url, "junkframes"))
        junk_session = recordings / "junkframes"
        check("the session survived and recorded its audio",
              bool(list(junk_session.glob("participant-*.wav"))))

        # That session has audio, so it is being post-processed in a worker
        # thread. Let its email land before counting messages in section 7,
        # or it arrives mid-check and looks like an extra one.
        wait_for(lambda: len(smtp.messages) > before_junk, timeout=30)

        print("\n7. media-json capture (stock Jitsi's framing)")
        settle(smtp)
        before_mediajson = len(smtp.messages)
        send_meeting(
            url,
            session_id="mediajson",
            participants=2,
            duration=4,
            fast=True,
            protocol="media-json",
            participant=["alice:Alice:alice@example.com", "bob:Bob:bob@example.com"],
        )

        mediajson = recordings / "mediajson"
        wait_for(lambda: (mediajson / "participant-alice-audio.wav").exists(), timeout=15)

        check(
            "media-json recordings are keyed by the source tag",
            (mediajson / "participant-alice-audio.wav").is_file()
            and (mediajson / "participant-bob-audio.wav").is_file(),
            f"contents: {sorted(p.name for p in mediajson.iterdir())}"
            if mediajson.is_dir()
            else "no session directory",
        )
        check("media events never reach metadata.json",
              not (mediajson / "metadata.json").exists())

        timeline_file = mediajson / "timeline.json"
        wait_for(lambda: timeline_file.exists(), timeout=15)
        check("a media-json session records a timeline", timeline_file.is_file(),
              f"contents: {sorted(p.name for p in mediajson.iterdir())}"
              if mediajson.is_dir() else "no session directory")
        if timeline_file.is_file():
            timeline = json.loads(timeline_file.read_text())
            speakers = {turn["participant"] for turn in timeline.get("turns", [])}
            check("the timeline has turns for both speakers", len(speakers) == 2,
                  f"{len(timeline.get('turns', []))} turn(s): {timeline.get('turns')}")
            check("every turn carries both clocks",
                  all({"start", "offset"} <= set(turn) for turn in timeline.get("turns", []))
                  and bool(timeline.get("started_at")),
                  str(timeline.get("turns"))[:120])
        check("the turn slices are cleaned up", not (mediajson / ".turns").exists())

        alice_json = mediajson / "participant-alice-audio.wav"
        if alice_json.is_file():
            with wave.open(str(alice_json), "rb") as handle:
                shape = (handle.getnchannels(), handle.getframerate(), handle.getsampwidth())
                check("media-json WAV is 16 kHz mono 16-bit", shape == (1, 16000, 2), f"{shape}")
                expected = 4 * 16000
                check(
                    "media-json WAV holds the audio that was sent",
                    abs(handle.getnframes() - expected) <= 1600,
                    f"{handle.getnframes()} frames, expected about {expected}",
                )

        wait_for(lambda: len(smtp.messages) > before_mediajson, timeout=60)
        transcript = mediajson / "transcript.txt"
        if transcript.is_file():
            body = transcript.read_text()
            # No names arrive on this protocol, so attribution falls back to
            # the recording's own filename.
            check("media-json speakers fall back to their tags",
                  "participant-alice-audio" in body, body.strip()[:80])
            blocks = [block for block in body.split("\n\n") if block.strip()]
            stamped = bool(blocks) and all(
                re.match(r"^\[\d\d:\d\d:\d\d\] ", block) for block in blocks
            )
            check(
                "the transcript is interleaved, every line stamped with its time",
                stamped,
                body.strip()[:120],
            )
            if stamped:
                stamps = [int(block[1:3]) * 60 + int(block[4:6]) for block in blocks]
                check("the lines are in speaking order", stamps == sorted(stamps), str(stamps))
        check("a media-json session is transcribed and emailed",
              len(smtp.messages) == before_mediajson + 1,
              f"{len(smtp.messages) - before_mediajson} message(s)")
        check(
            "with no metadata the room falls back to 'General Meeting'",
            bool(ollama.summary_prompts) and "General Meeting" in ollama.summary_prompts[-1],
        )
        if len(smtp.messages) > before_mediajson:
            check("the fallback recipient got the mail",
                  "fallback@example.com" in smtp.messages[-1])

        print("\n8. media-json edge cases")
        settle(smtp)
        before_edge = len(smtp.messages)
        check("junk media-json events do not kill the session, and the pong is answered",
              send_media_json_events(url, "mediajson-edge"))
        edge = recordings / "mediajson-edge"
        check("media sent before any start event is still recorded",
              bool(list(edge.glob("participant-edge-audio.wav"))))
        check("edge-case events never reach metadata.json",
              not (edge / "metadata.json").exists())
        wait_for(lambda: len(smtp.messages) > before_edge, timeout=60)

        print("\n9. a service that is momentarily down")
        # A 5xx is the service saying "not now": the request is retried, and a
        # participant whose every turn failed is handed over as one recording
        # rather than lost.
        settle(smtp)
        before_flaky = len(smtp.messages)
        whisper.fail_next = 2
        send_meeting(
            url, session_id="flaky", participants=1, duration=2, fast=True,
            protocol="media-json", participant=["dave:Dave:dave@example.com"],
        )
        flaky = recordings / "flaky"
        wait_for(lambda: len(smtp.messages) > before_flaky, timeout=60)
        transcript = flaky / "transcript.txt"
        check("a retried request still produces a transcript",
              transcript.is_file() and bool(transcript.read_text().strip()),
              transcript.read_text().strip()[:120] if transcript.is_file() else "no transcript")
        check("the retry did not lose the turn to the mail",
              len(smtp.messages) == before_flaky + 1,
              f"{len(smtp.messages) - before_flaky} message(s)")

        print("\n10. a reconnect is not the end of the meeting")
        # The JVB ends an export and opens another for the same conference, so
        # a connection ending must not finalise the meeting: no mail from the
        # first connection, no file truncated by the second, and one transcript
        # covering both once the session finally goes quiet.
        settle(smtp)
        before_reconnect = len(smtp.messages)
        send_meeting(
            url, session_id="reconnect", participants=1, duration=2, fast=True,
            protocol="media-json", participant=["carol:Carol:carol@example.com"],
        )
        first = recordings / "reconnect" / "participant-carol-audio.wav"
        wait_for(lambda: first.exists(), timeout=15)
        first_size = first.stat().st_size if first.is_file() else 0

        send_meeting(
            url, session_id="reconnect", participants=1, duration=2, fast=True,
            protocol="media-json", participant=["carol:Carol:carol@example.com"],
        )
        second = recordings / "reconnect" / "participant-carol-audio-2.wav"
        wait_for(lambda: second.exists(), timeout=15)

        contents = sorted(p.name for p in (recordings / "reconnect").iterdir())
        check("the second connection records its own part", second.is_file(),
              f"contents: {contents}")
        check("the first part is not truncated by the second",
              first.is_file() and first.stat().st_size == first_size and first_size > 0,
              f"{first_size} bytes before, {first.stat().st_size if first.is_file() else 0} after")

        wait_for(lambda: len(smtp.messages) > before_reconnect, timeout=60)
        check("one meeting, one email, however many connections",
              len(smtp.messages) == before_reconnect + 1,
              f"{len(smtp.messages) - before_reconnect} message(s)")
        transcript = recordings / "reconnect" / "transcript.txt"
        if transcript.is_file():
            body = transcript.read_text()
            blocks = [block for block in body.split("\n\n") if block.strip()]
            check(
                "both parts reach the transcript, each stamped",
                len(blocks) == 2
                and all(re.match(r"^\[\d\d:\d\d:\d\d\] ", block) for block in blocks),
                body.strip()[:160],
            )
        timeline = recordings / "reconnect" / "timeline.json"
        if timeline.is_file():
            parts = {turn["participant"] for turn in json.loads(timeline.read_text())["turns"]}
            check("the timeline keeps the parts apart", parts == {"carol-audio", "carol-audio-2"},
                  str(sorted(parts)))

        print("\n9. deployment-check probe (tools.verify_jitsi)")
        probe_url = f"{url}?sessionId=verify-jitsi"
        passed = verify_main(["--only", "probe", "--url", probe_url, "--timeout", "5"])
        check("the deployment probe passes against a healthy bridge", passed == 0,
              f"exit {passed}")
        refused = verify_main([
            "--only", "probe", "--url", f"ws://127.0.0.1:{port}/wrong-path",
            "--timeout", "5", "--ping-timeout", "2",
        ])
        check("the deployment probe fails on a path the bridge does not serve",
              refused == 1, f"exit {refused}")

        print("\n10. the meeting's recording is archived to S3")
        # Jibri's tree is empty until now: the recording appears when somebody
        # stops it, which is around when the meeting ends, so planting it just
        # before sending the meeting is what the daemon actually sees.
        planted = plant_jibri_recording(jibri_dir, "Video-Test-Room")
        send_meeting(
            url,
            session_id="s3video",
            participants=1,
            duration=2,
            room_name="Video Test Room",
            fast=True,
            participant=["erin:Erin:erin@example.com"],
        )

        video_session = recordings / "s3video"
        claim = video_session / "video.json"
        wait_for(claim.exists, timeout=60)
        check("a finished recording is uploaded and claimed", claim.is_file(),
              f"contents: {sorted(p.name for p in video_session.iterdir())}"
              if video_session.is_dir() else "no session directory")
        key = f"videos/Video-Test-Room/{planted.name}"
        check("the recording reached the bucket under the room's prefix",
              s3.objects.get(key) == planted.read_bytes(),
              f"keys: {s3.keys()}")
        check("the upload was signed",
              bool(s3.requests) and all(
                  auth.startswith("AWS4-HMAC-SHA256") for _, _, auth in s3.requests
              ),
              str(s3.requests)[:160])
        # Over 8 MiB, so this is the multipart path: initiated, part-uploaded
        # and completed, which is what a real recording does.
        multipart = [path for method, path, _ in s3.requests if method == "POST"]
        check("a recording the size of a real one is uploaded in parts",
              len(multipart) == 2 and any("uploads" in path for path in multipart),
              str([(m, p) for m, p, _ in s3.requests])[:200])
        if claim.is_file():
            written = json.loads(claim.read_text())
            check("the claim names where it went and where it came from",
                  written.get("key") == key and written.get("source") == str(planted),
                  json.dumps(written))
        check("the local recording is kept unless asked otherwise", planted.is_file())

        mail = mailed_for(smtp, "Video-Test-Room")
        check("the mail says where the recording is",
              f"/stub-recordings/videos/Video-Test-Room/{planted.name}" in mail,
              mail[-400:])
        check("and the link it carries is a signed one",
              "X-Amz-Signature=" in mail and "X-Amz-Algorithm=AWS4-HMAC-SHA256" in mail)
        link = recording_link(mail)
        check("the link can be followed", bool(link), mail[-300:])
        if link:
            with contextlib.closing(urllib.request.urlopen(link, timeout=10)) as answer:
                fetched = answer.read()
            check("and what comes back is the recording",
                  fetched == planted.read_bytes(),
                  f"{len(fetched)} bytes back, {planted.stat().st_size} planted")
        # A meeting nobody recorded still gets its summary, without a link.
        check("a meeting with no recording still mails, with no recording line",
              bool(smtp.messages) and "Recording" not in mailed_body(smtp.messages[0]))

        print("\n11. directory mode (--process-dir)")
        check_directory_mode(config_path, workdir, smtp, ollama)

    finally:
        bridge.stop()
        whisper.stop()
        ollama.stop()
        smtp.stop()
        s3.stop()

    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        print("\n--- daemon log ---")
        print(bridge.log())
        return 1
    print("SMOKE TEST PASSED")
    print(f"artifacts in {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
