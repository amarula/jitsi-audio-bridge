#!/usr/bin/env python3
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
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.send_meeting import main as send_main  # noqa: E402
from tools.stubs import OllamaStub, SmtpStub, WhisperStub, free_port, write_config  # noqa: E402
from tools.testenv import Bridge  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


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

    before = len(smtp.messages)
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
    check("directory mode sent an email", len(smtp.messages) == before + 1)
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

    before = len(smtp.messages)
    code, output = run_process_dir(config_path, master_dir)
    check("--process-dir handles a single master recording", code == 0, f"exit {code}")
    check("the master track was extracted to 16 kHz",
          (master_dir / "extracted_audio.wav").is_file())
    check("a transcript was written from the extracted audio",
          (master_dir / "transcript.txt").is_file())
    check("the master-recording session sent an email", len(smtp.messages) == before + 1)


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


def main() -> int:
    whisper, ollama, smtp = WhisperStub(), OllamaStub(), SmtpStub()
    whisper.start()
    ollama.start()
    smtp.start()

    workdir = Path("/tmp") / f"jitsi-bridge-smoke-{int(time.time())}"
    recordings = workdir / "recordings"
    mail_dir = workdir / "mail"
    recordings.mkdir(parents=True)
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

        print("\n7. directory mode (--process-dir)")
        check_directory_mode(config_path, workdir, smtp, ollama)

    finally:
        bridge.stop()
        whisper.stop()
        ollama.stop()
        smtp.stop()

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
