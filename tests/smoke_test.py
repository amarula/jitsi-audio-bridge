#!/usr/bin/env python3
"""End-to-end smoke test for jitsi-audio-bridge.

Stands up stub Whisper, Ollama and SMTP services, starts the real daemon
against them, connects a WebSocket client that behaves like the JVB, and
checks what came out the far end.

This is not part of the pytest suite: it binds sockets and spawns a
subprocess, so it is run explicitly:

    python3 tests/smoke_test.py

Exit status is 0 when every check passes.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.util
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Stub services
# ---------------------------------------------------------------------------


class _QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # silence per-request logging
        pass


class WhisperStub:
    """Accepts the transcribe payload and validates the audio it is handed."""

    def __init__(self) -> None:
        self.port = free_port()
        self.requests: list[dict] = []
        self.audio_report: dict = {}
        outer = self

        class Handler(_QuietHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                outer.requests.append(body)

                # Decode the attachment and inspect it as a WAV file.
                try:
                    raw = base64.b64decode(body["audio_base64"])
                    with wave.open(BytesIO(raw), "rb") as w:
                        outer.audio_report = {
                            "channels": w.getnchannels(),
                            "sampwidth": w.getsampwidth(),
                            "framerate": w.getframerate(),
                            "frames": w.getnframes(),
                            "filename": body.get("filename"),
                        }
                except Exception as exc:  # noqa: BLE001
                    outer.audio_report = {"error": f"{type(exc).__name__}: {exc}"}

                payload = json.dumps({"text": f"spoken words for {body.get('filename')}"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()


class OllamaStub:
    """Answers the language probe and the summary request."""

    def __init__(self) -> None:
        self.port = free_port()
        self.prompts: list[str] = []
        outer = self

        class Handler(_QuietHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                prompt = body.get("prompt", "")
                outer.prompts.append(prompt)

                # The language probe asks for a bare language name.
                text = "Italian" if "Identify the language" in prompt else "SUMMARY-OUTPUT"
                payload = json.dumps({"response": text}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()


class SmtpStub:
    """Just enough SMTP to accept one message and record it."""

    def __init__(self) -> None:
        self.port = free_port()
        self.messages: list[str] = []
        self._socket = socket.socket()
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", self.port))
        self._socket.listen(5)
        self._running = True

    def start(self) -> None:
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while self._running:
            try:
                conn, _ = self._socket.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn, conn.makefile("rwb") as stream:
            def send(line: str) -> None:
                stream.write(f"{line}\r\n".encode())
                stream.flush()

            send("220 stub ESMTP")
            while True:
                line = stream.readline()
                if not line:
                    return
                command = line.decode("utf-8", "replace").strip()
                upper = command.upper()

                if upper.startswith("EHLO") or upper.startswith("HELO"):
                    send("250-stub")
                    send("250 HELP")
                elif upper.startswith("MAIL FROM") or upper.startswith("RCPT TO"):
                    send("250 OK")
                elif upper.startswith("DATA"):
                    send("354 End data with <CR><LF>.<CR><LF>")
                    chunks: list[str] = []
                    while True:
                        data_line = stream.readline()
                        if not data_line or data_line.strip() == b".":
                            break
                        chunks.append(data_line.decode("utf-8", "replace"))
                    self.messages.append("".join(chunks))
                    send("250 OK: queued")
                elif upper.startswith("QUIT"):
                    send("221 Bye")
                    return
                else:
                    send("250 OK")

    def stop(self) -> None:
        self._running = False
        self._socket.close()


# ---------------------------------------------------------------------------
# Opus encoding, so the client sends something libopus will accept
# ---------------------------------------------------------------------------


def encode_opus_tone(frames: int = 8, frame_samples: int = 960, rate: int = 48000) -> list[bytes]:
    path = ctypes.util.find_library("opus")
    if not path:
        raise SystemExit("libopus is required for the smoke test")
    lib = ctypes.CDLL(path)

    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_encoder_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.opus_encode.restype = ctypes.c_int
    lib.opus_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
    ]
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]

    status = ctypes.c_int(0)
    encoder = lib.opus_encoder_create(rate, 1, 2048, ctypes.byref(status))
    packets: list[bytes] = []
    try:
        for index in range(frames):
            samples = [
                int(12000 * math.sin(2 * math.pi * 440 * (index * frame_samples + i) / rate))
                for i in range(frame_samples)
            ]
            source = (ctypes.c_int16 * frame_samples)(*samples)
            buffer = (ctypes.c_ubyte * 4000)()
            written = lib.opus_encode(encoder, source, frame_samples, buffer, 4000)
            assert written > 0
            packets.append(bytes(buffer[:written]))
    finally:
        lib.opus_encoder_destroy(encoder)
    return packets


# ---------------------------------------------------------------------------


def main() -> int:
    import asyncio

    import websockets

    whisper, ollama, smtp = WhisperStub(), OllamaStub(), SmtpStub()
    whisper.start()
    ollama.start()
    smtp.start()

    workdir = Path(tempfile.mkdtemp(prefix="jitsi-bridge-smoke-"))
    recordings = workdir / "recordings"
    recordings.mkdir()
    bridge_port = free_port()

    config_path = workdir / "config.ini"
    config_path.write_text(
        f"""[server]
host = 127.0.0.1
port = {bridge_port}

[storage]
recordings_dir = {recordings}

[whisper]
url = http://127.0.0.1:{whisper.port}/transcribe-b64
timeout = 30
verify_tls = true

[ollama]
url = http://127.0.0.1:{ollama.port}/api/generate
model = stub-model
timeout = 30
verify_tls = true

[smtp]
host = 127.0.0.1
port = {smtp.port}
sender = bridge@example.com
fallback_recipient = fallback@example.com
use_starttls = false
""",
        encoding="utf-8",
    )

    env = dict(os.environ, PYTHONPATH=str(SRC))
    log_path = workdir / "bridge.log"
    log_file = log_path.open("wb")

    print(f"working directory: {workdir}")
    print(f"bridge port: {bridge_port}\n")

    daemon = subprocess.Popen(
        [sys.executable, "-m", "jitsi_audio_bridge.daemon",
         "--config", str(config_path), "--log-level", "DEBUG"],
        env=env, stdout=log_file, stderr=subprocess.STDOUT,
    )

    try:
        # --- wait for the listener -----------------------------------------
        # Probe with a plain HTTP request rather than a bare TCP connect: the
        # server answers that cleanly, whereas an immediate disconnect is
        # logged as a failed handshake and clutters the output below.
        import urllib.error
        import urllib.request

        deadline = time.time() + 15
        started = False
        while time.time() < deadline:
            if daemon.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{bridge_port}/", timeout=0.5)
            except urllib.error.HTTPError:
                started = True  # any HTTP status means the listener is up
                break
            except OSError:
                time.sleep(0.2)
            else:
                started = True
                break

        print("1. daemon startup and websocket handshake")
        if not started:
            check("daemon is listening", False, f"exited with {daemon.poll()}")
            print(log_path.read_text(errors="replace"))
            return 1
        check("daemon is listening", True)

        packets = encode_opus_tone()

        async def run_client() -> None:
            # The happy path: a control frame plus real Opus audio.
            uri = f"ws://127.0.0.1:{bridge_port}/transcribe?sessionId=smoketest"
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({
                    "room_name": "Smoke Test Room",
                    "participants": [
                        {"id": "alice", "name": "Alice", "email": "alice@example.com"},
                        {"id": "bob", "name": "Bob", "email": "bob@example.com"},
                    ],
                }))
                for packet in packets:
                    for participant in ("alice", "bob"):
                        await ws.send(participant.encode().ljust(16, b"\x00") + packet)
                await asyncio.sleep(0.2)

            # A second connection exercising the traversal guard. It sends a
            # control frame but no audio, so it also covers the empty-session
            # guard: the directory must be created, but nothing may be
            # transcribed or emailed.
            async with websockets.connect(
                f"ws://127.0.0.1:{bridge_port}/transcribe?sessionId=../../tmp/pwned"
            ) as ws:
                await ws.send(json.dumps({"room_name": "Traversal"}))
                await asyncio.sleep(0.1)

            # A third connecting to the wrong path, which must be refused.
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{bridge_port}/somewhere-else"
                ) as ws:
                    await asyncio.sleep(0.2)
                    await ws.recv()
            except websockets.ConnectionClosed as exc:
                refused.append(exc.code)

        refused: list[int] = []
        asyncio.run(run_client())

        # --- the happy path -------------------------------------------------
        print("\n2. audio capture")
        deadline = time.time() + 20
        session_dir = recordings / "smoketest"
        alice = session_dir / "participant-alice.wav"
        while time.time() < deadline and not alice.exists():
            time.sleep(0.2)

        check("session directory created", session_dir.is_dir(), str(session_dir))
        check("metadata.json written", (session_dir / "metadata.json").is_file())
        check("alice's WAV exists", alice.is_file())

        if alice.is_file():
            with wave.open(str(alice), "rb") as w:
                check("WAV is 16 kHz mono 16-bit",
                      (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, 16000, 2),
                      f"{w.getnchannels()}ch {w.getframerate()}Hz {w.getsampwidth() * 8}bit")
                # 8 packets x 20 ms = 160 ms
                check("WAV holds the expected audio",
                      abs(w.getnframes() - 2560) <= 320,
                      f"{w.getnframes()} frames")

        print("\n3. post-processing (whisper, ollama, smtp)")
        deadline = time.time() + 30
        transcript = session_dir / "transcript.txt"
        while time.time() < deadline and not smtp.messages:
            time.sleep(0.2)

        check("transcript.txt written", transcript.is_file())
        if transcript.is_file():
            body = transcript.read_text()
            check("transcript is attributed to both speakers",
                  "Alice" in body and "Bob" in body)
            check("whisper output reached the transcript", "spoken words" in body)

        check("whisper received a parseable WAV",
              whisper.audio_report.get("framerate") == 16000
              and whisper.audio_report.get("channels") == 1,
              json.dumps(whisper.audio_report))
        check("whisper was called once per participant", len(whisper.requests) == 2,
              f"{len(whisper.requests)} calls")
        check("ollama was asked to detect the language",
              any("Identify the language" in p for p in ollama.prompts))
        check("the detected language was carried into the summary prompt",
              any("strictly in Italian" in p for p in ollama.prompts))
        check("the summary prompt names the room",
              any("Smoke Test Room" in p for p in ollama.prompts))

        check("exactly one email was sent", len(smtp.messages) == 1,
              f"{len(smtp.messages)} messages")
        if smtp.messages:
            mail = smtp.messages[0]
            check("email subject names the room", "Smoke Test Room" in mail)
            check("email went to both participants",
                  "alice@example.com" in mail and "bob@example.com" in mail)
            check("summary is in the body", "SUMMARY-OUTPUT" in mail)
            check("transcript is attached", "transcript.txt" in mail)

        print("\n4. security guards")
        check("path traversal did not escape the recordings dir",
              not Path("/tmp/pwned").exists())
        check("traversal sessionId was rewritten to a safe name",
              (recordings / "tmp_pwned").is_dir(),
              f"contents: {[p.name for p in recordings.iterdir()]}")
        check("connection to a wrong path was refused",
              refused == [1008], f"close codes: {refused}")

        print("\n5. empty session handling")
        traversal_dir = recordings / "tmp_pwned"
        check("empty session still recorded its metadata",
              (traversal_dir / "metadata.json").is_file())
        check("empty session produced no recording",
              not list(traversal_dir.glob("participant-*.wav")))
        check("empty session produced no transcript",
              not (traversal_dir / "transcript.txt").exists())
        check("no extra email was sent for the empty session",
              len(smtp.messages) == 1, f"{len(smtp.messages)} messages")

    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon.kill()
        log_file.close()
        whisper.stop()
        ollama.stop()
        smtp.stop()

    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        print("\n--- daemon log ---")
        print(log_path.read_text(errors="replace"))
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
