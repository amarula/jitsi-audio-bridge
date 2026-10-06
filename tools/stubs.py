"""Stub Whisper, Ollama and SMTP services.

These stand in for the three external dependencies so the whole pipeline can be
exercised without a GPU, a model download, or a mail relay. Each one records
what it was asked, which is usually more useful than the answer it gives.

Start all three and print where they landed:

    python3 -m tools.stubs

Or write a config.ini pointing the bridge at them:

    python3 -m tools.stubs --write-config /tmp/testenv.ini

The stub SMTP server also writes each accepted message to ``--mail-dir`` as a
``.eml`` file, so you can actually read the summary that was sent.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import socket
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

logger = logging.getLogger("tools.stubs")

#: Printed instead of a real summary, so its presence in an email is proof the
#: Ollama leg ran.
DEFAULT_SUMMARY = "STUB SUMMARY: the meeting covered the agenda items."

#: Distinguishes the bridge's two Ollama calls. The bridge asks the language
#: question and the summary question of the same endpoint, so the stub has to
#: tell them apart; keying on a phrase unique to the language prompt is more
#: robust than matching on wording that both prompts share.
LANGUAGE_PROBE_MARKER = "Return ONLY the English name of the language"


def free_port() -> int:
    """Ask the OS for a port nobody is using."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _QuietHandler(BaseHTTPRequestHandler):
    """Suppresses the per-request logging that would otherwise flood stdout."""

    def log_message(self, *args: Any) -> None:
        pass


class WhisperStub:
    """Accepts a base64 WAV and returns a fixed transcript.

    It decodes and inspects the audio it is handed rather than ignoring it, so
    a malformed or empty recording shows up as a reported problem instead of
    silently passing.
    """

    def __init__(self, text: str | None = None) -> None:
        self.port = free_port()
        self.fixed_text = text
        self.requests: list[dict] = []
        self.last_audio: dict = {}
        #: Answer this many requests with 503 first, the way a restarting
        #: service does; the daemon is expected to come back and ask again.
        self.fail_next = 0
        outer = self

        class Handler(_QuietHandler):
            def do_POST(self) -> None:  # noqa: N802 - required name
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length))
                except json.JSONDecodeError as exc:
                    outer._reply(self, 400, {"error": f"bad request: {exc}"})
                    return

                outer.requests.append(body)
                outer.last_audio = outer._describe(body.get("audio_base64", ""))

                if outer.fail_next > 0:
                    outer.fail_next -= 1
                    outer._reply(self, 503, {"error": "service unavailable"})
                    return

                text = outer.fixed_text
                if text is None:
                    name = str(body.get("filename", "audio.wav"))
                    text = f"stub transcript of {name}"
                outer._reply(self, 200, {"text": text})

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread: threading.Thread | None = None

    @staticmethod
    def _describe(encoded: str) -> dict:
        """Report the shape of the audio, or why it could not be read."""
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:  # noqa: BLE001 - diagnostic path
            return {"error": f"not valid base64: {exc}"}
        try:
            with wave.open(BytesIO(raw), "rb") as handle:
                return {
                    "bytes": len(raw),
                    "channels": handle.getnchannels(),
                    "sampwidth": handle.getsampwidth(),
                    "framerate": handle.getframerate(),
                    "frames": handle.getnframes(),
                    "seconds": round(handle.getnframes() / handle.getframerate(), 3),
                }
        except Exception as exc:  # noqa: BLE001 - diagnostic path
            return {"bytes": len(raw), "error": f"not a readable WAV: {exc}"}

    @staticmethod
    def _reply(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def start(self) -> WhisperStub:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/transcribe-b64"


class OllamaStub:
    """Answers the language probe, then the summary request.

    Records every prompt so you can inspect exactly what the bridge asked,
    which is the part worth checking by hand.
    """

    def __init__(self, language: str = "English", summary: str = DEFAULT_SUMMARY) -> None:
        self.port = free_port()
        self.language = language
        self.summary = summary
        self.prompts: list[str] = []
        outer = self

        class Handler(_QuietHandler):
            def do_POST(self) -> None:  # noqa: N802 - required name
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length))
                except json.JSONDecodeError as exc:
                    WhisperStub._reply(self, 400, {"error": f"bad request: {exc}"})
                    return

                prompt = str(body.get("prompt", ""))
                outer.prompts.append(prompt)

                # The bridge asks the same endpoint two different questions.
                is_language_probe = LANGUAGE_PROBE_MARKER in prompt
                response = outer.language if is_language_probe else outer.summary
                WhisperStub._reply(self, 200, {"response": response})

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread: threading.Thread | None = None

    def start(self) -> OllamaStub:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/generate"

    @property
    def summary_prompts(self) -> list[str]:
        """Only the prompts that asked for a summary, not the language probe."""
        return [p for p in self.prompts if LANGUAGE_PROBE_MARKER not in p]

    @property
    def language_prompts(self) -> list[str]:
        """Only the prompts that asked which language the transcript is in."""
        return [p for p in self.prompts if LANGUAGE_PROBE_MARKER in p]


class SmtpStub:
    """Just enough SMTP to accept a message and keep it.

    Speaks the bare minimum of the protocol: greeting, EHLO, MAIL, RCPT, DATA,
    QUIT. It does not offer STARTTLS, so point the bridge at it with
    ``use_starttls = false``.
    """

    def __init__(self, mail_dir: Path | None = None) -> None:
        self.port = free_port()
        self.messages: list[str] = []
        self.mail_dir = Path(mail_dir) if mail_dir else None
        if self.mail_dir:
            self.mail_dir.mkdir(parents=True, exist_ok=True)

        self._socket = socket.socket()
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", self.port))
        self._socket.listen(5)
        self._running = True
        self._thread: threading.Thread | None = None

    def start(self) -> SmtpStub:
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        return self

    def _accept_loop(self) -> None:
        while self._running:
            try:
                connection, _ = self._socket.accept()
            except OSError:
                return  # socket closed by stop()
            threading.Thread(target=self._session, args=(connection,), daemon=True).start()

    def _session(self, connection: socket.socket) -> None:
        with connection, connection.makefile("rwb") as stream:

            def send(line: str) -> None:
                stream.write(f"{line}\r\n".encode())
                stream.flush()

            send("220 stub ESMTP")
            while True:
                line = stream.readline()
                if not line:
                    return
                command = line.decode("utf-8", "replace").strip().upper()

                if command.startswith(("EHLO", "HELO")):
                    send("250-stub")
                    send("250 HELP")
                elif command.startswith(("MAIL FROM", "RCPT TO")):
                    send("250 OK")
                elif command.startswith("DATA"):
                    send("354 End data with <CR><LF>.<CR><LF>")
                    chunks: list[str] = []
                    while True:
                        data_line = stream.readline()
                        if not data_line or data_line.strip() == b".":
                            break
                        chunks.append(data_line.decode("utf-8", "replace"))
                    self._record("".join(chunks))
                    send("250 OK: queued")
                elif command.startswith("QUIT"):
                    send("221 Bye")
                    return
                else:
                    send("250 OK")

    def _record(self, message: str) -> None:
        self.messages.append(message)
        index = len(self.messages)
        logger.info("accepted message %d (%d bytes)", index, len(message))
        if self.mail_dir:
            path = self.mail_dir / f"message-{index:03d}.eml"
            path.write_text(message, encoding="utf-8")
            logger.info("wrote %s", path)

    def stop(self) -> None:
        self._running = False
        self._socket.close()


def write_config(
    path: Path,
    *,
    whisper: WhisperStub,
    ollama: OllamaStub,
    smtp: SmtpStub,
    recordings_dir: Path,
    bridge_port: int,
    host: str = "127.0.0.1",
) -> Path:
    """Write a config.ini pointing the bridge at the given stubs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""; Generated by tools/stubs.py — points the bridge at stub services.
[server]
host = {host}
port = {bridge_port}

[storage]
recordings_dir = {recordings_dir}
; A connection ending is not the meeting ending, so the daemon waits this long
; before post-processing.  The tests wait for the mail, so keep it short.
session_grace_seconds = 1

[whisper]
url = {whisper.url}
timeout = 30
verify_tls = true

[ollama]
url = {ollama.url}
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
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m tools.stubs",
        description="Run stub Whisper, Ollama and SMTP services.",
    )
    parser.add_argument("--write-config", metavar="PATH", help="write a matching config.ini here")
    parser.add_argument(
        "--recordings-dir", metavar="PATH", help="recordings_dir to put in that config"
    )
    parser.add_argument("--bridge-port", type=int, default=8080, help="port for that config")
    parser.add_argument("--mail-dir", metavar="PATH", help="write accepted messages here as .eml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    whisper = WhisperStub().start()
    ollama = OllamaStub().start()
    smtp = SmtpStub(mail_dir=args.mail_dir).start()

    print(f"whisper  {whisper.url}")
    print(f"ollama   {ollama.url}")
    print(f"smtp     127.0.0.1:{smtp.port}  (no STARTTLS)")

    if args.write_config:
        path = write_config(
            Path(args.write_config),
            whisper=whisper,
            ollama=ollama,
            smtp=smtp,
            recordings_dir=Path(args.recordings_dir or "/tmp/jitsi-audio-bridge/recordings"),
            bridge_port=args.bridge_port,
        )
        print(f"config   {path}")

    print("\nCtrl-C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print()
    finally:
        whisper.stop()
        ollama.stop()
        smtp.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
