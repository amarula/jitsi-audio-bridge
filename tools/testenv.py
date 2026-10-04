"""Bring up a self-contained environment for testing the bridge.

Starts stub Whisper, Ollama and SMTP services and the real daemon, wired
together with a generated config, then either waits for you to drive it or runs
a scripted meeting itself.

    # Interactive: start everything, print the details, wait for Ctrl-C.
    python3 -m tools.testenv

    # One-shot: run a 10-second three-person meeting and report what came out.
    python3 -m tools.testenv --auto --participants 3 --duration 10

    # Keep the recordings and generated config for inspection.
    python3 -m tools.testenv --auto --workdir /tmp/bridge-demo --keep

Nothing here needs a GPU, a model download, or a mail relay. The point is to
exercise the bridge, not the models: the stubs report what they were sent, and
the SMTP stub writes each accepted message to disk so the summary can be read.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tools import SRC
from tools.stubs import OllamaStub, SmtpStub, WhisperStub, free_port, write_config

#: How long to wait for the bridge to bind, and for post-processing to finish.
STARTUP_TIMEOUT = 20.0
PROCESSING_TIMEOUT = 60.0


class Bridge:
    """The daemon under test, running as a subprocess."""

    def __init__(self, config_path: Path, port: int, workdir: Path, log_level: str = "INFO"):
        self.config_path = config_path
        self.port = port
        self.workdir = workdir
        self.log_path = workdir / "bridge.log"
        self._log_file = None
        self._process: subprocess.Popen | None = None
        self.env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONUNBUFFERED="1")
        self.log_level = log_level

    def start(self) -> Bridge:
        self._log_file = self.log_path.open("wb")
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "jitsi_audio_bridge.daemon",
                "--config",
                str(self.config_path),
                "--log-level",
                self.log_level,
            ],
            env=self.env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        return self

    def wait_until_listening(self, timeout: float = STARTUP_TIMEOUT) -> bool:
        """Poll with a real HTTP request: the server answers, so no noise."""
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                return False
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=0.5)
            except urllib.error.HTTPError:
                return True  # any HTTP status means it is up
            except OSError:
                time.sleep(0.15)
            else:
                return True
        return False

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._log_file:
            self._log_file.close()

    def log(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path.exists() else ""


def report(session_dir: Path, ollama: OllamaStub, whisper: WhisperStub, smtp: SmtpStub) -> None:
    """Print what the run produced."""
    print("\n--- results ---")
    print(f"session directory: {session_dir}")

    if session_dir.is_dir():
        for name in sorted(p.name for p in session_dir.iterdir()):
            print(f"  {name}")

    print(f"\nwhisper calls: {len(whisper.requests)}")
    if whisper.last_audio:
        print(f"  last audio:  {whisper.last_audio}")

    summary_prompts = ollama.summary_prompts
    print(f"ollama summary requests: {len(summary_prompts)}")

    print(f"emails accepted: {len(smtp.messages)}")
    if smtp.mail_dir:
        for path in sorted(smtp.mail_dir.glob("*.eml")):
            print(f"  {path}")

    transcript = session_dir / "transcript.txt"
    if transcript.is_file():
        print("\n--- transcript.txt ---")
        print(transcript.read_text(errors="replace").strip()[:1500])

    if summary_prompts:
        print("\n--- prompt sent to ollama (first 600 chars) ---")
        print(summary_prompts[0][:600])


def main(argv: list[str] | None = None) -> int:
    # Imported here rather than at module level so the interactive mode does
    # not pull in the Opus binding before it is needed.
    from tools.send_meeting import PROTOCOLS
    from tools.send_meeting import main as send_main

    parser = argparse.ArgumentParser(
        prog="python3 -m tools.testenv",
        description="Start stub services and the bridge, ready to be driven.",
    )
    parser.add_argument("--auto", action="store_true", help="run a scripted meeting, then exit")
    parser.add_argument(
        "--participants", type=int, default=2, help="speakers in the scripted meeting"
    )
    parser.add_argument("--duration", type=float, default=10.0, help="seconds of scripted audio")
    parser.add_argument(
        "--session-id", default="testenv", help="session id for the scripted meeting"
    )
    parser.add_argument(
        "--workdir", metavar="PATH", help="where to put config, logs and recordings"
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the workdir (auto mode deletes it)"
    )
    parser.add_argument(
        "--bridge-port", type=int, help="port for the bridge (default: an unused one)"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="bridge log level (default: %(default)s)"
    )
    parser.add_argument(
        "--wav", action="append", metavar="PATH", help="replay a WAV instead of tones"
    )
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default="binary",
        help="wire protocol for the scripted meeting (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if args.workdir:
        workdir = Path(args.workdir)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="bridge-testenv-"))
    workdir.mkdir(parents=True, exist_ok=True)
    recordings = workdir / "recordings"
    mail_dir = workdir / "mail"
    recordings.mkdir(exist_ok=True)
    # Port 8080 is a common collision (SeaweedFS listens there on some hosts),
    # so unless one is pinned, take whatever the OS hands out.
    bridge_port = args.bridge_port or free_port()

    whisper = WhisperStub().start()
    ollama = OllamaStub().start()
    smtp = SmtpStub(mail_dir=mail_dir).start()
    config_path = write_config(
        workdir / "config.ini",
        whisper=whisper,
        ollama=ollama,
        smtp=smtp,
        recordings_dir=recordings,
        bridge_port=bridge_port,
    )
    bridge = Bridge(config_path, bridge_port, workdir, log_level=args.log_level).start()

    # Flushed as it goes: in interactive mode this banner is the only thing
    # telling the operator where to point the sender, and stdout is block
    # buffered when it is not a terminal.
    for line in (
        "test environment",
        f"  workdir    {workdir}",
        f"  config     {config_path}",
        f"  recordings {recordings}",
        f"  mail       {mail_dir}",
        f"  whisper    {whisper.url}",
        f"  ollama     {ollama.url}",
        f"  smtp       127.0.0.1:{smtp.port}",
        f"  bridge     ws://127.0.0.1:{bridge_port}/transcribe",
    ):
        print(line, flush=True)

    try:
        if not bridge.wait_until_listening():
            print("\nthe bridge did not start; its log follows:", file=sys.stderr)
            print(bridge.log(), file=sys.stderr)
            return 1
        print("  status     listening\n", flush=True)

        if not args.auto:
            print("Send a simulated meeting from another shell, for example:\n", flush=True)
            print(
                f"  python3 -m tools.send_meeting --url ws://127.0.0.1:{bridge_port}/transcribe"
                f" --participants 3 --duration 10\n",
                flush=True,
            )
            print("Ctrl-C to stop.\n", flush=True)
            try:
                while bridge.is_running:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print()
            report(recordings / args.session_id, ollama, whisper, smtp)
            return 0

        # Scripted meeting.
        print("running a scripted meeting...")
        send_argv = [
            "--url",
            f"ws://127.0.0.1:{bridge_port}/transcribe",
            "--session-id",
            args.session_id,
            "--participants",
            str(args.participants),
            "--duration",
            str(args.duration),
            "--protocol",
            args.protocol,
            "--fast",
        ]
        for path in args.wav or []:
            send_argv += ["--wav", path]
        if args.wav:
            send_argv += ["--audio", "wav"]

        # The sender is synchronous and runs its own loop; called from here
        # rather than as a subprocess so tracebacks stay in one place.
        with contextlib.suppress(KeyboardInterrupt):
            send_main(send_argv)

        # Wait for post-processing to finish.
        deadline = time.monotonic() + PROCESSING_TIMEOUT
        session_dir = recordings / args.session_id
        while time.monotonic() < deadline:
            if smtp.messages:
                break
            time.sleep(0.2)

        report(session_dir, ollama, whisper, smtp)

        if not smtp.messages:
            print("\nno email was sent; the bridge log follows:")
            print(bridge.log())
            return 1
        return 0

    finally:
        bridge.stop()
        whisper.stop()
        ollama.stop()
        smtp.stop()

        if args.auto and not args.keep and not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        elif not args.auto:
            print(f"\nartifacts kept in {workdir}")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
    raise SystemExit(130)
