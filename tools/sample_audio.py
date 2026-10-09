# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate speech samples for replaying through the bridge.

The tone generator in ``tools.send_meeting`` proves that audio survives the
round trip, but a tone gives Whisper nothing to transcribe — the transcript
comes back empty, correctly. To exercise the transcription path properly you
need real speech, and ffmpeg can synthesise it:

    python3 -m tools.sample_audio --outdir /tmp/samples

    python3 -m tools.send_meeting --audio wav \\
        --wav /tmp/samples/alice.wav --wav /tmp/samples/bob.wav \\
        --participant alice:Alice --participant bob:Bob

Each file says something distinct, so a transcript that attributes the right
words to the right speaker is proof the whole chain worked.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

#: One line per speaker. The wording is deliberately different for each, so a
#: misplaced attribution is visible in the transcript.
DEFAULT_LINES = (
    ("alice", "Good morning everyone, let's review the release schedule."),
    ("bob", "I can have the database migration finished by Thursday."),
    ("carol", "The staging environment is ready for the load test."),
)

#: libflite voices that ship with ffmpeg.
VOICES = ("slt", "kal", "awb", "rms")


class SynthesisUnavailable(RuntimeError):
    """Raised when ffmpeg cannot synthesise speech on this machine."""


def check_available() -> None:
    """Verify ffmpeg exists and was built with the flite filter."""
    if not shutil.which("ffmpeg"):
        raise SynthesisUnavailable("ffmpeg was not found on PATH")

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    if "flite" not in result.stdout:
        raise SynthesisUnavailable(
            "this ffmpeg was built without libflite, so it cannot synthesise speech; "
            "supply your own WAV files with --wav instead"
        )


def synthesise(text: str, out_path: Path, voice: str = "slt", rate: int = 48000) -> Path:
    """Render *text* to a mono 16-bit WAV at *rate*.

    The rate must be one libopus can encode, which keeps the replay path free
    of resampling.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # flite's text argument is a filter option, so single quotes inside the
    # text would terminate it early; strip them rather than escaping.
    safe = text.replace("'", "").replace(":", " ")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"flite=text='{safe}':voice={voice}",
        "-ar",
        str(rate),
        "-ac",
        "1",
        "-y",
        str(out_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m tools.sample_audio",
        description="Synthesise speech WAV files for replaying through the bridge.",
    )
    parser.add_argument("--outdir", default="/tmp/jitsi-samples", help="where to write the files")
    parser.add_argument("--rate", type=int, default=48000, help="sample rate (default: 48000)")
    parser.add_argument("--voice", default="slt", choices=VOICES, help="libflite voice")
    args = parser.parse_args(argv)

    try:
        check_available()
    except SynthesisUnavailable as exc:
        print(f"cannot synthesise speech: {exc}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    for index, (name, line) in enumerate(DEFAULT_LINES):
        voice = VOICES[index % len(VOICES)] if args.voice == "slt" else args.voice
        path = synthesise(line, outdir / f"{name}.wav", voice=voice, rate=args.rate)
        print(f"{path}  ({path.stat().st_size // 1024} KiB)  {voice}: {line}")

    print("\nreplay them with:\n  python3 -m tools.send_meeting --audio wav \\")
    for name, _ in DEFAULT_LINES:
        print(f"      --wav {outdir / f'{name}.wav'} \\")
    print("      --participant alice:Alice --participant bob:Bob --participant carol:Carol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
