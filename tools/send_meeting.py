"""Simulate a meeting by sending audio to a running bridge.

Stands in for whatever forwards participant audio from the Jitsi side, speaking
the wire protocol documented in README.md:

    ws://<host>:<port>/transcribe?sessionId=<id>
      text frame   -> JSON control frame (room, participants)
      binary frame -> [16-byte participant id][one Opus packet]

Examples:

    # Three participants, ten seconds of distinguishable tones, in real time.
    python3 -m tools.send_meeting --participants 3 --duration 10

    # Replay real speech, one file per participant, as fast as possible.
    python3 -m tools.send_meeting --audio wav --wav alice.wav --wav bob.wav --fast

    # A control frame with no audio at all, to check nothing gets emailed.
    python3 -m tools.send_meeting --audio none

Because this is the only implementation of the sender side, it doubles as the
way to confirm the protocol against a real sender: point it at the bridge and
compare what the bridge records with what the real one produces.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import struct
import sys
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from jitsi_audio_bridge.audio import OpusEncoder

#: Opus frames are 20 ms, which is what Jitsi itself uses.
FRAME_MS = 20

#: The identifier field is fixed width and NUL padded.
ID_FIELD_BYTES = 16

#: The sample rates libopus can encode at. A WAV in any other rate has to be
#: converted before it can be replayed.
ENCODABLE_RATES = (8000, 12000, 16000, 24000, 48000)

DEFAULT_RATE = 48000

#: Amplitude of a generated tone, well below full scale.
_TONE_AMPLITUDE = 12000


@dataclass
class Participant:
    """One simulated speaker."""

    identifier: str
    name: str
    email: str
    #: Distinguishes participants audibly when tones are generated.
    frequency: float = 440.0
    #: Encoded Opus packets, filled in by :func:`encode_streams`.
    packets: list[bytes] = field(default_factory=list)
    packets_sent: int = 0

    @property
    def frame_bytes(self) -> bytes:
        """The fixed-width identifier field."""
        return self.identifier.encode("utf-8")[:ID_FIELD_BYTES].ljust(ID_FIELD_BYTES, b"\x00")


def frame_samples(rate: int) -> int:
    """Samples per channel in one 20 ms frame at *rate*."""
    return rate * FRAME_MS // 1000


def tone_frame(frequency: float, index: int, rate: int) -> bytes:
    """Generate one frame of a sine tone as 16-bit mono PCM at *rate*."""
    count = frame_samples(rate)
    start = index * count
    samples = [
        int(_TONE_AMPLITUDE * math.sin(2 * math.pi * frequency * (start + i) / rate))
        for i in range(count)
    ]
    return struct.pack(f"<{count}h", *samples)


def load_wav_frames(path: Path) -> tuple[list[bytes], int]:
    """Read a WAV into 20 ms PCM frames, downmixing stereo to mono.

    Returns the frames and the rate they were read at, so the caller can encode
    at the file's own rate rather than resampling it.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        file_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    if width != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM, found {width * 8}-bit")
    if file_rate not in ENCODABLE_RATES:
        raise SystemExit(
            f"{path}: {file_rate} Hz is not a rate libopus can encode. Convert it first:\n"
            f"    ffmpeg -i {path} -ar 48000 -ac 1 {path.with_suffix('.48k.wav')}"
        )
    if channels not in (1, 2):
        raise SystemExit(f"{path}: {channels} channels is not supported")

    if channels == 2:
        stereo = struct.unpack(f"<{len(raw) // 2}h", raw)
        raw = struct.pack(f"<{len(stereo) // 2}h", *stereo[::2])

    size = frame_samples(file_rate) * 2
    frames = [raw[offset : offset + size] for offset in range(0, len(raw), size)]
    # Drop a trailing partial frame: Opus needs exactly one frame's worth.
    if frames and len(frames[-1]) != size:
        frames.pop()
    if not frames:
        raise SystemExit(f"{path}: shorter than one {FRAME_MS} ms frame")
    return frames, file_rate


def pcm_stream(
    participant: Participant,
    wav_frames: list[bytes] | None,
    total_frames: int,
    rate: int,
) -> Iterator[bytes]:
    """Yield *total_frames* frames of PCM for one participant.

    Replayed audio loops if the meeting outlasts the file, so a fixed
    ``--duration`` always produces the same number of packets.
    """
    if wav_frames:
        for index in range(total_frames):
            yield wav_frames[index % len(wav_frames)]
    else:
        for index in range(total_frames):
            yield tone_frame(participant.frequency, index, rate)


def encode_streams(
    participants: list[Participant],
    wav_sources: dict[str, tuple[list[bytes], int]],
    total_frames: int,
    rate: int,
) -> None:
    """Encode every participant's stream up front.

    Doing the work here keeps the send loop free of per-packet encoding, so
    real-time pacing measures the bridge rather than this script.
    """
    for participant in participants:
        frames, source_rate = wav_sources.get(participant.identifier, (None, rate))
        # Each Opus packet is self-describing, so a replayed file can be encoded
        # at its own rate while generated tones use --rate. The bridge decodes
        # both to 16 kHz regardless.
        encoder_rate = source_rate if frames else rate
        encoder = OpusEncoder(encoder_rate, 1, "audio")
        try:
            participant.packets = [
                encoder.encode(pcm, frame_samples(encoder_rate))
                for pcm in pcm_stream(participant, frames, total_frames, encoder_rate)
            ]
        finally:
            encoder.close()


async def stream(
    websocket: object,
    participants: list[Participant],
    realtime: bool,
) -> None:
    """Send one packet per participant per tick, interleaved.

    Interleaving is what a real conference looks like, and it is what exercises
    the bridge's per-participant decoder state.
    """
    total = max(len(p.packets) for p in participants) if participants else 0
    for index in range(total):
        for participant in participants:
            if index >= len(participant.packets):
                continue
            try:
                await websocket.send(participant.frame_bytes + participant.packets[index])
            except Exception as exc:  # noqa: BLE001 - report rather than traceback
                print(f"send failed at frame {index}: {exc}", file=sys.stderr)
                return
            participant.packets_sent += 1
        if realtime:
            await asyncio.sleep(FRAME_MS / 1000)


async def run(args: argparse.Namespace) -> int:
    import websockets

    participants = build_participants(args)
    total_frames = max(1, round(args.duration * 1000 / FRAME_MS))

    wav_sources: dict[str, tuple[list[bytes], int]] = {}
    for participant, path in zip(participants, args.wav or [], strict=False):
        frames, source_rate = load_wav_frames(Path(path))
        wav_sources[participant.identifier] = (frames, source_rate)
        print(
            f"{participant.identifier}: replaying {path} "
            f"({source_rate} Hz, {len(frames) * FRAME_MS / 1000:.1f}s)"
        )

    if args.audio != "none":
        encode_streams(participants, wav_sources, total_frames, args.rate)

    separator = "&" if "?" in args.url else "?"
    uri = f"{args.url}{separator}sessionId={args.session_id}" if args.session_id else args.url

    # flush so this lands before any error text, which goes to stderr unbuffered
    print(f"connecting to {uri}", flush=True)
    try:
        async with websockets.connect(uri) as websocket:
            await exchange(websocket, args, participants, total_frames)
    except OSError as exc:
        # The common case by far: nothing is listening yet.
        print(
            f"cannot reach {uri}: {exc}\n"
            "Is the bridge running? Start one with: python3 -m tools.testenv",
            file=sys.stderr,
        )
        return 1
    except websockets.WebSocketException as exc:
        print(f"{uri} refused the connection: {exc}", file=sys.stderr)
        return 1
    return 0


async def exchange(
    websocket: object,
    args: argparse.Namespace,
    participants: list[Participant],
    total_frames: int,
) -> None:
    """Send the control frame and the audio, then let the connection close."""
    if args.metadata:
        control = {
            "room_name": args.room_name,
            "participants": [
                {"id": p.identifier, "name": p.name, "email": p.email} for p in participants
            ],
        }
        await websocket.send(json.dumps(control))
        print(f"sent control frame: {len(participants)} participant(s)")
    else:
        print("skipping the control frame (--no-metadata)")

    if args.audio == "none":
        print("sending no audio (--audio none)")
    else:
        print(
            f"streaming {total_frames} frames "
            f"({total_frames * FRAME_MS / 1000:.1f}s per participant) "
            f"{'in real time' if args.realtime else 'as fast as possible'}"
        )
        await stream(websocket, participants, args.realtime)

    await asyncio.sleep(0.3)  # let the last frames drain before closing

    if args.audio != "none":
        sent = sum(p.packets_sent for p in participants)
        print(f"sent {sent} packets across {len(participants)} participant(s)")

    print("closed; the bridge will now transcribe, summarise and email")


def build_participants(args: argparse.Namespace) -> list[Participant]:
    """Build the participant list, either explicitly or from --participants."""
    if args.participant:
        participants = []
        for index, spec in enumerate(args.participant):
            identifier, _, rest = spec.partition(":")
            name, _, email = rest.partition(":")
            participants.append(
                Participant(
                    identifier=identifier,
                    name=name or identifier.title(),
                    email=email or f"{identifier}@example.com",
                    frequency=440.0 + 110.0 * index,
                )
            )
    else:
        participants = [
            Participant(
                identifier=f"participant-{index + 1}",
                name=f"Speaker {index + 1}",
                email=f"speaker{index + 1}@example.com",
                frequency=440.0 + 110.0 * index,
            )
            for index in range(args.participants)
        ]

    if len(args.wav or []) > len(participants):
        raise SystemExit(
            f"--wav was given {len(args.wav)} time(s) but there are only "
            f"{len(participants)} participants; add more participants or pass fewer files"
        )
    return participants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m tools.send_meeting",
        description="Simulate a Jitsi meeting against a running bridge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for worked examples.",
    )
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8080/transcribe",
        help="bridge endpoint (default: %(default)s)",
    )
    parser.add_argument("--session-id", default="testenv", help="meeting identifier")
    parser.add_argument(
        "--room-name", default="Simulated Meeting", help="room name in the metadata"
    )
    parser.add_argument(
        "--participants", type=int, default=2, help="how many speakers (default: 2)"
    )
    parser.add_argument(
        "--participant",
        action="append",
        metavar="ID[:NAME[:EMAIL]]",
        help="define a speaker explicitly; repeatable, overrides --participants",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="seconds of audio (default: %(default)s)"
    )
    parser.add_argument(
        "--audio",
        choices=["tone", "wav", "none"],
        default="tone",
        help="tone generates sine waves, wav replays --wav, none sends no audio",
    )
    parser.add_argument(
        "--wav",
        action="append",
        metavar="PATH",
        help="16-bit WAV to replay, one per participant; repeatable",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=DEFAULT_RATE,
        choices=ENCODABLE_RATES,
        help="encoder rate for generated tones (default: %(default)s)",
    )
    parser.add_argument(
        "--fast",
        dest="realtime",
        action="store_false",
        help="send as fast as possible instead of pacing at 20 ms per frame",
    )
    parser.add_argument(
        "--no-metadata",
        dest="metadata",
        action="store_false",
        help="skip the control frame, leaving the bridge without room or participants",
    )
    parser.set_defaults(realtime=True, metadata=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audio == "wav" and not args.wav:
        raise SystemExit("--audio wav requires at least one --wav PATH")
    if args.wav and args.audio != "wav":
        raise SystemExit("--wav was given but --audio is not 'wav'")
    with contextlib.suppress(KeyboardInterrupt):
        return asyncio.run(run(args))
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
