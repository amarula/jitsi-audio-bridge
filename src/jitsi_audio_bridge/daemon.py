"""WebSocket server that captures per-participant audio from a Jitsi bridge.

One connection corresponds to one meeting.  The peer sends a JSON control
frame describing the room and its participants, then a stream of binary frames
each holding a single participant's Opus packet.  When the connection closes,
the recorded audio is transcribed, summarised and emailed.

This is the only module that knows about asyncio.  The blocking half of the
pipeline lives in :func:`process_completed_session` and is pushed onto a worker
thread via :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import json
import logging
import os
import re
import signal
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import websockets

from . import __version__
from .ai_client import generate_summary, transcribe_audio
from .audio import OpusError, OpusParticipantRecorder, parse_metadata
from .config import Config, ConfigError, load_config
from .mailer import send_meeting_email

logger = logging.getLogger(__name__)

#: The only request path accepted.  The JVB is expected to connect here.
WEBSOCKET_PATH = "/transcribe"

#: Size of the participant-identifier prefix on every binary frame.
PARTICIPANT_ID_BYTES = 16

#: Session identifiers are used as directory names, so they are capped.
MAX_IDENTIFIER_LENGTH = 64

#: Used when a connection carries no usable ``sessionId``.
DEFAULT_SESSION_ID = "session_default"

#: Filename of the control frame's payload, inside the meeting directory.
METADATA_FILENAME = "metadata.json"

#: How many meetings may be transcribed and summarised at once.  Whisper and
#: Ollama are the bottleneck; unbounded fan-out would just queue inside them.
MAX_CONCURRENT_JOBS = 2

#: Anything outside this set is replaced when an identifier becomes a path
#: component or a filename.
_UNSAFE_IDENTIFIER_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

_PROCESSING_SEMAPHORE: asyncio.Semaphore | None = None


class StreamError(Exception):
    """Raised when a frame from the peer cannot be interpreted."""


def sanitize_identifier(raw: str, fallback: str = DEFAULT_SESSION_ID) -> str:
    """Reduce *raw* to something safe to use as a single path component.

    Identifiers reach this function straight off the wire, so they must never
    be trusted: ``sessionId=../../etc`` would otherwise escape the recordings
    directory.  Everything outside ``[A-Za-z0-9._-]`` — including ``/`` — is
    replaced, and leading and trailing dots are stripped so that ``.`` and
    ``..`` cannot survive.
    """
    cleaned = _UNSAFE_IDENTIFIER_CHARS.sub("_", raw).strip("._")
    cleaned = cleaned[:MAX_IDENTIFIER_LENGTH].strip("._")
    return cleaned or fallback


def extract_session_id(request_path: str) -> str:
    """Pull the meeting's ``sessionId`` out of a request path."""
    query = parse_qs(urlparse(request_path).query)
    values = query.get("sessionId") or []
    if not values:
        logger.warning("no sessionId in %r; using %r", request_path, DEFAULT_SESSION_ID)
        return DEFAULT_SESSION_ID

    raw = values[0]
    session_id = sanitize_identifier(raw)
    if session_id != raw:
        logger.warning("sessionId %r sanitized to %r", raw, session_id)
    return session_id


def split_frame(message: bytes) -> tuple[str, bytes]:
    """Split a binary frame into ``(participant identifier, Opus payload)``.

    Raises:
        StreamError: if the frame is too short to hold an identifier and some
            payload, or carries no identifier.
    """
    if len(message) <= PARTICIPANT_ID_BYTES:
        raise StreamError(
            f"frame of {len(message)} bytes is too short to carry a "
            f"{PARTICIPANT_ID_BYTES}-byte identifier and any payload"
        )

    # Identifiers arrive NUL- or space-padded into a fixed-width field.
    raw_id = message[:PARTICIPANT_ID_BYTES].rstrip(b"\x00").decode("utf-8", "replace").strip()
    if not raw_id:
        raise StreamError("frame carries an empty participant identifier")

    participant_id = sanitize_identifier(raw_id, fallback="")
    if not participant_id:
        raise StreamError(f"participant identifier {raw_id!r} has no usable characters")

    return participant_id, message[PARTICIPANT_ID_BYTES:]


def write_metadata(meeting_dir: Path, raw: str) -> bool:
    """Store a control frame as ``metadata.json``, atomically.

    Written to a temporary file and renamed into place so that a reader — or
    the post-processing step of a concurrent session — never observes a
    truncated document.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("ignoring malformed control frame: %s", exc)
        return False

    if not isinstance(data, dict):
        logger.warning("ignoring control frame that is not a JSON object")
        return False

    target = meeting_dir / METADATA_FILENAME
    handle_fd, temporary = tempfile.mkstemp(dir=meeting_dir, prefix=".metadata-", suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.replace(temporary, target)
    except OSError as exc:
        logger.error("cannot write %s: %s", target, exc)
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        return False

    return True


def _semaphore() -> asyncio.Semaphore:
    """Return the processing semaphore, binding it to the running loop."""
    global _PROCESSING_SEMAPHORE
    if _PROCESSING_SEMAPHORE is None:
        _PROCESSING_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    return _PROCESSING_SEMAPHORE


def process_completed_session(meeting_dir: Path, config: Config) -> None:
    """Transcribe, summarise and email one finished meeting.

    Blocking: transcription and summary generation take minutes.  Always call
    this through :func:`asyncio.to_thread`, never directly from a coroutine.
    """
    logger.info("post-processing %s", meeting_dir)
    metadata = parse_metadata(meeting_dir)

    recordings = sorted(meeting_dir.glob("participant-*.wav"))
    if not recordings:
        logger.info("no participant recordings in %s; nothing to do", meeting_dir)
        return

    transcript_lines: list[str] = []
    for recording in recordings:
        participant_id = recording.stem.removeprefix("participant-")
        speaker = metadata["id_to_name"].get(participant_id, participant_id)

        text = transcribe_audio(recording, config.whisper)
        if text:
            transcript_lines.append(f"[{speaker}]: {text}")
        else:
            logger.warning("no transcript produced for %s", recording.name)

    if not transcript_lines:
        logger.warning("nothing was transcribed for %s; skipping the email", meeting_dir)
        return

    transcript = "\n\n".join(transcript_lines)
    transcript_path = meeting_dir / "transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    logger.info("wrote %s (%d participants)", transcript_path, len(transcript_lines))

    summary = generate_summary(transcript, metadata["room_name"], config.ollama)
    send_meeting_email(
        metadata["recipients"], metadata["room_name"], summary, transcript_path, config.smtp
    )


async def _run_post_processing(meeting_dir: Path, config: Config) -> None:
    """Run the blocking pipeline off the event loop, bounded by a semaphore."""
    async with _semaphore():
        try:
            await asyncio.to_thread(process_completed_session, meeting_dir, config)
        except Exception:
            # A failed pipeline must not propagate into the connection handler.
            logger.exception("post-processing failed for %s", meeting_dir)


async def handle_jvb_stream(websocket: Any, config: Config) -> None:
    """Serve one meeting's WebSocket connection."""
    request_path = websocket.request.path
    if urlparse(request_path).path != WEBSOCKET_PATH:
        logger.warning(
            "rejecting connection to unsupported path %r from %s",
            request_path,
            websocket.remote_address,
        )
        await websocket.close(code=1008, reason="unsupported path")
        return

    session_id = extract_session_id(request_path)
    meeting_dir = config.storage.recordings_dir / session_id
    try:
        meeting_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("cannot create %s: %s", meeting_dir, exc)
        await websocket.close(code=1011, reason="recording storage unavailable")
        return

    logger.info("session %s started, recording to %s", session_id, meeting_dir)

    recorders: dict[str, OpusParticipantRecorder] = {}
    frames = 0
    malformed = 0

    try:
        async for message in websocket:
            if isinstance(message, str):
                write_metadata(meeting_dir, message)
                continue

            frames += 1
            try:
                participant_id, payload = split_frame(message)
            except StreamError as exc:
                malformed += 1
                logger.warning("session %s: discarding frame: %s", session_id, exc)
                continue

            recorder = recorders.get(participant_id)
            if recorder is None:
                wav_path = meeting_dir / f"participant-{participant_id}.wav"
                try:
                    recorder = OpusParticipantRecorder(wav_path)
                except (OpusError, OSError) as exc:
                    logger.error(
                        "session %s: cannot record participant %s: %s",
                        session_id,
                        participant_id,
                        exc,
                    )
                    continue
                recorders[participant_id] = recorder
                logger.info("session %s: recording participant %s", session_id, participant_id)

            recorder.write_packet(payload)
    except websockets.ConnectionClosed:
        logger.info("session %s: peer disconnected", session_id)
    finally:
        for participant_id, recorder in recorders.items():
            recorder.close()
            logger.info(
                "session %s: participant %s recorded %.1fs (%d packets dropped)",
                session_id,
                participant_id,
                recorder.duration_seconds,
                recorder.dropped_packets,
            )

        logger.info(
            "session %s finished: %d frames, %d malformed, %d participants",
            session_id,
            frames,
            malformed,
            len(recorders),
        )

        # Only run the pipeline when there is actually something to process:
        # a connection that sent nothing must not email an empty meeting.
        if frames and any(recorder.has_audio for recorder in recorders.values()):
            await _run_post_processing(meeting_dir, config)
        else:
            logger.info("session %s: no audio captured; skipping post-processing", session_id)


def check_storage(config: Config) -> None:
    """Fail fast if the recordings directory cannot be written to.

    Under the shipped systemd unit ``ProtectSystem=strict`` makes everything
    outside ``ReadWritePaths`` read-only, so a misconfigured or missing
    recordings directory is a realistic failure. Better to say so at startup
    than to discover it when the first meeting ends.
    """
    directory = config.storage.recordings_dir
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"cannot create the recordings directory {directory}: {exc}") from exc

    if not os.access(directory, os.W_OK | os.X_OK):
        raise ConfigError(
            f"the recordings directory {directory} is not writable by uid {os.getuid()}"
        )


async def serve(config: Config) -> None:
    """Run the WebSocket server until a termination signal arrives."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for caught in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(caught, stop.set)

    _semaphore()  # bind the semaphore to this loop before serving

    handler = functools.partial(handle_jvb_stream, config=config)
    try:
        async with websockets.serve(handler, config.server.host, config.server.port):
            logger.info(
                "listening on ws://%s:%d%s (recordings in %s)",
                config.server.host,
                config.server.port,
                WEBSOCKET_PATH,
                config.storage.recordings_dir,
            )
            await stop.wait()
    except OSError as exc:
        logger.error(
            "cannot bind %s:%d: %s (is another service already using that port?)",
            config.server.host,
            config.server.port,
            exc,
        )
        raise

    logger.info("shutting down")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jitsi-audio-bridge",
        description=(
            "Record per-participant audio from a Jitsi bridge, then transcribe, "
            "summarise and email it."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "configuration file to read; overrides $JITSI_AUDIO_BRIDGE_CONFIG "
            "and the default search path"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging verbosity (default: INFO)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a process exit status."""
    args = build_argument_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
        check_storage(config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

    try:
        asyncio.run(serve(config))
    except KeyboardInterrupt:
        logger.info("interrupted")
        return 0
    except OSError:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
