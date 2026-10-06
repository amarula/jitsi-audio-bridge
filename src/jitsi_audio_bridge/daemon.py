"""WebSocket server that captures per-participant audio from a Jitsi bridge.

One connection corresponds to one meeting.  Two wire framings are accepted on
the same route, distinguished per frame:

* the custom binary framing — a JSON control frame describing the room and its
  participants, then binary frames each holding ``[16-byte participant id]
  [Opus packet]``;
* stock Jitsi's media-json framing, which the JVB uses for bridge-based
  transcription — JSON text frames with an ``event`` discriminator (``info``,
  ``start``, ``media``, ``ping``, ``session-end``) and base64 Opus payloads,
  tagged per participant.  See docs/jitsi-integration.md.

When the connection closes, the recorded audio is transcribed, summarised and
emailed.

This is the only module that knows about asyncio.  The blocking half of the
pipeline lives in :func:`process_completed_session` and is pushed onto a worker
thread via :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import functools
import json
import logging
import os
import re
import shutil
import signal
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import websockets

from . import __version__
from .ai_client import generate_summary, transcribe_audio
from .audio import (
    EXTRACTED_AUDIO_NAME,
    METADATA_FILENAME,
    TIMELINE_FILENAME,
    TURN_DIR_NAME,
    OpusError,
    OpusParticipantRecorder,
    attribute_speaker,
    discover_audio,
    extract_audio_track,
    parse_metadata,
    participant_id_from_path,
    slice_wav,
)
from .config import Config, ConfigError, load_config
from .mailer import send_meeting_email
from .timeline import SessionTimeline, TurnTracker, format_offset, merge_turns, utc_now

logger = logging.getLogger(__name__)

#: The only request path accepted.  The JVB is expected to connect here.
WEBSOCKET_PATH = "/transcribe"

#: Size of the participant-identifier prefix on every binary frame.
PARTICIPANT_ID_BYTES = 16

#: Session identifiers are used as directory names, so they are capped.
MAX_IDENTIFIER_LENGTH = 64

#: Used when a connection carries no usable ``sessionId``.
DEFAULT_SESSION_ID = "session_default"

#: ``event`` values of the JVB's media-json framing that this receiver acts on.
#: Every other event — ``info``, ``sources``, ``stop``, ``transcription-result``
#: or one a future bridge adds — is logged and ignored, never fatal.
MEDIA_JSON_MEDIA = "media"
MEDIA_JSON_PING = "ping"
MEDIA_JSON_START = "start"
MEDIA_JSON_SESSION_END = "session-end"

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


def parse_media_json_event(raw: str) -> dict[str, Any] | None:
    """Return the media-json event carried by *raw*, or ``None`` if it is not one.

    A frame belongs to the JVB's media-json framing when it is a JSON object
    with an ``event`` key: that key is the protocol's discriminator, and a
    legacy control frame never carries it.  Dispatch is per frame rather than
    per connection, so both senders can share the socket without negotiating.

    The value of ``event`` is deliberately not inspected here — an event whose
    name is missing or not a string must be skipped, not allowed to overwrite
    ``metadata.json`` as a control frame would.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "event" not in data:
        return None
    return data


def extract_media_json_media(event: dict[str, Any]) -> tuple[str, str, bytes]:
    """Return ``(participant id, raw tag, Opus packet)`` from a ``media`` event.

    The identifier is the source's ``tag`` — the only identity present on every
    media event, including one that arrives before its ``start`` — reduced with
    the same sanitiser as a session id, because it becomes a filename.  The raw
    tag comes back too, so two tags that reduce to one identifier can be
    reported rather than silently sharing a recording.

    Raises:
        StreamError: if there is no ``media`` object, no usable tag, or a
            payload that is missing, empty, or not standard base64.
    """
    media = event.get("media")
    if not isinstance(media, dict):
        raise StreamError("media event carries no media object")

    tag = media.get("tag")
    if not isinstance(tag, str) or not tag.strip():
        raise StreamError("media event carries no tag")
    participant_id = sanitize_identifier(tag, fallback="")
    if not participant_id:
        raise StreamError(f"source tag {tag!r} has no usable characters")

    payload = media.get("payload")
    if not isinstance(payload, str):
        raise StreamError(f"media event for {tag!r} carries no payload")
    try:
        # validate=True rejects non-alphabet characters and bad padding alike,
        # so a corrupted payload is counted rather than silently decoded.
        packet = base64.b64decode(payload, validate=True)
    except ValueError as exc:
        raise StreamError(f"media payload for {tag!r} is not valid base64: {exc}") from exc
    if not packet:
        raise StreamError(f"media event for {tag!r} carries an empty payload")

    return participant_id, tag, packet


def build_media_json_pong(event: dict[str, Any]) -> str | None:
    """Serialise the ``pong`` a ``ping`` event requires, or ``None`` if unusable.

    The JVB drops and reconnects a peer that does not answer within its ping
    timeout (3 s after a 10 s interval by default), so answering is not
    optional.  The reply must echo the request's ``id``, which the protocol
    defines as a natural number; a missing or non-numeric id cannot be echoed.
    ``bool`` is excluded explicitly because JSON ``true`` is an ``int`` in
    Python and would otherwise be answered as ``1``.
    """
    ping_id = event.get("id")
    if isinstance(ping_id, bool) or not isinstance(ping_id, int):
        return None
    return json.dumps({"event": "pong", "id": ping_id})


def describe_media_json_start(event: dict[str, Any]) -> str:
    """Describe a ``start`` event for the log, without trusting its shape.

    The tag is what recordings are keyed by and ``endpointId`` is the closest
    thing to a participant identity this protocol carries, so both are logged
    together to make later correlation possible.  The encoding is included so a
    source that does not announce Opus is visible without another code path.
    """
    start = event.get("start") if isinstance(event.get("start"), dict) else {}
    custom = start.get("customParameters")
    custom = custom if isinstance(custom, dict) else {}
    media_format = start.get("mediaFormat")
    media_format = media_format if isinstance(media_format, dict) else {}
    return (
        f"source {start.get('tag', '?')!r} "
        f"(endpoint {custom.get('endpointId', '?')!r}, "
        f"{media_format.get('encoding', '?')} "
        f"{media_format.get('sampleRate', '?')} Hz, "
        f"{media_format.get('channels', '?')} ch)"
    )


def extract_media_json_vad(event: dict[str, Any]) -> bool | None:
    """The exporter's own voice-activity flag, when it set one.

    It is absent on most export streams, which is why the level of the decoded
    audio is the fallback — see
    :class:`~jitsi_audio_bridge.timeline.TurnTracker`.
    """
    media = event.get("media")
    if not isinstance(media, dict):
        return None
    vad = media.get("vad")
    return vad if isinstance(vad, bool) else None


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
    temporary: str | None = None
    try:
        # mkstemp is inside the guard too: a meeting directory that cannot be
        # written must return False rather than raise out of the connection
        # handler, which only catches ConnectionClosed.
        handle_fd, temporary = tempfile.mkstemp(
            dir=meeting_dir, prefix=".metadata-", suffix=".tmp"
        )
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.replace(temporary, target)
    except OSError as exc:
        logger.error("cannot write %s: %s", target, exc)
        if temporary is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
        return False

    return True


#: A dropped metadata file describes a meeting; anything this size is not one.
MAX_SESSION_METADATA_BYTES = 1 << 20


def adopt_session_metadata(meeting_dir: Path, source_dir: Path | None) -> Path | None:
    """Take the metadata a companion service dropped for this session.

    Stock Jitsi's framing carries no names, addresses or room name, so a
    session it produced has nothing to attribute speakers with and no
    recipient to address.  A Prosody module can write what it knows about the
    room under the meeting id — which is the session's own id — and this moves
    that file in as the session's ``metadata.json`` before post-processing
    reads it.  A ``metadata.json`` the session produced itself always wins.

    Returns the path written, or ``None`` when there was nothing to take.
    """
    if source_dir is None:
        return None
    target = meeting_dir / METADATA_FILENAME
    if target.exists():
        return None

    source = source_dir / f"{meeting_dir.name}.json"
    try:
        if source.stat().st_size > MAX_SESSION_METADATA_BYTES:
            logger.warning(
                "ignoring %s: larger than %d bytes", source, MAX_SESSION_METADATA_BYTES
            )
            return None
        raw = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("cannot read %s: %s", source, exc)
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("cannot parse %s: %s", source, exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning("%s is not a JSON object; ignoring it", source)
        return None

    if not write_metadata(meeting_dir, raw):
        return None
    logger.info(
        "adopted the session metadata %s wrote (%d participant(s))",
        source.name,
        len(parsed.get("participants") or []),
    )
    try:
        source.unlink()
    except OSError as exc:
        logger.warning("cannot remove %s: %s", source, exc)
    return target


def _semaphore() -> asyncio.Semaphore:
    """Return the processing semaphore, binding it to the running loop."""
    global _PROCESSING_SEMAPHORE
    if _PROCESSING_SEMAPHORE is None:
        _PROCESSING_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    return _PROCESSING_SEMAPHORE


def _cleanup(paths: Iterable[Path | None]) -> None:
    """Delete processed files.  Only ever called when explicitly enabled.

    ``None`` is a legitimate entry: the caller passes the extraction it may
    never have made, and an ``AttributeError`` here would abandon the rest of
    the cleanup.
    """
    for path in paths:
        if path is None:
            continue
        try:
            path.unlink()
            logger.info("removed %s", path.name)
        except OSError as exc:
            logger.warning("could not remove %s: %s", path, exc)


#: How many speaking turns one participant is transcribed in.  The request
#: count is what costs, so past this the closest turns are merged — the order
#: survives, only the resolution drops — rather than falling back to a
#: monologue, which is the shape this whole feature exists to replace.
MAX_TURNS_PER_FILE = 200

#: The same budget for a session, shared out over the participants that spoke.
MAX_TURNS_PER_SESSION = 600

#: One line of the transcript: when it was said (seconds into the session, or
#: ``None`` for a recording the timeline does not cover), who, and what.
TranscriptLine = tuple[float | None, str, str]


def render_transcript(lines: Sequence[TranscriptLine]) -> str:
    """One document, in the order people spoke.

    Lines that carry a session offset are ordered by it and prefixed with it;
    lines that could not be placed — recordings from a meeting the timeline
    does not describe, such as one processed in batch mode — keep the original
    shape and follow them.
    """
    timed = sorted(
        (line for line in lines if line[0] is not None), key=lambda line: line[0] or 0.0
    )
    rendered = [
        f"[{format_offset(start)}] {speaker}: {text}" for start, speaker, text in timed
    ]
    rendered.extend(
        f"[{speaker}]: {text}" for start, speaker, text in lines if start is None
    )
    return "\n\n".join(rendered)


#: How long to wait before giving the failed turns a second chance.  Their
#: retries are seconds apart, which covers a service that hiccups; a service
#: that is busy for longer needs the whole pass to finish first.
RETRY_PASS_PAUSE_SECONDS = 10.0


def _transcribe_turns(
    recording: Path,
    speaker: str,
    turns: Sequence[Any],
    turns_dir: Path,
    config: Config,
) -> tuple[list[TranscriptLine], list[tuple[Any, Path]]]:
    """Cut one participant's speaking turns out and transcribe them.

    Returns the lines it got and the turns it could not, slice and all, so the
    caller can tell a participant who said little from a service that was
    refusing everything and give the lost ones another try later.
    """
    lines: list[TranscriptLine] = []
    lost: list[tuple[Any, Path]] = []
    for index, turn in enumerate(turns):
        slice_path = turns_dir / f"{recording.stem}-{index:04d}.wav"
        try:
            slice_wav(recording, turn.offset, turn.duration, slice_path)
        except (OpusError, OSError) as exc:
            logger.warning("cannot cut %s at %.3fs: %s", recording.name, turn.offset, exc)
            continue
        text = transcribe_audio(slice_path, config.whisper)
        if text:
            lines.append((turn.start, speaker, text))
        else:
            logger.warning("no transcript produced for turn %d of %s", index, speaker)
            lost.append((turn, slice_path))
    return lines, lost


def transcribe_recordings(
    participant_files: Sequence[Path],
    metadata: dict[str, Any],
    timeline: SessionTimeline | None,
    config: Config,
) -> list[TranscriptLine]:
    """Transcribe every recording, turn by turn when the timeline allows it.

    Falls back to whole-file transcription — the shape the transcripts had
    before any of this — whenever the timeline does not cover a recording or a
    session simply has too many turns to split.
    """
    interleave = config.transcript.interleave and timeline is not None
    allowance = MAX_TURNS_PER_FILE
    if interleave and timeline is not None:
        speakers = [participant_id_from_path(path) for path in participant_files]
        wanted = sum(len(timeline.turns_for(participant_id or "")) for participant_id in speakers)
        if wanted > MAX_TURNS_PER_SESSION:
            speaking = sum(
                1 for participant_id in speakers if timeline.turns_for(participant_id or "")
            )
            allowance = max(1, MAX_TURNS_PER_SESSION // max(1, speaking))
            logger.info(
                "%d speaking turns in this session; keeping the %d longest-lived ones per "
                "participant so the transcript stays in order",
                wanted,
                allowance,
            )

    turns_dir = (participant_files[0].parent if participant_files else Path(".")) / TURN_DIR_NAME
    lines: list[TranscriptLine] = []
    #: Turns whose request failed, kept for a second pass once everything else
    #: has been tried: an intermittent service is usually back by then, and the
    #: slices are still on disk.
    retry: list[tuple[str, Any, Path]] = []
    try:
        for recording in participant_files:
            speaker = attribute_speaker(recording, metadata["id_to_name"])
            turns: Sequence[Any] = []
            if interleave and timeline is not None:
                turns = timeline.turns_for(participant_id_from_path(recording) or "")
                if len(turns) > allowance:
                    logger.info(
                        "%s has %d speaking turns; merging the closest to %d",
                        speaker,
                        len(turns),
                        allowance,
                    )
                    turns = merge_turns(list(turns), allowance)
            pending: list[tuple[str, Any, Path]] = []
            if turns:
                logger.info("transcribing %d turn(s) of %s", len(turns), speaker)
                turn_lines, lost = _transcribe_turns(
                    recording, speaker, turns, turns_dir, config
                )
                if turn_lines:
                    lines.extend(turn_lines)
                    retry.extend((speaker, turn, slice_path) for turn, slice_path in lost)
                    continue
                if lost:
                    # Every turn failed, which says more about the service than
                    # about this participant: hand it the whole recording, one
                    # request, the way it worked before turns existed.  If that
                    # fails too, the slices are still on disk and the retry pass
                    # gets them.
                    logger.warning(
                        "none of %s's %d turn(s) produced text; transcribing the recording "
                        "whole instead",
                        speaker,
                        len(turns),
                    )
                    pending = [(speaker, turn, slice_path) for turn, slice_path in lost]

            text = transcribe_audio(recording, config.whisper)
            if text:
                lines.append((None, speaker, text))
            else:
                logger.warning("no transcript produced for %s", recording.name)
                retry.extend(pending)
        if retry:
            logger.info(
                "%d turn(s) produced no text; giving them a second chance in %.0fs",
                len(retry),
                RETRY_PASS_PAUSE_SECONDS,
            )
            time.sleep(RETRY_PASS_PAUSE_SECONDS)
            recovered = 0
            for speaker, turn, slice_path in retry:
                text = transcribe_audio(slice_path, config.whisper)
                if text:
                    lines.append((turn.start, speaker, text))
                    recovered += 1
            logger.info(
                "recovered %d of %d lost turn(s)", recovered, len(retry)
            )
    finally:
        shutil.rmtree(turns_dir, ignore_errors=True)
    return lines


def process_completed_session(meeting_dir: Path, config: Config) -> bool:
    """Transcribe, summarise and email one finished meeting.

    Works on either shape of meeting directory: one recording per participant,
    or a single master track that gets extracted with ffmpeg first.

    Blocking: transcription and summary generation take minutes.  Always call
    this through :func:`asyncio.to_thread`, never directly from a coroutine.

    Returns whether a summary email was sent.
    """
    logger.info("post-processing %s", meeting_dir)
    meeting_dir = Path(meeting_dir)
    adopt_session_metadata(meeting_dir, config.storage.session_metadata_dir)
    metadata = parse_metadata(meeting_dir)

    participant_files, master_file = discover_audio(meeting_dir)
    extracted: Path | None = None
    if not participant_files and master_file is not None:
        try:
            extracted = extract_audio_track(master_file, meeting_dir / EXTRACTED_AUDIO_NAME)
        except OpusError as exc:
            logger.error("%s", exc)
            return False
        participant_files = [extracted]
    elif not participant_files:
        logger.info("no audio in %s; nothing to do", meeting_dir)
        return False

    logger.info("transcribing %d audio file(s)", len(participant_files))
    timeline = SessionTimeline.load(meeting_dir / TIMELINE_FILENAME)
    if timeline is None:
        logger.info(
            "no %s in %s; the transcript keeps the per-participant shape",
            TIMELINE_FILENAME,
            meeting_dir,
        )
    transcript_lines = transcribe_recordings(participant_files, metadata, timeline, config)

    if not transcript_lines:
        logger.warning("nothing was transcribed for %s; skipping the email", meeting_dir)
        return False

    transcript = render_transcript(transcript_lines)
    transcript_path = meeting_dir / "transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    logger.info(
        "wrote %s (%d speaker(s), %d line(s))",
        transcript_path.name,
        len({speaker for _, speaker, _ in transcript_lines}),
        len(transcript_lines),
    )

    summary = generate_summary(
        transcript,
        metadata["room_name"],
        metadata["participants"],
        config.ollama,
    )
    if not summary:
        logger.warning("no summary was produced for %s; the transcript is kept", meeting_dir)
        return False

    summary_path = meeting_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("wrote %s", summary_path.name)

    sent = send_meeting_email(
        metadata["recipients"],
        metadata["room_name"],
        summary,
        transcript_path,
        summary_path,
        config.smtp,
    )

    if sent and config.storage.cleanup_after_send:
        # Only after a confirmed send, and only when asked for: these files are
        # the only copy of the meeting.
        logger.info("cleanup_after_send is enabled; removing the processed files")
        _cleanup([*participant_files, extracted, transcript_path, summary_path])

    return sent


def process_directory(meeting_dir: Path, config: Config) -> int:
    """Process one meeting directory that already exists on disk.

    The batch counterpart to the WebSocket server, for recordings produced by
    something else. Returns a process exit status.
    """
    if not meeting_dir.is_dir():
        logger.error("not a directory: %s", meeting_dir)
        return 2

    try:
        sent = process_completed_session(meeting_dir, config)
    except Exception:
        logger.exception("processing failed for %s", meeting_dir)
        return 1
    return 0 if sent else 1


def _schedule_finalisation(state: SessionState, config: Config) -> None:
    """Post-process this session once it has been quiet long enough."""
    if state.pending is not None:
        state.pending.cancel()
    state.pending = asyncio.create_task(_finalise_later(state, config))


async def _finalise_later(state: SessionState, config: Config) -> None:
    """Wait for the meeting to be over, then process it.

    A new connection cancels this — see :func:`_session_state` — and a
    shutdown sets ``_SHUTDOWN``, which ends the wait early rather than losing
    a meeting that has just finished.
    """
    grace = config.storage.session_grace_seconds
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(_SHUTDOWN.wait(), timeout=grace)
    state.pending = None
    _SESSIONS.pop(state.session_id, None)
    if not any(seconds > 0 for seconds in state.recorded.values()):
        logger.info("session %s: no audio captured; nothing to process", state.session_id)
        return
    logger.info(
        "session %s: quiet for %ds; processing the meeting (%.1fs of audio)",
        state.session_id,
        int(grace),
        sum(state.recorded.values()),
    )
    await _run_post_processing(state.meeting_dir, config)


async def _run_post_processing(meeting_dir: Path, config: Config) -> None:
    """Run the blocking pipeline off the event loop, bounded by a semaphore."""
    async with _semaphore():
        try:
            await asyncio.to_thread(process_completed_session, meeting_dir, config)
        except Exception:
            # A failed pipeline must not propagate into the connection handler.
            logger.exception("post-processing failed for %s", meeting_dir)


@dataclass
class SessionState:
    """One meeting, however many connections it takes.

    The JVB ends an export and opens a new one for the same meeting id when a
    transcriber is restarted, and a bridge reconnect does the same; the
    connection is therefore not the meeting.  This is what survives between
    them — the clock, the turns heard so far, and how much audio each part
    holds — so that a transcript is only made once the meeting is really over.
    """

    session_id: str
    meeting_dir: Path
    started_monotonic: float
    started_at: str
    tracker: TurnTracker
    #: Recording key -> seconds of audio in it, accumulated per part.
    recorded: dict[str, float] = field(default_factory=dict)
    saw_media_json: bool = False
    #: The grace timer holding off post-processing, if one is running.
    pending: asyncio.Task[None] | None = None


#: Sessions whose connections ended but whose meetings may not be over.
_SESSIONS: dict[str, SessionState] = {}

#: Set when the daemon is shutting down, so a pending grace timer stops
#: waiting and processes its meeting instead of losing it.
_SHUTDOWN = asyncio.Event()


def _session_state(session_id: str, meeting_dir: Path, config: Config) -> SessionState:
    """Resume the session *session_id*, or start it.

    A connection arriving while a grace timer is running means the meeting is
    still going: the timer is cancelled and the same state carries on, so the
    transcript covers the whole meeting rather than the part that happened to
    fit in one connection.
    """
    state = _SESSIONS.get(session_id)
    if state is not None:
        if state.pending is not None:
            state.pending.cancel()
            state.pending = None
            logger.info(
                "session %s: a new connection arrived; the meeting continues", session_id
            )
        return state

    state = SessionState(
        session_id=session_id,
        meeting_dir=meeting_dir,
        started_monotonic=time.monotonic(),
        started_at=utc_now(),
        tracker=TurnTracker(merge_gap=config.transcript.merge_gap_seconds),
    )
    _SESSIONS[session_id] = state
    return state


def _free_wav_path(meeting_dir: Path, participant_id: str) -> Path:
    """A recording path no earlier connection is already using.

    ``wave.open`` truncates, so a reconnect that reused the first file would
    destroy the audio recorded before it — and the timeline, whose offsets are
    per file, would then name the wrong audio.  Each connection gets its own
    part instead, and the parts are transcribed as separate recordings of the
    same speaker.
    """
    path = meeting_dir / f"participant-{participant_id}.wav"
    part = 1
    while path.exists():
        part += 1
        path = meeting_dir / f"participant-{participant_id}-{part}.wav"
    return path


def _recorder_for(
    recorders: dict[str, OpusParticipantRecorder],
    meeting_dir: Path,
    participant_id: str,
    session_id: str,
) -> OpusParticipantRecorder | None:
    """Return the recorder for *participant_id*, creating the WAV on first use.

    Shared by both framings.  Returns ``None`` when the file cannot be created:
    the packet is skipped and the session continues, because one unwritable
    participant must not cost the others their audio.
    """
    recorder = recorders.get(participant_id)
    if recorder is not None:
        return recorder

    wav_path = _free_wav_path(meeting_dir, participant_id)
    try:
        recorder = OpusParticipantRecorder(wav_path)
    except (OpusError, OSError) as exc:
        logger.error(
            "session %s: cannot record participant %s: %s",
            session_id,
            participant_id,
            exc,
        )
        return None
    recorders[participant_id] = recorder
    logger.info("session %s: recording participant %s", session_id, participant_id)
    return recorder


async def handle_jvb_stream(websocket: Any, config: Config) -> None:
    """Serve one meeting's WebSocket connection.

    Frames are dispatched by shape, so the custom binary sender and the JVB's
    media-json framing can both be served without negotiating a mode: a
    binary frame is ``[16-byte id][Opus]``, a text frame carrying an ``event``
    key is a media-json event, and any other text frame is a control frame.
    """
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

    # The meeting outlives this connection: a reconnect resumes the clock and
    # the turns rather than starting a second meeting in the same directory.
    state = _session_state(session_id, meeting_dir, config)

    recorders: dict[str, OpusParticipantRecorder] = {}
    #: Sanitised identifier -> the raw tag that claimed it, so two tags
    #: reducing to one filename are reported instead of silently merging.
    source_tags: dict[str, str] = {}
    frames = 0
    malformed = 0

    try:
        async for message in websocket:
            if isinstance(message, str):
                event = parse_media_json_event(message)
                if event is None:
                    # A legacy control frame, exactly as before.
                    if not write_metadata(meeting_dir, message):
                        malformed += 1
                    continue

                kind = event.get("event")
                if not isinstance(kind, str):
                    malformed += 1
                    logger.warning(
                        "session %s: media-json event without a usable name: %.120r",
                        session_id,
                        message,
                    )
                    continue

                if kind == MEDIA_JSON_PING:
                    pong = build_media_json_pong(event)
                    if pong is None:
                        malformed += 1
                        logger.warning(
                            "session %s: ignoring a ping without a natural-number id",
                            session_id,
                        )
                    else:
                        await websocket.send(pong)
                    continue

                if kind == MEDIA_JSON_SESSION_END:
                    # The bridge closes right after this; finalise now rather
                    # than waiting for the close to be observed.
                    logger.info("session %s: the bridge ended the session", session_id)
                    break

                if kind == MEDIA_JSON_START:
                    logger.info("session %s: %s", session_id, describe_media_json_start(event))
                    continue

                if kind != MEDIA_JSON_MEDIA:
                    if kind == "info":
                        logger.info(
                            "session %s: bridge %s %s (region %s)",
                            session_id,
                            event.get("application", "?"),
                            event.get("version", "?"),
                            event.get("region", "?"),
                        )
                    else:
                        logger.debug("session %s: ignoring %s event", session_id, kind)
                    continue

                frames += 1
                try:
                    participant_id, raw_tag, payload = extract_media_json_media(event)
                except StreamError as exc:
                    malformed += 1
                    logger.warning("session %s: discarding media event: %s", session_id, exc)
                    continue

                claimed = source_tags.setdefault(participant_id, raw_tag)
                if claimed != raw_tag:
                    logger.warning(
                        "session %s: tags %r and %r both record as %r; their audio shares a file",
                        session_id,
                        claimed,
                        raw_tag,
                        participant_id,
                    )
            else:
                frames += 1
                try:
                    participant_id, payload = split_frame(message)
                except StreamError as exc:
                    malformed += 1
                    logger.warning("session %s: discarding frame: %s", session_id, exc)
                    continue

            recorder = _recorder_for(recorders, meeting_dir, participant_id, session_id)
            if recorder is None:
                continue
            # The file offset is read before the packet lands in the file, and
            # the level after it was decoded — the same decode the WAV needed,
            # so nothing is decoded twice.
            file_offset = recorder.decoded_samples / recorder.sample_rate
            if recorder.write_packet(payload):
                media_json = isinstance(message, str)
                state.saw_media_json = state.saw_media_json or media_json
                # Keyed by the recording, not by the tag: a reconnect records
                # into its own part file, and a turn's offset only means
                # something next to the file it came from.
                state.tracker.add(
                    participant_id_from_path(recorder.wav_path) or participant_id,
                    session_offset=time.monotonic() - state.started_monotonic,
                    file_offset=file_offset,
                    duration=recorder.decoded_samples / recorder.sample_rate - file_offset,
                    level=recorder.last_level,
                    vad=extract_media_json_vad(event) if media_json else None,
                )
    except websockets.ConnectionClosed:
        logger.info("session %s: peer disconnected", session_id)
    finally:
        for participant_id, recorder in recorders.items():
            recorder.close()
            key = participant_id_from_path(recorder.wav_path) or participant_id
            state.recorded[key] = recorder.duration_seconds
            logger.info(
                "session %s: participant %s recorded %.1fs (%d packets dropped)",
                session_id,
                key,
                recorder.duration_seconds,
                recorder.dropped_packets,
            )

        logger.info(
            "session %s: connection finished: %d frames, %d malformed, %d participant(s); "
            "audio so far: %.1fs",
            session_id,
            frames,
            malformed,
            len(recorders),
            sum(state.recorded.values()),
        )

        if config.storage.capture_timeline and state.saw_media_json and state.recorded:
            turns = state.tracker.finish()
            elapsed = time.monotonic() - state.started_monotonic
            timeline = SessionTimeline(
                started_at=state.started_at,
                # A stream that arrived faster than it plays makes the wall
                # clock shorter than the meeting it describes; the turns know
                # better, so keep the larger of the two.
                duration=max(elapsed, turns[-1].end if turns else 0.0),
                recorded=dict(state.recorded),
                turns=turns,
            )
            if timeline.write(meeting_dir / TIMELINE_FILENAME):
                logger.info(
                    "session %s: %d speaking turn(s) written to %s",
                    session_id,
                    len(timeline.turns),
                    TIMELINE_FILENAME,
                )

        # A connection ending is not the meeting ending: the JVB ends an
        # export to start a new one for the same conference, and a restart
        # does the same.  Wait for quiet before transcribing and mailing.
        if sum(state.recorded.values()) > 0:
            _schedule_finalisation(state, config)
        else:
            logger.info("session %s: no audio captured yet", session_id)


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
    _SHUTDOWN.clear()

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

    # A meeting that ended moments ago is still waiting out its grace period;
    # shutting down must not lose it.
    _SHUTDOWN.set()
    pending = [state.pending for state in _SESSIONS.values() if state.pending is not None]
    if pending:
        logger.info("processing %d session(s) that were still waiting", len(pending))
        await asyncio.gather(*pending, return_exceptions=True)

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
        "--process-dir",
        metavar="PATH",
        help=(
            "process an existing meeting directory and exit, instead of serving. "
            "The directory may hold one recording per participant, or a single "
            "master recording to extract audio from."
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
        # Directory mode is handed an existing directory, so the recordings
        # directory is only required when serving.
        if args.process_dir is None:
            check_storage(config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

    if args.process_dir is not None:
        return process_directory(Path(args.process_dir), config)

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
