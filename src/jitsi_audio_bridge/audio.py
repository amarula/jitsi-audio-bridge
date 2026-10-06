"""Opus decoding and meeting-metadata parsing.

The JVB delivers bare Opus packets — one WebSocket binary message per packet,
with no container around them.  ffmpeg cannot read that: it has an Ogg Opus
*muxer* but no raw Opus *demuxer*, so ``ffmpeg -f opus`` fails outright with
"Unknown input format: 'opus'".  Rather than wrap every packet in an Ogg
container just to hand it to a subprocess, this module binds libopus directly
through ctypes.

The binding is deliberately small.  libopus exposes a stable C ABI and the
shared library is already present on any host running ffmpeg or a Jitsi stack,
so this costs no third-party dependency — which matters, because the obvious
Python wrapper (``opuslib``) was last released in 2018 and its maintained fork
does not yet claim support for the Python version this project targets.

No network access happens in this module.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import wave
from array import array
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

#: The control frame's payload, inside a meeting directory.
METADATA_FILENAME = "metadata.json"

#: Audio produced by live capture, one file per participant.
PARTICIPANT_GLOB = "participant-*.wav"

#: Audio written by Jitsi's own recording, typically named after the speaker.
PARTICIPANT_AUDIO_GLOB = "*_audio.wav"

#: The file ffmpeg extracts a single master track into.
EXTRACTED_AUDIO_NAME = "extracted_audio.wav"

#: Who spoke when, as captured while the meeting was running.
TIMELINE_FILENAME = "timeline.json"

#: Where the post-processing step cuts a participant's turns out, before
#: transcribing them one by one.  Removed as soon as they have been read.
TURN_DIR_NAME = ".turns"

#: Containers a master recording may arrive in, in preference order.
MASTER_MEDIA_SUFFIXES = (".wav", ".mp4", ".m4a", ".mkv")

#: libopus needs an application hint when a decoder is created.  Voice-over-IP
#: is the right one for speech transcription.
_OPUS_APPLICATION_VOIP = 2048

#: libopus status code for success, and the control request used to set a
#: bitrate through the variadic opus_encoder_ctl.
OPUS_OK = 0
_OPUS_SET_BITRATE_REQUEST = 4002

#: An Opus frame is at most 120 ms.  At 48 kHz that is 5760 samples, so a
#: buffer this large can hold any packet libopus will ever hand back.  A
#: smaller ``frame_size`` makes ``opus_decode`` return OPUS_BUFFER_TOO_SMALL.
_MAX_FRAME_SAMPLES_48K = 5760

#: Candidate sonames, tried in order if ``find_library`` comes up empty.
_LIBRARY_NAMES = ("libopus.so.0", "libopus.so", "libopus.0.dylib", "opus")

_BYTES_PER_SAMPLE = 2  # signed 16-bit PCM, what Whisper wants


class OpusError(RuntimeError):
    """Raised when libopus is unavailable or a decoder cannot be created."""


class _LibOpus:
    """A loaded libopus shared library with its prototypes declared.

    Wrapping the library in a class keeps every ctypes call in one place and
    lets tests inject a stub.
    """

    def __init__(self, library: Any | None = None) -> None:
        self._lib = library if library is not None else self._load()
        if library is None:
            self._declare_prototypes()

    @staticmethod
    def _load() -> Any:
        attempted: list[str] = []
        found = ctypes.util.find_library("opus")
        candidates = [found, *_LIBRARY_NAMES] if found else list(_LIBRARY_NAMES)

        for name in candidates:
            if not name or name in attempted:
                continue
            attempted.append(name)
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue

        raise OpusError(
            "could not load the libopus shared library (tried: "
            + ", ".join(attempted)
            + "); install it with 'apt install libopus0', or set the library "
            "path so that ctypes.util.find_library('opus') can locate it"
        )

    def _declare_prototypes(self) -> None:
        lib = self._lib

        lib.opus_decoder_create.restype = ctypes.c_void_p
        lib.opus_decoder_create.argtypes = [
            ctypes.c_int,  # Fs
            ctypes.c_int,  # channels
            ctypes.POINTER(ctypes.c_int),  # error out-param
        ]

        lib.opus_decode.restype = ctypes.c_int
        lib.opus_decode.argtypes = [
            ctypes.c_void_p,  # decoder state
            ctypes.POINTER(ctypes.c_ubyte),  # packet
            ctypes.c_int,  # packet length
            ctypes.POINTER(ctypes.c_int16),  # pcm out
            ctypes.c_int,  # frame_size
            ctypes.c_int,  # decode_fec
        ]

        lib.opus_decoder_destroy.restype = None
        lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]

        lib.opus_encoder_create.restype = ctypes.c_void_p
        lib.opus_encoder_create.argtypes = [
            ctypes.c_int,  # Fs
            ctypes.c_int,  # channels
            ctypes.c_int,  # application
            ctypes.POINTER(ctypes.c_int),  # error out-param
        ]

        lib.opus_encode.restype = ctypes.c_int
        lib.opus_encode.argtypes = [
            ctypes.c_void_p,  # encoder state
            ctypes.POINTER(ctypes.c_int16),  # pcm in
            ctypes.c_int,  # frame_size
            ctypes.POINTER(ctypes.c_ubyte),  # packet out
            ctypes.c_int,  # max packet size
        ]

        lib.opus_encoder_destroy.restype = None
        lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]

        # opus_encoder_ctl is variadic: only the two fixed parameters are
        # declared, and the request's argument is appended at the call site.
        lib.opus_encoder_ctl.restype = ctypes.c_int
        lib.opus_encoder_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int]

        lib.opus_strerror.restype = ctypes.c_char_p
        lib.opus_strerror.argtypes = [ctypes.c_int]

    def create_decoder(self, sample_rate: int, channels: int) -> tuple[Any, int]:
        """Create a decoder.  Returns the handle and the libopus status code."""
        status = ctypes.c_int(0)
        handle = self._lib.opus_decoder_create(sample_rate, channels, ctypes.byref(status))
        return handle, status.value

    def destroy_decoder(self, handle: Any) -> None:
        if handle:
            self._lib.opus_decoder_destroy(handle)

    def create_encoder(self, sample_rate: int, channels: int, application: int) -> tuple[Any, int]:
        """Create an encoder.  Returns the handle and the libopus status code."""
        status = ctypes.c_int(0)
        handle = self._lib.opus_encoder_create(
            sample_rate, channels, application, ctypes.byref(status)
        )
        return handle, status.value

    def destroy_encoder(self, handle: Any) -> None:
        if handle:
            self._lib.opus_encoder_destroy(handle)

    def encode(
        self, handle: Any, source: Any, frame_samples: int, destination: Any, capacity: int
    ) -> int:
        """Encode one frame of PCM.  Returns the packet length, or < 0."""
        return self._lib.opus_encode(handle, source, frame_samples, destination, capacity)

    def set_bitrate(self, handle: Any, bitrate: int) -> int:
        """Apply OPUS_SET_BITRATE_REQUEST.  Returns the libopus status code."""
        return self._lib.opus_encoder_ctl(handle, _OPUS_SET_BITRATE_REQUEST, ctypes.c_int(bitrate))

    def decode(
        self, handle: Any, source: Any, length: int, destination: Any, capacity: int
    ) -> int:
        """Decode one packet in place.  Returns samples per channel, or < 0."""
        return self._lib.opus_decode(handle, source, length, destination, capacity, 0)

    def error_text(self, code: int) -> str:
        raw = self._lib.opus_strerror(code)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", "replace")
        return f"unknown libopus error {code}"


_LIBRARY: _LibOpus | None = None


def _libopus() -> _LibOpus:
    """Return the process-wide libopus binding, loading it on first use."""
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = _LibOpus()
        logger.debug("libopus loaded")
    return _LIBRARY


class OpusDecoder:
    """A single libopus decoder instance.

    Decoders hold inter-packet state (the previous frame, for loss
    concealment), so one instance must be used per participant and never
    shared.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        if channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {channels}")

        self._lib = _libopus()
        self.sample_rate = sample_rate
        self.channels = channels
        #: Largest frame libopus could produce at this sample rate.
        self.capacity = _MAX_FRAME_SAMPLES_48K * sample_rate // 48000
        #: Why the most recent decode failed; empty when it succeeded.
        self.last_error = ""

        self._handle, status = self._lib.create_decoder(sample_rate, channels)
        if not self._handle:
            raise OpusError(
                f"libopus could not allocate a decoder: {self._lib.error_text(status)}"
            )
        if status != 0:
            self._lib.destroy_decoder(self._handle)
            self._handle = None
            raise OpusError(
                f"libopus rejected a decoder at {sample_rate} Hz / {channels} ch: "
                f"{self._lib.error_text(status)}"
            )

        self._pcm = (ctypes.c_int16 * (self.capacity * channels))()

    def decode(self, packet: bytes) -> bytes:
        """Decode one Opus packet to interleaved 16-bit PCM.

        Returns ``b""`` and sets :attr:`last_error` if the packet is unusable,
        rather than raising: a single corrupt or non-Opus packet must not be
        able to abort a participant's whole recording.
        """
        if self._handle is None:
            raise OpusError("decoder has already been closed")
        if not packet:
            self.last_error = "empty packet"
            return b""

        source = (ctypes.c_ubyte * len(packet)).from_buffer_copy(packet)
        produced = self._lib.decode(
            self._handle, source, len(packet), self._pcm, self.capacity
        )
        if produced < 0:
            self.last_error = self._lib.error_text(produced)
            return b""

        self.last_error = ""
        sample_count = produced * self.channels
        return memoryview(self._pcm).cast("B")[: sample_count * _BYTES_PER_SAMPLE].tobytes()

    def close(self) -> None:
        if self._handle is not None:
            self._lib.destroy_decoder(self._handle)
            self._handle = None

    def __enter__(self) -> OpusDecoder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        # Nothing useful can be done about a failure during interpreter
        # shutdown, so this only exists to release the decoder handle.
        with contextlib.suppress(Exception):
            self.close()


class OpusEncoder:
    """A single libopus encoder instance.

    The daemon only ever decodes; this exists so that the test suite and the
    test environment in ``tools/`` can generate real Opus packets instead of
    committing audio fixtures.  It shares the binding above rather than
    declaring a second set of ctypes prototypes.
    """

    #: Opus will not emit a single frame larger than this; libopus's own
    #: recommendation for the output buffer.
    MAX_PACKET_BYTES = 4000

    #: Encoder tuning hints, as libopus names them.
    APPLICATIONS = {
        "voip": 2048,
        "audio": 2049,
        "restricted_lowdelay": 2051,
    }

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        application: str = "voip",
        bitrate: int = 24000,
    ) -> None:
        if application not in self.APPLICATIONS:
            known = ", ".join(sorted(self.APPLICATIONS))
            raise ValueError(f"unknown application {application!r}; expected one of {known}")

        self._lib = _libopus()
        self.sample_rate = sample_rate
        self.channels = channels
        self.application = application

        self._handle, status = self._lib.create_encoder(
            sample_rate, channels, self.APPLICATIONS[application]
        )
        if not self._handle:
            raise OpusError(
                f"libopus could not allocate an encoder: {self._lib.error_text(status)}"
            )
        if status != 0:
            self._lib.destroy_encoder(self._handle)
            self._handle = None
            raise OpusError(
                f"libopus rejected an encoder at {sample_rate} Hz / {channels} ch: "
                f"{self._lib.error_text(status)}"
            )

        if bitrate:
            status = self._lib.set_bitrate(self._handle, bitrate)
            if status != OPUS_OK:
                self.close()
                raise OpusError(
                    f"libopus rejected a bitrate of {bitrate}: {self._lib.error_text(status)}"
                )
        self.bitrate = bitrate

    def encode(self, pcm: bytes, frame_samples: int) -> bytes:
        """Encode interleaved 16-bit PCM into one Opus packet.

        ``frame_samples`` is per channel; the returned bytes are a complete
        packet, ready to be prefixed with a participant identifier and sent.
        """
        if self._handle is None:
            raise OpusError("encoder has already been closed")

        expected = frame_samples * self.channels * _BYTES_PER_SAMPLE
        if len(pcm) != expected:
            raise ValueError(
                f"expected {expected} bytes of PCM for {frame_samples} samples "
                f"across {self.channels} channel(s), got {len(pcm)}"
            )

        source = (ctypes.c_int16 * (frame_samples * self.channels)).from_buffer_copy(pcm)
        destination = (ctypes.c_ubyte * self.MAX_PACKET_BYTES)()
        written = self._lib.encode(
            self._handle, source, frame_samples, destination, self.MAX_PACKET_BYTES
        )
        if written < 0:
            raise OpusError(f"libopus could not encode the frame: {self._lib.error_text(written)}")

        return bytes(destination[:written])

    def close(self) -> None:
        if self._handle is not None:
            self._lib.destroy_encoder(self._handle)
            self._handle = None

    def __enter__(self) -> OpusEncoder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        with contextlib.suppress(Exception):
            self.close()


def _rms(pcm: bytes) -> float:
    """Root-mean-square of interleaved 16-bit PCM, as a fraction of full scale.

    ``audioop`` would do this and is gone from Python 3.13, so the arithmetic
    is here instead: one packet is a few hundred samples, and the sum of their
    squares stays exact in a float.
    """
    if not pcm:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % _BYTES_PER_SAMPLE])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    total = sum(value * value for value in samples)
    return math.sqrt(total / len(samples)) / 32768.0


class OpusParticipantRecorder:
    """Decodes one participant's Opus stream into a mono WAV file.

    Not safe to share between participants: it owns a stateful decoder.
    """

    def __init__(
        self,
        wav_path: str | Path,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self.wav_path = Path(wav_path)
        self.sample_rate = sample_rate
        self.channels = channels
        #: Samples per channel successfully written.
        self.decoded_samples = 0
        #: Packets libopus refused; a few are normal (comfort noise, RED).
        self.dropped_packets = 0
        #: RMS of the last decoded packet, 0.0 to 1.0 of full scale.  Kept for
        #: the timeline: whether a packet is speech has to be decided while the
        #: audio is in hand, and this is the same decode the WAV needed.
        self.last_level: float | None = None
        self._closed = False

        self._decoder = OpusDecoder(sample_rate, channels)
        # Deliberately not a context manager: the WAV stays open for the whole
        # recording and is finalised by close().
        self._wav = wave.open(str(self.wav_path), "wb")  # noqa: SIM115
        self._wav.setnchannels(channels)
        self._wav.setsampwidth(_BYTES_PER_SAMPLE)
        self._wav.setframerate(sample_rate)

    @property
    def duration_seconds(self) -> float:
        """Length of audio decoded so far."""
        return self.decoded_samples / self.sample_rate if self.sample_rate else 0.0

    @property
    def has_audio(self) -> bool:
        """Whether any audio at all was decoded."""
        return self.decoded_samples > 0

    def write_packet(self, packet: bytes) -> bool:
        """Decode and append one Opus packet.  Returns whether it was usable."""
        if self._closed:
            raise OpusError(f"recorder for {self.wav_path.name} is already closed")

        pcm = self._decoder.decode(packet)
        if not pcm:
            self.dropped_packets += 1
            logger.debug(
                "%s: dropped an Opus packet (%s)", self.wav_path.name, self._decoder.last_error
            )
            return False

        self._wav.writeframes(pcm)
        self.decoded_samples += len(pcm) // (_BYTES_PER_SAMPLE * self.channels)
        self.last_level = _rms(pcm)
        return True

    def close(self) -> None:
        """Finalise the WAV header and release the decoder.

        The ``wave`` module only writes the true frame count into the header
        here, so a process killed before ``close()`` leaves an unreadable file.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._wav.close()
        finally:
            self._decoder.close()

    def __enter__(self) -> OpusParticipantRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


#: Matches an email address anywhere in a blob of text. Used only as a last
#: resort, when the structured fields yielded nothing.
_EMAIL_IN_TEXT = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def _add_participant(
    participants: list[str], seen: set[str], name: str, email: str = ""
) -> None:
    """Add a participant to the human-readable list, once.

    This list is what the summariser is given as the people in the meeting,
    and it is told to attribute points to them.  An address is a nice thing to
    have beside a name, but it is not what makes the attribution possible: a
    deployment that authenticates nobody has names and no addresses, and a
    list built only from addresses would be empty for it.
    """
    key = name or email
    if not key or key in seen:
        return
    seen.add(key)
    if name and email:
        participants.append(f"{name} ({email})")
    else:
        participants.append(key)


def room_name_from_metadata(meta: dict[str, Any]) -> str | None:
    """Work out the room name from an explicit field, or from ``meeting_url``.

    Jitsi's metadata does not carry a ``room_name``: the meeting is identified
    by a URL such as ``https://meet.example.com/Weekly-Planning``, whose last
    path segment is the room. An explicit ``room_name`` is honoured first so
    that senders which do provide one keep working.
    """
    explicit = meta.get("room_name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    meeting_url = meta.get("meeting_url")
    if isinstance(meeting_url, str) and meeting_url.strip():
        segments = [segment for segment in urlparse(meeting_url).path.split("/") if segment]
        if segments:
            return segments[-1]
    return None


def parse_metadata(meeting_dir: str | Path) -> dict[str, Any]:
    """Parse the room name, participants and recipients from ``metadata.json``.

    Always returns a usable mapping.  A missing or malformed file degrades to
    defaults rather than raising, so a broken control frame cannot cost a
    meeting its transcript.

    Returns a dict with ``room_name``, ``recipients`` (sorted, de-duplicated),
    ``participants`` (human-readable ``"Name <email>"`` strings) and
    ``id_to_name``, which maps both participant ids and email addresses to a
    display name so that a recording can be attributed by either.
    """
    metadata_path = Path(meeting_dir) / METADATA_FILENAME
    info: dict[str, Any] = {
        "room_name": "General Meeting",
        "recipients": [],
        "participants": [],
        "id_to_name": {},
    }

    if not metadata_path.exists():
        logger.warning("no %s in %s; using defaults", METADATA_FILENAME, meeting_dir)
        return info

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("cannot read %s: %s", metadata_path, exc)
        return info

    if not isinstance(meta, dict):
        logger.warning("%s is not a JSON object; using defaults", METADATA_FILENAME)
        return info

    info["room_name"] = room_name_from_metadata(meta) or info["room_name"]

    recipients: list[str] = []
    id_to_name: dict[str, str] = {}
    seen_participants: set[str] = set()

    for participant in meta.get("participants", []) or []:
        if not isinstance(participant, dict):
            continue

        # Jitsi has emitted both flat and nested participant records, and the
        # field names have varied, so every known spelling is accepted.
        nested = participant.get("user")
        if not isinstance(nested, dict):
            nested = {}

        email = participant.get("email") or participant.get("mail") or nested.get("email")
        name = participant.get("name") or participant.get("display_name") or nested.get("name")
        name = name if isinstance(name, str) else ""
        participant_id = participant.get("id") or nested.get("id")

        if isinstance(email, str) and "@" in email:
            clean = email.strip()
            if clean and clean not in recipients:
                recipients.append(clean)
            # Attribute by email too: a recording is often named after the
            # address rather than the opaque participant id.
            if name:
                id_to_name[clean] = name
            _add_participant(info["participants"], seen_participants, name, clean)

        if isinstance(participant_id, str) and participant_id and name:
            id_to_name[participant_id] = name
            # Named but unaddressed: still a person the summary should know
            # about, and the only kind a deployment without tokens has.
            _add_participant(info["participants"], seen_participants, name)

    if not recipients:
        # Nothing structured, but an address may still be buried somewhere in
        # the document. Better to find it than to silently fall back.
        found = _EMAIL_IN_TEXT.findall(json.dumps(meta))
        if found:
            logger.info("no structured recipients; recovered %d from the raw metadata", len(found))
            recipients = list(dict.fromkeys(address.strip() for address in found))
            if not info["participants"]:
                info["participants"] = list(recipients)

    info["recipients"] = sorted(recipients)
    info["id_to_name"] = id_to_name
    return info


def discover_audio(meeting_dir: str | Path) -> tuple[list[Path], Path | None]:
    """Find the audio belonging to a finished meeting.

    Two shapes are supported, because both occur in practice: one file per
    participant (what live capture produces, and what per-speaker recording
    produces), or a single master recording of the whole room.

    Returns ``(participant_files, master_file)``. Exactly one of the two is
    non-empty; ``([], None)`` means there was nothing to process.
    """
    directory = Path(meeting_dir)

    participant_files = {
        path
        for pattern in (PARTICIPANT_GLOB, PARTICIPANT_AUDIO_GLOB)
        for path in directory.glob(pattern)
    }
    participant_files.discard(directory / EXTRACTED_AUDIO_NAME)
    if participant_files:
        return sorted(participant_files), None

    for suffix in MASTER_MEDIA_SUFFIXES:
        candidates = sorted(
            path
            for path in directory.glob(f"*{suffix}")
            if path.name != EXTRACTED_AUDIO_NAME and path.is_file()
        )
        if candidates:
            return [], candidates[0]
    return [], None


def extract_audio_track(source: str | Path, destination: str | Path) -> Path:
    """Extract a mono 16 kHz PCM track from a master recording.

    ffmpeg is used only here. Unlike raw Opus packets, a real container (mp4,
    mkv, m4a) is something ffmpeg reads natively, so this is the case it is
    genuinely the right tool for.

    Raises:
        OpusError: if ffmpeg is missing or fails.
    """
    source, destination = Path(source), Path(destination)
    if not shutil.which("ffmpeg"):
        raise OpusError(
            f"ffmpeg is needed to extract audio from {source.name}, but was not found on PATH"
        )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise OpusError(f"ffmpeg could not extract audio from {source.name}: {tail[0]}")
    logger.info("extracted %s from %s", destination.name, source.name)
    return destination


def participant_id_from_path(audio_path: str | Path) -> str | None:
    """The participant a recording is named after, or ``None``.

    Live capture names files ``participant-<id>.wav``, and the id is the
    sanitised source tag the timeline keys on.  Recordings that Jitsi's own
    recorder wrote are named after the speaker instead, so they have no id to
    match a timeline with — and no need of one.
    """
    stem = Path(audio_path).stem
    prefix = "participant-"
    return stem[len(prefix) :] if stem.startswith(prefix) else None


def slice_wav(
    source: str | Path, start: float, duration: float, destination: str | Path
) -> Path:
    """Copy *duration* seconds of *source*, from *start*, into *destination*.

    Used to cut one speaking turn out of a participant's recording so Whisper
    is handed the turn rather than the meeting.  The frame count is clamped to
    what the file holds, so a timeline that disagrees with a truncated WAV
    yields a short slice rather than an error.
    """
    source_path, destination_path = Path(source), Path(destination)
    with wave.open(str(source_path), "rb") as reader:
        params = reader.getparams()
        rate = reader.getframerate() or 1
        first = max(0, int(start * rate))
        if first >= reader.getnframes():
            raise OpusError(f"{source_path.name}: no audio at {start:.3f}s")
        reader.setpos(first)
        wanted = max(1, int(duration * rate))
        frames = reader.readframes(wanted)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination_path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)
    return destination_path


def attribute_speaker(audio_path: str | Path, id_to_name: dict[str, str]) -> str:
    """Name the speaker a recording belongs to.

    ``id_to_name`` keys are matched as substrings of the path, because a
    recording is named ``participant-<id>.wav`` or ``<email>_audio.wav`` and the
    key may be either. Falls back to the file's stem.
    """
    path = str(audio_path)
    for identifier, name in id_to_name.items():
        if identifier and identifier in path:
            return name
    return Path(path).stem
