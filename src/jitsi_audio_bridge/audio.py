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
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: libopus needs an application hint when a decoder is created.  Voice-over-IP
#: is the right one for speech transcription.
_OPUS_APPLICATION_VOIP = 2048

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


def parse_metadata(meeting_dir: str | Path) -> dict[str, Any]:
    """Parse room name, participants and email addresses from metadata.json.

    Always returns a usable mapping.  A missing or malformed file degrades to
    defaults rather than raising, so a broken control frame cannot cost a
    meeting its transcript.
    """
    metadata_path = Path(meeting_dir) / "metadata.json"
    info: dict[str, Any] = {"room_name": "General Meeting", "recipients": [], "id_to_name": {}}

    if not metadata_path.exists():
        logger.warning("no metadata.json in %s; using defaults", meeting_dir)
        return info

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("cannot read %s: %s", metadata_path, exc)
        return info

    if not isinstance(meta, dict):
        logger.warning("metadata.json is not a JSON object; using defaults")
        return info

    info["room_name"] = meta.get("room_name") or info["room_name"]

    recipients: set[str] = set()
    for participant in meta.get("participants", []) or []:
        if not isinstance(participant, dict):
            continue
        # Jitsi has emitted both flat and nested participant records, so both
        # shapes are accepted.
        nested = participant.get("user") or {}
        if not isinstance(nested, dict):
            nested = {}

        email = participant.get("email") or nested.get("email")
        name = participant.get("name") or nested.get("name")
        participant_id = participant.get("id") or nested.get("id")

        if email:
            recipients.add(email)
        if participant_id and name:
            info["id_to_name"][participant_id] = name

    info["recipients"] = sorted(recipients)
    return info
