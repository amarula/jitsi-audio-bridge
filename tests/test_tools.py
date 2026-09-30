"""Unit tests for the Opus encoder and the test-environment tooling.

The encoder lives in the package because both the test suite and the tools
need to produce real Opus packets; testing it here keeps the round-trip
guarantee in one place. Nothing in this file touches the network.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from jitsi_audio_bridge.audio import OpusDecoder, OpusEncoder, OpusError
from tools.send_meeting import (
    DEFAULT_RATE,
    ENCODABLE_RATES,
    FRAME_MS,
    Participant,
    build_parser,
    build_participants,
    frame_samples,
    load_wav_frames,
    tone_frame,
)

# --------------------------------------------------------------------------
# OpusEncoder
# --------------------------------------------------------------------------


def test_encoder_output_decodes_back_to_the_right_length() -> None:
    """The round trip the whole pipeline depends on."""
    encoder = OpusEncoder(48000, 1, "voip")
    decoder = OpusDecoder(16000, 1)
    try:
        for index in range(10):
            packet = encoder.encode(tone_frame(440.0, index, 48000), 960)
            assert packet, "encoder produced an empty packet"
            decoded = decoder.decode(packet)
            assert len(decoded) // 2 == 320  # 20 ms at 16 kHz
    finally:
        encoder.close()
        decoder.close()


@pytest.mark.parametrize("rate", ENCODABLE_RATES)
def test_encoder_supports_every_libopus_rate(rate: int) -> None:
    encoder = OpusEncoder(rate, 1, "audio")
    try:
        packet = encoder.encode(tone_frame(440.0, 0, rate), frame_samples(rate))
        assert packet
    finally:
        encoder.close()


def test_encoder_rejects_an_unknown_application() -> None:
    with pytest.raises(ValueError, match="application"):
        OpusEncoder(48000, 1, "nonsense")


def test_encoder_rejects_pcm_of_the_wrong_size() -> None:
    encoder = OpusEncoder(48000, 1, "voip")
    try:
        with pytest.raises(ValueError, match="expected"):
            encoder.encode(b"\x00\x00", 960)
    finally:
        encoder.close()


def test_encoding_after_close_is_an_error() -> None:
    encoder = OpusEncoder(48000, 1, "voip")
    encoder.close()
    with pytest.raises(OpusError):
        encoder.encode(tone_frame(440.0, 0, 48000), 960)


def test_encoder_close_is_idempotent() -> None:
    encoder = OpusEncoder(48000, 1, "voip")
    encoder.close()
    encoder.close()


def test_encoder_works_as_a_context_manager() -> None:
    with OpusEncoder(48000, 1, "voip") as encoder:
        assert encoder.encode(tone_frame(440.0, 0, 48000), 960)


def test_stereo_encoder_round_trips_to_mono() -> None:
    encoder = OpusEncoder(48000, 2, "audio")
    decoder = OpusDecoder(16000, 1)
    try:
        samples = [int(8000 * (1 if i % 2 else -1)) for i in range(960 * 2)]
        packet = encoder.encode(struct.pack(f"<{len(samples)}h", *samples), 960)
        assert decoder.decode(packet)
    finally:
        encoder.close()
        decoder.close()


# --------------------------------------------------------------------------
# Frame generation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rate", ENCODABLE_RATES)
def test_frame_samples_match_the_frame_duration(rate: int) -> None:
    assert frame_samples(rate) == rate * FRAME_MS // 1000


@pytest.mark.parametrize("rate", ENCODABLE_RATES)
def test_tone_frame_has_the_expected_byte_length(rate: int) -> None:
    assert len(tone_frame(440.0, 0, rate)) == frame_samples(rate) * 2


def test_tone_frames_advance_in_phase() -> None:
    """Consecutive frames must be different, or the stream is a constant."""
    assert tone_frame(440.0, 0, 48000) != tone_frame(440.0, 1, 48000)


def test_participant_frame_field_is_fixed_width() -> None:
    short = Participant(identifier="a", name="A", email="a@example.com")
    assert len(short.frame_bytes) == 16
    assert short.frame_bytes.startswith(b"a\x00")

    long = Participant(identifier="x" * 40, name="X", email="x@example.com")
    assert len(long.frame_bytes) == 16  # truncated, never overlong


# --------------------------------------------------------------------------
# WAV loading
# --------------------------------------------------------------------------


def _write_wav(path: Path, *, rate: int, channels: int, frames: int) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        # A sawtooth well inside int16 range; the waveform does not matter,
        # only that the frames are non-constant and correctly sized.
        samples = [(index % 100 - 50) * 200 for index in range(frames * channels)]
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def test_load_wav_frames_splits_into_twenty_millisecond_chunks(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "mono.wav", rate=48000, channels=1, frames=48000)  # 1 second
    frames, rate = load_wav_frames(path)
    assert rate == 48000
    assert len(frames) == 50  # 48000 samples / 960 per frame = 1 second
    assert all(len(frame) == 960 * 2 for frame in frames)


def test_load_wav_frames_downmixes_stereo(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "stereo.wav", rate=48000, channels=2, frames=9600)
    frames, rate = load_wav_frames(path)
    assert rate == 48000
    assert len(frames) == 10
    assert all(len(frame) == 960 * 2 for frame in frames)  # mono, not 1920


def test_load_wav_frames_drops_a_partial_trailing_frame(tmp_path: Path) -> None:
    # Half a frame longer than two whole ones.
    path = _write_wav(tmp_path / "ragged.wav", rate=48000, channels=1, frames=960 * 2 + 480)
    frames, _ = load_wav_frames(path)
    assert len(frames) == 2


def test_load_wav_frames_rejects_a_rate_opus_cannot_encode(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "cd.wav", rate=44100, channels=1, frames=44100)
    with pytest.raises(SystemExit, match="not a rate libopus can encode"):
        load_wav_frames(path)


def test_load_wav_frames_rejects_eight_bit_audio(tmp_path: Path) -> None:
    path = tmp_path / "eight.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(48000)
        handle.writeframes(b"\x80" * 48000)
    with pytest.raises(SystemExit, match="16-bit"):
        load_wav_frames(path)


def test_load_wav_frames_rejects_a_file_shorter_than_one_frame(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "tiny.wav", rate=48000, channels=1, frames=10)
    with pytest.raises(SystemExit, match="shorter than one"):
        load_wav_frames(path)


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


def test_default_rate_is_a_rate_libopus_accepts() -> None:
    assert DEFAULT_RATE in ENCODABLE_RATES


def test_generated_participants_get_distinct_identifiers_and_tones() -> None:
    args = build_parser().parse_args(["--participants", "3"])
    participants = build_participants(args)
    assert len(participants) == 3
    assert len({p.identifier for p in participants}) == 3
    assert len({p.frequency for p in participants}) == 3


def test_explicit_participants_parse_all_three_fields() -> None:
    args = build_parser().parse_args(["--participant", "alice:Alice:alice@example.com"])
    (participant,) = build_participants(args)
    assert participant.identifier == "alice"
    assert participant.name == "Alice"
    assert participant.email == "alice@example.com"


def test_explicit_participant_defaults_name_and_email() -> None:
    args = build_parser().parse_args(["--participant", "bob"])
    (participant,) = build_participants(args)
    assert participant.name == "Bob"
    assert participant.email == "bob@example.com"


def test_more_wav_files_than_participants_is_rejected() -> None:
    args = build_parser().parse_args(["--participants", "1", "--wav", "a.wav", "--wav", "b.wav"])
    with pytest.raises(SystemExit, match="only 1 participants"):
        build_participants(args)


def test_realtime_is_the_default_and_fast_opts_out() -> None:
    assert build_parser().parse_args([]).realtime is True
    assert build_parser().parse_args(["--fast"]).realtime is False


def test_metadata_is_sent_by_default() -> None:
    assert build_parser().parse_args([]).metadata is True
    assert build_parser().parse_args(["--no-metadata"]).metadata is False
