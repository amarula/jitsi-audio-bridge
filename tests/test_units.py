"""Unit tests for the parts of the bridge that need no Whisper, Ollama or JVB.

The Opus round-trip at the bottom exercises the real libopus binding by
encoding a tone and decoding it back through the recorder.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

from jitsi_audio_bridge import config as config_module
from jitsi_audio_bridge.audio import (
    OpusDecoder,
    OpusEncoder,
    OpusError,
    OpusParticipantRecorder,
    parse_metadata,
)
from jitsi_audio_bridge.config import ConfigError, load_config
from jitsi_audio_bridge.daemon import (
    DEFAULT_SESSION_ID,
    StreamError,
    extract_session_id,
    sanitize_identifier,
    split_frame,
    write_metadata,
)
from jitsi_audio_bridge.mailer import _subject_for, _usable_recipients, safe_attachment_name

# --------------------------------------------------------------------------
# Session and participant identifiers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "../../etc/passwd",
        "..%2f..%2fetc",
        "/absolute/path",
        "....//....//x",
        "a/b/c",
        "with spaces",
        "\x00null",
        "..",
        ".",
        "",
        "///",
    ],
)
def test_sanitize_identifier_never_yields_a_path(raw: str) -> None:
    result = sanitize_identifier(raw)
    assert result
    assert "/" not in result
    assert "\\" not in result
    assert not result.startswith(".")
    assert not result.endswith(".")
    assert result != ".."


def test_sanitize_identifier_is_bounded() -> None:
    assert len(sanitize_identifier("x" * 500)) <= 64


def test_sanitize_identifier_keeps_ordinary_values() -> None:
    assert sanitize_identifier("abc-123_XYZ.9") == "abc-123_XYZ.9"


def test_sanitize_identifier_respects_a_custom_fallback() -> None:
    assert sanitize_identifier("...", fallback="") == ""


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/transcribe?sessionId=abc123", "abc123"),
        ("/transcribe?sessionId=abc123&other=1", "abc123"),
        ("/transcribe?other=1&sessionId=abc123", "abc123"),
        ("/transcribe", DEFAULT_SESSION_ID),
        ("/transcribe?sessionId=", DEFAULT_SESSION_ID),
        ("/transcribe?sessionId=../../escape", "escape"),
    ],
)
def test_extract_session_id(path: str, expected: str) -> None:
    assert extract_session_id(path) == expected


def test_extract_session_id_takes_the_first_of_several() -> None:
    assert extract_session_id("/transcribe?sessionId=first&sessionId=second") == "first"


# --------------------------------------------------------------------------
# Binary frame splitting
# --------------------------------------------------------------------------


def _frame(participant_id: str, payload: bytes = b"\x01\x02\x03") -> bytes:
    return participant_id.encode().ljust(16, b"\x00") + payload


def test_split_frame_round_trips() -> None:
    participant_id, payload = split_frame(_frame("participant-1"))
    assert participant_id == "participant-1"
    assert payload == b"\x01\x02\x03"


def test_split_frame_accepts_exactly_seventeen_bytes() -> None:
    participant_id, payload = split_frame(b"a" * 16 + b"\xff")
    assert participant_id == "a" * 16
    assert payload == b"\xff"


@pytest.mark.parametrize("size", [0, 1, 15, 16])
def test_split_frame_rejects_frames_without_payload(size: int) -> None:
    with pytest.raises(StreamError):
        split_frame(b"a" * size)


def test_split_frame_rejects_an_empty_identifier() -> None:
    with pytest.raises(StreamError):
        split_frame(b"\x00" * 16 + b"payload")


def test_split_frame_sanitizes_a_hostile_identifier() -> None:
    participant_id, _ = split_frame(_frame("../../evil"))
    assert "/" not in participant_id


# --------------------------------------------------------------------------
# Control frames
# --------------------------------------------------------------------------


def test_write_metadata_stores_a_json_object(tmp_path: Path) -> None:
    assert write_metadata(tmp_path, json.dumps({"room_name": "Standup"}))
    assert json.loads((tmp_path / "metadata.json").read_text())["room_name"] == "Standup"


def test_write_metadata_replaces_rather_than_appends(tmp_path: Path) -> None:
    write_metadata(tmp_path, json.dumps({"room_name": "First"}))
    write_metadata(tmp_path, json.dumps({"room_name": "Second"}))
    stored = json.loads((tmp_path / "metadata.json").read_text())
    assert stored == {"room_name": "Second"}


@pytest.mark.parametrize("raw", ["not json", "[1, 2, 3]", '"a string"', ""])
def test_write_metadata_rejects_non_objects(tmp_path: Path, raw: str) -> None:
    assert not write_metadata(tmp_path, raw)
    assert not (tmp_path / "metadata.json").exists()


def test_write_metadata_leaves_no_temporary_files(tmp_path: Path) -> None:
    write_metadata(tmp_path, json.dumps({"room_name": "Standup"}))
    assert [p.name for p in tmp_path.iterdir()] == ["metadata.json"]


# --------------------------------------------------------------------------
# Metadata parsing
# --------------------------------------------------------------------------


def test_parse_metadata_reads_flat_participants(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "room_name": "Planning",
                "participants": [
                    {"id": "p1", "name": "Ada", "email": "ada@example.com"},
                    {"id": "p2", "name": "Grace", "email": "grace@example.com"},
                ],
            }
        )
    )
    meta = parse_metadata(tmp_path)
    assert meta["room_name"] == "Planning"
    assert meta["id_to_name"] == {"p1": "Ada", "p2": "Grace"}
    assert meta["recipients"] == ["ada@example.com", "grace@example.com"]


def test_parse_metadata_reads_nested_user_objects(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "room_name": "Review",
                "participants": [
                    {"user": {"id": "u1", "name": "Alan", "email": "alan@example.com"}}
                ],
            }
        )
    )
    meta = parse_metadata(tmp_path)
    assert meta["id_to_name"] == {"u1": "Alan"}
    assert meta["recipients"] == ["alan@example.com"]


def test_parse_metadata_deduplicates_recipients(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "participants": [
                    {"id": "p1", "name": "Ada", "email": "same@example.com"},
                    {"id": "p2", "name": "Grace", "email": "same@example.com"},
                ]
            }
        )
    )
    assert parse_metadata(tmp_path)["recipients"] == ["same@example.com"]


def test_parse_metadata_survives_a_missing_file(tmp_path: Path) -> None:
    meta = parse_metadata(tmp_path)
    assert meta["room_name"] == "General Meeting"
    assert meta["recipients"] == []
    assert meta["id_to_name"] == {}


def test_parse_metadata_survives_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text("{ this is not json")
    assert parse_metadata(tmp_path)["room_name"] == "General Meeting"


def test_parse_metadata_skips_junk_participants(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps({"participants": [None, "string", 42, {"id": "p1", "name": "Ada"}]})
    )
    assert parse_metadata(tmp_path)["id_to_name"] == {"p1": "Ada"}


# --------------------------------------------------------------------------
# Attachment names
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "room_name",
    ["../../../etc/passwd", "a\r\nBcc: evil@example.com", "", "/", "..", 'quote"and\\slash'],
)
def test_safe_attachment_name_is_flat_and_bounded(room_name: str) -> None:
    name = safe_attachment_name(room_name)
    assert name.endswith("_transcript.txt")
    assert "/" not in name
    assert "\\" not in name
    assert "\r" not in name
    assert "\n" not in name
    assert '"' not in name


def test_safe_attachment_name_keeps_an_ordinary_room_name() -> None:
    assert safe_attachment_name("Daily Standup") == "Daily_Standup_transcript.txt"


def test_subject_flattens_embedded_newlines() -> None:
    """A newline in the room name would otherwise raise inside EmailMessage."""
    subject = _subject_for("Standup\r\nBcc: evil@example.com")
    assert "\r" not in subject
    assert "\n" not in subject
    assert "Bcc:" in subject  # neutralised into the subject, not injected


def test_subject_falls_back_for_an_empty_room_name() -> None:
    assert _subject_for("") == "Meeting Summary: Meeting"


@pytest.mark.parametrize(
    "recipients",
    [
        ["ok@example.com\r\nBcc: evil@example.com"],
        [""],
        ["   "],
        [None],
        ["\n"],
    ],
)
def test_unusable_recipients_are_dropped(recipients: list) -> None:
    assert _usable_recipients(recipients) == []


def test_usable_recipients_keeps_ordinary_addresses() -> None:
    assert _usable_recipients(["a@example.com", " b@example.com "]) == [
        "a@example.com",
        "b@example.com",
    ]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every JITSI_AUDIO_BRIDGE_* override the host might have set."""
    for name in list(config_module.os.environ):
        if name.startswith(config_module.ENV_PREFIX):
            monkeypatch.delenv(name, raising=False)


def test_load_config_uses_defaults_when_no_file_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    monkeypatch.setattr(config_module, "SEARCH_PATHS", (tmp_path / "nothing.ini",))
    config = load_config()
    assert config.source is None
    assert config.server.port == 8080
    assert config.storage.recordings_dir == Path("/srv/recordings")
    assert config.whisper.verify_tls is True
    assert config.smtp.use_starttls is True


def test_load_config_reads_a_file(tmp_path: Path, clean_env: None) -> None:
    path = tmp_path / "config.ini"
    path.write_text(
        "[server]\nhost = 0.0.0.0\nport = 9000\n"
        "[storage]\nrecordings_dir = /tmp/rec\n"
        "[ollama]\nmodel = llama3:8b\nverify_tls = false\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.source == path
    assert (config.server.host, config.server.port) == ("0.0.0.0", 9000)
    assert config.storage.recordings_dir == Path("/tmp/rec")
    assert config.ollama.model == "llama3:8b"
    assert config.ollama.verify_tls is False
    # untouched sections keep their defaults
    assert config.whisper.timeout == 600


def test_environment_overrides_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    path = tmp_path / "config.ini"
    path.write_text("[server]\nport = 9000\n", encoding="utf-8")
    monkeypatch.setenv("JITSI_AUDIO_BRIDGE_SERVER_PORT", "9999")
    assert load_config(path).server.port == 9999


def test_environment_supplies_the_smtp_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    path = tmp_path / "config.ini"
    path.write_text("[smtp]\nuser = bridge\n", encoding="utf-8")
    monkeypatch.setenv("JITSI_AUDIO_BRIDGE_SMTP_PASSWORD", "s3cret")
    assert load_config(path).smtp.password == "s3cret"


def test_password_containing_a_percent_sign_is_literal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """Interpolation must stay off, or '%' in a password breaks configparser."""
    path = tmp_path / "config.ini"
    path.write_text("[smtp]\npassword = 100%sure\n", encoding="utf-8")
    assert load_config(path).smtp.password == "100%sure"


def test_explicit_path_that_does_not_exist_is_an_error(tmp_path: Path, clean_env: None) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.ini")


def test_environment_config_path_is_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    path = tmp_path / "from-env.ini"
    path.write_text("[server]\nport = 7000\n", encoding="utf-8")
    monkeypatch.setenv(config_module.ENV_CONFIG_PATH, str(path))
    assert load_config().server.port == 7000


@pytest.mark.parametrize("value", ["not-a-number", "8080.5", ""])
def test_a_malformed_port_names_the_offending_setting(
    tmp_path: Path, clean_env: None, value: str
) -> None:
    path = tmp_path / "config.ini"
    path.write_text(f"[server]\nport = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "server" in str(excinfo.value)
    assert "port" in str(excinfo.value)


@pytest.mark.parametrize("value", ["0", "65536", "-1"])
def test_an_out_of_range_port_is_rejected(tmp_path: Path, clean_env: None, value: str) -> None:
    path = tmp_path / "config.ini"
    path.write_text(f"[server]\nport = {value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="range"):
        load_config(path)


def test_a_malformed_boolean_is_rejected(tmp_path: Path, clean_env: None) -> None:
    path = tmp_path / "config.ini"
    path.write_text("[whisper]\nverify_tls = maybe\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="boolean"):
        load_config(path)


def test_an_empty_required_value_is_rejected(tmp_path: Path, clean_env: None) -> None:
    path = tmp_path / "config.ini"
    path.write_text("[whisper]\nurl =\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must not be empty"):
        load_config(path)


# --------------------------------------------------------------------------
# Opus round-trip against the real libopus binding
# --------------------------------------------------------------------------


def _encode_tone(
    frames: int, frame_samples: int = 960, rate: int = 48000, channels: int = 1
) -> list[bytes]:
    """Encode a 440 Hz tone as a list of discrete Opus packets."""
    encoder = OpusEncoder(rate, channels, "audio")
    packets: list[bytes] = []
    try:
        for index in range(frames):
            samples = [
                int(12000 * math.sin(2 * math.pi * 440 * (index * frame_samples + i) / rate))
                for i in range(frame_samples)
            ]
            interleaved = [value for value in samples for _ in range(channels)]
            pcm = struct.pack(f"<{len(interleaved)}h", *interleaved)
            packets.append(encoder.encode(pcm, frame_samples))
    finally:
        encoder.close()
    return packets


def test_decoder_rejects_nonsense_packets_without_raising() -> None:
    decoder = OpusDecoder(sample_rate=16000, channels=1)
    try:
        # An empty packet is refused outright; a truncated one must either
        # decode or be dropped, but must never raise.
        assert decoder.decode(b"") == b""
        assert decoder.last_error
    finally:
        decoder.close()


def test_recorder_writes_a_playable_wav(tmp_path: Path) -> None:
    """End-to-end: Opus packets in, a valid 16 kHz mono WAV out."""
    packets = _encode_tone(frames=10)
    wav_path = tmp_path / "participant-test.wav"

    recorder = OpusParticipantRecorder(wav_path, sample_rate=16000, channels=1)
    try:
        for packet in packets:
            assert recorder.write_packet(packet)
    finally:
        recorder.close()

    assert recorder.dropped_packets == 0
    # 10 packets x 20 ms = 200 ms of audio.
    assert recorder.duration_seconds == pytest.approx(0.2, abs=0.01)
    assert recorder.has_audio

    with wave.open(str(wav_path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == recorder.decoded_samples
        assert handle.readframes(handle.getnframes())


def test_recorder_keeps_going_after_a_bad_packet(tmp_path: Path) -> None:
    """A single unusable packet must not abort a participant's recording."""
    packets = _encode_tone(frames=4)
    wav_path = tmp_path / "participant-test.wav"

    recorder = OpusParticipantRecorder(wav_path, sample_rate=16000, channels=1)
    try:
        assert recorder.write_packet(packets[0])
        assert not recorder.write_packet(b"")  # dropped, not fatal
        assert recorder.write_packet(packets[1])
    finally:
        recorder.close()

    assert recorder.dropped_packets == 1
    assert recorder.decoded_samples == 2 * 320  # two good 20 ms frames at 16 kHz


def test_recorder_reports_no_audio_when_it_received_none(tmp_path: Path) -> None:
    recorder = OpusParticipantRecorder(tmp_path / "empty.wav")
    recorder.close()
    assert not recorder.has_audio
    assert recorder.duration_seconds == 0.0

    with wave.open(str(tmp_path / "empty.wav"), "rb") as handle:
        assert handle.getnframes() == 0


def test_writing_after_close_is_an_error(tmp_path: Path) -> None:
    recorder = OpusParticipantRecorder(tmp_path / "closed.wav")
    recorder.close()
    with pytest.raises(OpusError):
        recorder.write_packet(b"\x00" * 8)


def test_close_is_idempotent(tmp_path: Path) -> None:
    recorder = OpusParticipantRecorder(tmp_path / "twice.wav")
    recorder.close()
    recorder.close()


def test_participant_id_with_an_embedded_null_is_sanitized() -> None:
    """A NUL in the middle of the field must not reach a filename.

    ``str.strip()`` does not remove NUL, so an id like ``ab\\x00cd`` would
    otherwise produce a path that ``open()`` rejects with "embedded null byte".
    """
    participant_id, payload = split_frame(b"ab\x00cd".ljust(16, b"\x00") + b"payload")
    assert "\x00" not in participant_id
    assert participant_id == "ab_cd"
    assert payload == b"payload"


def test_participant_id_keeps_a_normally_padded_identifier() -> None:
    """NUL padding is transport framing, not part of the id."""
    participant_id, _ = split_frame(b"alice".ljust(16, b"\x00") + b"payload")
    assert participant_id == "alice"


def test_stereo_stream_decodes_into_a_mono_recording(tmp_path: Path) -> None:
    """A stereo Opus stream must still decode through a mono decoder.

    Whisper wants mono, so the recorder is always created with one channel;
    this asserts that a sender producing stereo does not silently yield
    nothing.
    """
    packets = _encode_tone(frames=4, channels=2)

    wav_path = tmp_path / "stereo-source.wav"
    recorder = OpusParticipantRecorder(wav_path, sample_rate=16000, channels=1)
    try:
        for packet in packets:
            recorder.write_packet(packet)
    finally:
        recorder.close()

    assert recorder.has_audio, "a stereo stream produced no mono audio"
    assert recorder.decoded_samples == 4 * 320  # 4 x 20 ms at 16 kHz
    with wave.open(str(wav_path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getnframes() == 4 * 320


# --------------------------------------------------------------------------
# Module isolation
# --------------------------------------------------------------------------


def test_only_the_config_module_reads_files_or_the_environment() -> None:
    """The package's stated isolation rule, enforced rather than trusted.

    ``config`` is the single place that resolves configuration, so no other
    module may reach for the environment or parse a config file itself.
    """
    package = Path(__file__).resolve().parent.parent / "src" / "jitsi_audio_bridge"
    offenders: list[str] = []

    for module in sorted(package.glob("*.py")):
        if module.name == "config.py":
            continue
        source = module.read_text(encoding="utf-8")
        for needle in ("os.environ", "os.getenv", "configparser"):
            if needle in source:
                offenders.append(f"{module.name}: {needle}")

    assert not offenders, "configuration is leaking outside config.py: " + ", ".join(offenders)
