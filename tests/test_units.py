"""Unit tests for the parts of the bridge that need no Whisper, Ollama or JVB.

The Opus round-trip at the bottom exercises the real libopus binding by
encoding a tone and decoding it back through the recorder.
"""

from __future__ import annotations

import base64
import http.server
import json
import math
import os
import struct
import tempfile
import threading
import time
import urllib.parse
import wave
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jitsi_audio_bridge import config as config_module
from jitsi_audio_bridge import daemon as daemon_module
from jitsi_audio_bridge import s3_upload
from jitsi_audio_bridge.audio import (
    EXTRACTED_AUDIO_NAME,
    TIMELINE_FILENAME,
    OpusDecoder,
    OpusEncoder,
    OpusError,
    OpusParticipantRecorder,
    attribute_speaker,
    discover_audio,
    parse_metadata,
    participant_id_from_path,
    room_name_from_metadata,
    slice_wav,
)
from jitsi_audio_bridge.config import ConfigError, load_config
from jitsi_audio_bridge.daemon import (
    DEFAULT_SESSION_ID,
    StreamError,
    _free_wav_path,
    adopt_session_metadata,
    build_media_json_pong,
    describe_media_json_start,
    extract_media_json_media,
    extract_session_id,
    parse_media_json_event,
    render_transcript,
    sanitize_identifier,
    split_frame,
    write_metadata,
)
from jitsi_audio_bridge.mailer import _subject_for, _usable_recipients, safe_attachment_name
from jitsi_audio_bridge.timeline import (
    SessionTimeline,
    Turn,
    TurnTracker,
    format_offset,
    merge_turns,
)

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
# Media-json framing (stock Jitsi's JVB)
# --------------------------------------------------------------------------


def _media_event(tag: str = "alice-audio", payload: bytes = b"\x01\x02\x03") -> str:
    return json.dumps(
        {
            "event": "media",
            "sequenceNumber": "2",
            "media": {
                "tag": tag,
                "chunk": "42",
                "timestamp": "1234567",
                "payload": base64.b64encode(payload).decode("ascii"),
            },
        }
    )


@pytest.mark.parametrize("kind", ["info", "start", "media", "ping", "session-end", "sources"])
def test_parse_media_json_event_accepts_documented_events(kind: str) -> None:
    assert parse_media_json_event(json.dumps({"event": kind})) == {"event": kind}


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[1, 2, 3]",
        '"a string"',
        "42",
        "null",
        "{}",
        json.dumps({"meeting_url": "https://meet.example.com/Room"}),
    ],
)
def test_parse_media_json_event_passes_other_frames_through(raw: str) -> None:
    assert parse_media_json_event(raw) is None


def test_parse_media_json_event_claims_a_non_string_event_name() -> None:
    # ``event`` is the discriminator: a frame carrying it must never fall
    # through to the control-frame path and overwrite metadata.json.
    assert parse_media_json_event('{"event": 5}') == {"event": 5}


def test_extract_media_json_media_round_trips() -> None:
    participant_id, raw_tag, packet = extract_media_json_media(
        json.loads(_media_event(payload=b"\xff\xfe"))
    )
    assert participant_id == "alice-audio"
    assert raw_tag == "alice-audio"
    assert packet == b"\xff\xfe"


def test_extract_media_json_media_returns_the_raw_tag_beside_the_safe_one() -> None:
    participant_id, raw_tag, _ = extract_media_json_media(json.loads(_media_event(tag="../evil")))
    assert "/" not in participant_id
    assert raw_tag == "../evil"


@pytest.mark.parametrize(
    "media",
    [
        None,
        "not an object",
        {},
        {"tag": ""},
        {"tag": "   "},
        {"tag": 7},
        {"tag": "...", "payload": "AA=="},
        {"tag": "alice-audio"},
        {"tag": "alice-audio", "payload": 7},
        {"tag": "alice-audio", "payload": "!!!"},
        {"tag": "alice-audio", "payload": "abc"},
        {"tag": "alice-audio", "payload": ""},
    ],
)
def test_extract_media_json_media_rejects_unusable_events(media: object) -> None:
    with pytest.raises(StreamError):
        extract_media_json_media({"event": "media", "media": media})


def test_build_media_json_pong_echoes_the_id() -> None:
    assert json.loads(build_media_json_pong({"event": "ping", "id": 7})) == {
        "event": "pong",
        "id": 7,
    }


def test_build_media_json_pong_accepts_zero() -> None:
    assert json.loads(build_media_json_pong({"event": "ping", "id": 0}))["id"] == 0


def test_build_media_json_pong_refuses_a_missing_id() -> None:
    assert build_media_json_pong({"event": "ping"}) is None


@pytest.mark.parametrize("ping_id", [None, "1", 1.5, True, False])
def test_build_media_json_pong_refuses_an_unusable_id(ping_id: object) -> None:
    # bool is excluded deliberately: JSON true is an int in Python and would
    # otherwise be echoed as 1.
    assert build_media_json_pong({"event": "ping", "id": ping_id}) is None


def test_describe_media_json_start_summarises_a_full_event() -> None:
    description = describe_media_json_start(
        {
            "event": "start",
            "start": {
                "tag": "alice-audio",
                "mediaFormat": {"encoding": "opus", "sampleRate": 48000, "channels": 2},
                "customParameters": {"endpointId": "endpoint-alice"},
            },
        }
    )
    assert "alice-audio" in description
    assert "endpoint-alice" in description
    assert "opus" in description


@pytest.mark.parametrize(
    "event",
    [{"event": "start"}, {"event": "start", "start": 5}, {"event": "start", "start": {}}],
)
def test_describe_media_json_start_tolerates_a_sparse_event(event: dict[str, object]) -> None:
    assert "?" in describe_media_json_start(event)


# --------------------------------------------------------------------------
# Control frames
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# Session metadata dropped by a companion service
# --------------------------------------------------------------------------


def test_adopted_metadata_becomes_the_sessions_own(tmp_path: Path) -> None:
    drop = tmp_path / "drop"
    drop.mkdir()
    session = tmp_path / "abc-123"
    session.mkdir()
    (drop / "abc-123.json").write_text(json.dumps({
        "room_name": "Weekly-Planning",
        "participants": [{"id": "8aa1c4ba", "name": "Alice", "email": "a@example.com"}],
    }))

    assert adopt_session_metadata(session, drop) == session / "metadata.json"
    stored = json.loads((session / "metadata.json").read_text())
    assert stored["room_name"] == "Weekly-Planning"
    # Consumed, so it is not picked up again by a later reprocessing.
    assert not (drop / "abc-123.json").exists()

    info = parse_metadata(session)
    assert info["room_name"] == "Weekly-Planning"
    assert info["recipients"] == ["a@example.com"]
    recording = session / "participant-8aa1c4ba-a0.wav"
    assert attribute_speaker(recording, info["id_to_name"]) == "Alice"


def test_adoption_leaves_the_sessions_own_metadata_alone(tmp_path: Path) -> None:
    drop = tmp_path / "drop"
    drop.mkdir()
    session = tmp_path / "abc-123"
    session.mkdir()
    (session / "metadata.json").write_text(json.dumps({"room_name": "From the sender"}))
    (drop / "abc-123.json").write_text(json.dumps({"room_name": "From Prosody"}))

    assert adopt_session_metadata(session, drop) is None
    assert json.loads((session / "metadata.json").read_text())["room_name"] == "From the sender"
    assert (drop / "abc-123.json").exists()          # still there if wanted later


def test_adoption_is_off_and_tolerant(tmp_path: Path) -> None:
    session = tmp_path / "abc-123"
    session.mkdir()
    drop = tmp_path / "drop"
    drop.mkdir()

    assert adopt_session_metadata(session, None) is None       # feature off
    assert adopt_session_metadata(session, drop) is None       # nothing dropped

    for bad in ("not json", "[1, 2]", '"text"'):
        (drop / "abc-123.json").write_text(bad)
        assert adopt_session_metadata(session, drop) is None
        assert not (session / "metadata.json").exists()

    # Too large to be a description of a meeting.
    (drop / "abc-123.json").write_text(json.dumps({"x": "y" * (1 << 21)}))
    assert adopt_session_metadata(session, drop) is None
    assert not (session / "metadata.json").exists()

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


def test_write_metadata_reports_a_missing_directory(tmp_path: Path) -> None:
    # Must not raise: an unwritable directory is a failed write, not a reason
    # to take the connection down.
    assert not write_metadata(tmp_path / "missing", json.dumps({"room_name": "Standup"}))


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
    # Mapped by both id and address: a recording may be named after either.
    assert meta["id_to_name"] == {
        "p1": "Ada",
        "ada@example.com": "Ada",
        "p2": "Grace",
        "grace@example.com": "Grace",
    }
    assert meta["recipients"] == ["ada@example.com", "grace@example.com"]
    assert meta["participants"] == ["Ada (ada@example.com)", "Grace (grace@example.com)"]



def test_named_participants_without_addresses_reach_the_summary_list(
    tmp_path: Path,
) -> None:
    """A deployment without tokens has names and no addresses; the summary
    prompt is told who was there from the names alone."""
    (tmp_path / "metadata.json").write_text(json.dumps({
        "room_name": "Standup",
        "participants": [
            {"id": "8aa1c4ba", "name": "Michael"},
            {"id": "bb7b6e09", "name": "Anna", "email": "anna@example.com"},
            {"id": "cc1d2e3f", "name": "Michael"},
        ],
    }))
    info = parse_metadata(tmp_path)
    assert info["participants"] == ["Michael", "Anna (anna@example.com)"]
    assert info["recipients"] == ["anna@example.com"]
    assert info["id_to_name"]["8aa1c4ba"] == "Michael"
    assert info["id_to_name"]["bb7b6e09"] == "Anna"
    assert info["room_name"] == "Standup"



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
    assert meta["id_to_name"] == {"u1": "Alan", "alan@example.com": "Alan"}
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


# --- the shapes the real sender actually produces -------------------------


def test_room_name_comes_from_the_meeting_url(tmp_path: Path) -> None:
    """Jitsi's metadata has no room_name; the room is the URL's last segment."""
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "meeting_url": "https://meet.example.com/Weekly-Planning",
                "participants": [{"user": {"id": "u1", "name": "Ada"}}],
            }
        )
    )
    assert parse_metadata(tmp_path)["room_name"] == "Weekly-Planning"


def test_room_name_from_a_url_with_a_trailing_slash() -> None:
    assert room_name_from_metadata({"meeting_url": "https://m.example.com/Room/"}) == "Room"


def test_an_explicit_room_name_beats_the_url() -> None:
    meta = {"room_name": "Explicit", "meeting_url": "https://m.example.com/FromUrl"}
    assert room_name_from_metadata(meta) == "Explicit"


def test_room_name_is_absent_when_nothing_says_otherwise() -> None:
    assert room_name_from_metadata({}) is None
    assert room_name_from_metadata({"meeting_url": ""}) is None
    assert room_name_from_metadata({"meeting_url": "https://m.example.com"}) is None


def test_a_url_of_only_separators_yields_nothing() -> None:
    assert room_name_from_metadata({"meeting_url": "https://m.example.com///"}) is None


def test_parse_metadata_accepts_the_alternate_field_spellings(tmp_path: Path) -> None:
    """``mail`` and ``display_name`` appear in the wild alongside email/name."""
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "meeting_url": "https://meet.example.com/Standup",
                "participants": [{"id": "p1", "display_name": "Ada", "mail": "ada@example.com"}],
            }
        )
    )
    meta = parse_metadata(tmp_path)
    assert meta["recipients"] == ["ada@example.com"]
    assert meta["id_to_name"]["ada@example.com"] == "Ada"


def test_parse_metadata_recovers_addresses_from_unstructured_metadata(tmp_path: Path) -> None:
    """When no structured recipient exists, an address buried in the document is
    still better than silently falling back to the admin."""
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "meeting_url": "https://meet.example.com/Standup",
                "note": "contact grace@example.com for the minutes",
            }
        )
    )
    meta = parse_metadata(tmp_path)
    assert meta["recipients"] == ["grace@example.com"]


def test_parse_metadata_ignores_things_that_are_not_addresses(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps({"participants": [{"id": "p1", "name": "Ada", "email": "not-an-address"}]})
    )
    assert parse_metadata(tmp_path)["recipients"] == []


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
# Speaker attribution and audio discovery
# --------------------------------------------------------------------------


def test_attribute_speaker_matches_a_participant_id(tmp_path: Path) -> None:
    path = tmp_path / "participant-abc123.wav"
    assert attribute_speaker(path, {"abc123": "Ada"}) == "Ada"


def test_attribute_speaker_matches_an_email_address(tmp_path: Path) -> None:
    path = tmp_path / "ada@example.com_audio.wav"
    assert attribute_speaker(path, {"ada@example.com": "Ada"}) == "Ada"


def test_attribute_speaker_falls_back_to_the_filename(tmp_path: Path) -> None:
    path = tmp_path / "participant-unknown.wav"
    assert attribute_speaker(path, {"abc123": "Ada"}) == "participant-unknown"


def test_attribute_speaker_without_any_mapping(tmp_path: Path) -> None:
    path = tmp_path / "participant-abc.wav"
    assert attribute_speaker(path, {}) == "participant-abc"


def _touch(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def test_discover_audio_finds_live_capture_recordings(tmp_path: Path) -> None:
    _touch(tmp_path / "participant-alice.wav")
    _touch(tmp_path / "participant-bob.wav")
    participants, master = discover_audio(tmp_path)
    assert [p.name for p in participants] == ["participant-alice.wav", "participant-bob.wav"]
    assert master is None


def test_discover_audio_finds_per_speaker_recordings(tmp_path: Path) -> None:
    _touch(tmp_path / "alice@example.com_audio.wav")
    participants, master = discover_audio(tmp_path)
    assert [p.name for p in participants] == ["alice@example.com_audio.wav"]
    assert master is None


def test_discover_audio_prefers_participant_files_over_a_master(tmp_path: Path) -> None:
    _touch(tmp_path / "participant-alice.wav")
    _touch(tmp_path / "room.mp4")
    participants, master = discover_audio(tmp_path)
    assert len(participants) == 1
    assert master is None


def test_discover_audio_falls_back_to_a_master_recording(tmp_path: Path) -> None:
    _touch(tmp_path / "room-recording.mp4")
    participants, master = discover_audio(tmp_path)
    assert participants == []
    assert master is not None and master.name == "room-recording.mp4"


def test_discover_audio_never_returns_its_own_extraction(tmp_path: Path) -> None:
    """extracted_audio.wav is derived, so it must not be picked up as a source."""
    _touch(tmp_path / EXTRACTED_AUDIO_NAME)
    participants, master = discover_audio(tmp_path)
    assert participants == []
    assert master is None


def test_discover_audio_on_an_empty_directory(tmp_path: Path) -> None:
    assert discover_audio(tmp_path) == ([], None)


def test_master_media_is_chosen_by_container_preference(tmp_path: Path) -> None:
    _touch(tmp_path / "recording.mkv")
    _touch(tmp_path / "recording.wav")
    _, master = discover_audio(tmp_path)
    assert master is not None and master.suffix == ".wav"


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






def test_the_language_rule_follows_the_transcript() -> None:
    """A software meeting is full of English words whatever language it is in."""
    from jitsi_audio_bridge.ai_client import build_summary_prompt

    prompt = build_summary_prompt(
        "[michael]: il fix e' merged, guarda il file name dello widget",
        "panicking", ["michael"], "Italian",
    )
    after_transcript = prompt.split("Meeting Transcript:", 1)[1]
    assert "answer in ITALIAN" in after_transcript
    assert "Do not reply in English" in after_transcript
    # And the instruction is not only at the top, where the transcript buries it.
    assert prompt.index("answer in ITALIAN") > prompt.index("Meeting Transcript:")


def test_the_sign_off_follows_the_language_too() -> None:
    from jitsi_audio_bridge.mailer import mail_strings

    assert mail_strings("Italian")[3].startswith("Cordiali saluti")
    assert mail_strings("english")[3].startswith("Best regards")
    assert mail_strings("Klingon")[3] == mail_strings("English")[3]


def test_correction_is_off_unless_it_is_asked_for(clean_env: None) -> None:
    config = load_config()
    assert config.ollama.correct_transcript is False



def test_the_summary_reads_the_corrected_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """Repaired words go to the summary and the mail; the raw text stays."""
    monkeypatch.setattr(config_module, "SEARCH_PATHS", (tmp_path / "nothing.ini",))
    config = load_config()
    config = replace(
        config, ollama=replace(config.ollama, correct_transcript=True)
    )

    session = tmp_path / "session"
    session.mkdir()
    recording = session / "participant-michael-a0.wav"
    recorder = OpusParticipantRecorder(recording)
    for packet in _encode_tone(frames=25):
        recorder.write_packet(packet)
    recorder.close()
    SessionTimeline(
        started_at="2026-10-06T18:50:00+00:00",
        duration=0.5,
        recorded={"michael-a0": 0.5},
        turns=[Turn("michael-a0", 0.0, 0.5, 0.0, 8000)],
    ).write(session / "timeline.json")

    seen: dict[str, object] = {}
    monkeypatch.setattr(daemon_module, "detect_language", lambda text, ep: "Italian")
    monkeypatch.setattr(
        daemon_module,
        "transcribe_audio",
        lambda path, ep: "[Michael]: we discussed teh deployment",
    )

    def fake_correct(text: str, endpoint: object, language: str | None = None) -> str:
        seen["corrected_from"] = text
        return "[Michael]: we discussed the deployment"

    def fake_summary(text: str, room: str, participants: list, endpoint: object,
                     language: str | None = None) -> str:
        seen["summarised"] = text
        return "riassunto"

    def fake_mail(recipients: object, room: str, summary: str, transcript: Path,
                  summary_path: Path, smtp: object, **kwargs: object) -> bool:
        seen["attached"] = transcript
        return True

    monkeypatch.setattr(daemon_module, "correct_transcript", fake_correct)
    monkeypatch.setattr(daemon_module, "generate_summary", fake_summary)
    monkeypatch.setattr(daemon_module, "send_meeting_email", fake_mail)

    assert daemon_module.process_completed_session(session, config) is True

    # The line arrives rendered, timestamp and speaker already in place.
    assert "teh deployment" in seen["corrected_from"]
    assert seen["corrected_from"].startswith("[00:00:00] participant-michael-a0: ")
    assert seen["summarised"] == "[Michael]: we discussed the deployment"
    assert seen["attached"] == session / "transcript.corrected.txt"
    # What was actually said is still on disk, untouched.
    assert "teh deployment" in (session / "transcript.txt").read_text()
    assert (session / "transcript.corrected.txt").read_text() == (
        "[Michael]: we discussed the deployment"
    )


def test_a_failed_correction_falls_back_to_the_raw_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    monkeypatch.setattr(config_module, "SEARCH_PATHS", (tmp_path / "nothing.ini",))
    config = replace(
        load_config(), ollama=replace(load_config().ollama, correct_transcript=True)
    )

    session = tmp_path / "session"
    session.mkdir()
    recorder = OpusParticipantRecorder(session / "participant-x-a0.wav")
    for packet in _encode_tone(frames=25):
        recorder.write_packet(packet)
    recorder.close()

    monkeypatch.setattr(daemon_module, "detect_language", lambda text, ep: "Italian")
    monkeypatch.setattr(daemon_module, "transcribe_audio", lambda path, ep: "raw text")
    monkeypatch.setattr(daemon_module, "correct_transcript", lambda *a, **k: "")
    seen: dict[str, object] = {}

    def fake_summary(text: str, room: str, participants: list, endpoint: object,
                     language: str | None = None) -> str:
        seen["summarised"] = text
        return ""

    monkeypatch.setattr(daemon_module, "generate_summary", fake_summary)
    assert daemon_module.process_completed_session(session, config) is False
    assert "raw text" in seen["summarised"]
    assert not (session / "transcript.corrected.txt").exists()



def test_correction_chunks_keep_speaker_lines_whole() -> None:
    from jitsi_audio_bridge.ai_client import split_for_correction

    transcript = "\n".join(f"[Alice]: sentence number {i} " + "word " * 20 for i in range(60))
    chunks = split_for_correction(transcript, 700)
    assert len(chunks) > 1
    assert all(len(chunk) <= 700 for chunk in chunks)
    # No line is cut in half, and nothing is dropped.
    assert all(line.startswith("[Alice]: ") for chunk in chunks for line in chunk.splitlines())
    assert sum(len(chunk.splitlines()) for chunk in chunks) == 60


def test_a_line_too_long_to_fit_is_cut_anyway() -> None:
    from jitsi_audio_bridge.ai_client import split_for_correction

    chunks = split_for_correction("x" * 25, 10)
    assert "".join(chunks) == "x" * 25



def test_the_mail_is_written_in_the_meetings_language() -> None:
    """An Italian meeting should not be headed in English."""
    from jitsi_audio_bridge.mailer import _subject_for, mail_strings

    subject, heading, introduction, sign_off = mail_strings("Italian")
    assert heading == "RIEPILOGO DELLA RIUNIONE"
    assert "riunione" in introduction.format(room="Weekly")
    assert sign_off.startswith("Cordiali saluti")
    assert _subject_for("Weekly", "", None, "Italian") == (
        "Riepilogo e trascrizione della riunione: Weekly"
    )
    # However the model spelled it, and case-insensitively.
    assert mail_strings("italian") == mail_strings("Italian")
    assert mail_strings(" French ")[1] == "RÉSUMÉ DE LA RÉUNION"

    # A language we have no words for is mailed in English, not in nothing.
    assert mail_strings("Klingon") == mail_strings("English")
    assert mail_strings(None) == mail_strings("English")


def test_the_subject_says_when_the_meeting_was() -> None:
    """One room, many meetings: the name alone cannot tell them apart."""
    from jitsi_audio_bridge.mailer import _format_when, _subject_for

    assert _subject_for("Weekly", "") == "Meeting Summary & Transcript: Weekly"
    assert _subject_for("Weekly", "", "2026-10-06T18:50:00") == (
        "Meeting Summary & Transcript: Weekly (2026-10-06 18:50)"
    )
    assert _subject_for("Weekly", "Amarula", "2026-10-06T18:50:00") == (
        "Meeting Summary & Transcript: Weekly (2026-10-06 18:50) - Amarula"
    )
    # Anything unparseable is left out; the mail still goes.
    assert _format_when("not a time") == ""
    assert _format_when(None) == ""
    assert _subject_for("Weekly", "", "not a time") == "Meeting Summary & Transcript: Weekly"


def test_subject_flattens_embedded_newlines() -> None:
    """A newline in the room name would otherwise raise inside EmailMessage."""
    subject = _subject_for("Standup\r\nBcc: evil@example.com")
    assert "\r" not in subject
    assert "\n" not in subject
    assert "Bcc:" in subject  # neutralised into the subject, not injected


def test_subject_falls_back_for_an_empty_room_name() -> None:
    assert _subject_for("") == "Meeting Summary & Transcript: Meeting"


def test_subject_appends_a_configured_suffix() -> None:
    subject = _subject_for("Standup", "- Amarula Solutions")
    assert subject == "Meeting Summary & Transcript: Standup - Amarula Solutions"


def test_subject_suffix_is_flattened_and_optional() -> None:
    assert _subject_for("Standup", "") == "Meeting Summary & Transcript: Standup"
    assert "\n" not in _subject_for("Standup", "x\r\ny")


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
# Speaking turns, the timeline, and the interleaved transcript
# --------------------------------------------------------------------------

#: One Opus frame as the JVB sends it: 20 ms.
FRAME = 0.02


def _speak(
    tracker: TurnTracker,
    participant: str,
    start: float,
    seconds: float,
    *,
    level: float | None = 0.2,
    vad: bool | None = None,
) -> float:
    """Feed *seconds* of 20 ms packets; returns where they ended."""
    cursor = start
    for _ in range(int(seconds / FRAME)):
        tracker.add(
            participant,
            session_offset=cursor,
            file_offset=cursor,
            duration=FRAME,
            level=level,
            vad=vad,
        )
        cursor += FRAME
    return cursor


def test_tracker_turns_continuous_speech_into_one_turn() -> None:
    tracker = TurnTracker()
    _speak(tracker, "alice", 0.0, 1.0)
    _speak(tracker, "alice", 1.0, 0.5, level=0.0)          # comfort noise, not speech
    (turn,) = tracker.finish()
    assert turn.participant == "alice"
    assert (turn.start, turn.end) == (0.0, pytest.approx(1.0, abs=FRAME))
    assert turn.offset == 0.0


def test_tracker_merges_a_pause_and_splits_a_longer_one() -> None:
    tracker = TurnTracker(merge_gap=0.5)
    _speak(tracker, "alice", 0.0, 1.0)
    resumed = _speak(tracker, "alice", 1.0, 0.3, level=0.0)
    _speak(tracker, "alice", resumed, 1.0)
    (turn,) = tracker.finish()
    assert turn.end == pytest.approx(2.3, abs=FRAME)       # one turn across the pause

    tracker = TurnTracker(merge_gap=0.5)
    _speak(tracker, "alice", 0.0, 1.0)
    resumed = _speak(tracker, "alice", 1.0, 1.5, level=0.0)
    _speak(tracker, "alice", resumed, 1.0)
    assert [round(turn.start, 2) for turn in tracker.finish()] == [0.0, 2.5]


def test_tracker_trusts_the_vad_flag_over_the_level() -> None:
    tracker = TurnTracker()
    _speak(tracker, "alice", 0.0, 1.0, level=0.0, vad=True)
    (turn,) = tracker.finish()
    assert turn.duration == pytest.approx(1.0, abs=FRAME)

    tracker = TurnTracker()
    _speak(tracker, "alice", 0.0, 1.0, level=0.5, vad=False)
    assert tracker.finish() == []


def test_tracker_treats_a_packet_with_no_hints_as_speech() -> None:
    """An older sender gives neither a level nor a flag; assume someone talks."""
    tracker = TurnTracker()
    _speak(tracker, "alice", 0.0, 0.5, level=None)
    (turn,) = tracker.finish()
    assert turn.duration == pytest.approx(0.5, abs=FRAME)

    # But a level that is there and low still means silence.
    tracker = TurnTracker()
    _speak(tracker, "alice", 0.0, 0.5, level=0.001)
    assert tracker.finish() == []


def test_tracker_drops_blips_and_splits_monologues() -> None:
    tracker = TurnTracker(min_turn=0.3)
    tracker.add("alice", session_offset=0.0, file_offset=0.0, duration=FRAME, level=0.5)
    assert tracker.finish() == []                          # 20 ms is a click, not speech

    tracker = TurnTracker(max_turn=1.0)
    _speak(tracker, "alice", 0.0, 3.0)
    turns = tracker.finish()
    assert [round(turn.duration, 2) for turn in turns] == [1.0, 1.0, 1.0]
    assert [round(turn.start, 2) for turn in turns] == [0.0, 1.0, 2.0]


def test_tracker_keeps_participants_apart() -> None:
    tracker = TurnTracker()
    _speak(tracker, "alice", 0.0, 0.5)
    _speak(tracker, "bob", 0.0, 0.5)
    turns = tracker.finish()
    assert [turn.participant for turn in turns] == ["alice", "bob"]
    assert all(turn.duration >= 0.3 for turn in turns)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00:00"), (61, "00:01:01"), (3725, "01:02:05"), (-4, "00:00:00")],
)
def test_format_offset(seconds: float, expected: str) -> None:
    assert format_offset(seconds) == expected


def test_merging_turns_keeps_the_order_and_loses_resolution() -> None:
    turns = [
        Turn("a", 0.0, 1.0, 0.0, 3),
        Turn("a", 1.2, 2.0, 1.0, 3),      # closest pair: 0.2 s apart
        Turn("a", 9.0, 10.0, 2.0, 3),
    ]
    merged = merge_turns(list(turns), 2)
    assert [(turn.start, turn.end, turn.offset) for turn in merged] == [
        (0.0, 2.0, 0.0),
        (9.0, 10.0, 2.0),
    ]
    assert merged[0].samples == 6

    assert merge_turns(list(turns), 5) == turns        # nothing to do
    assert len(merge_turns(list(turns), 1)) == 1


def test_timeline_round_trips_and_survives_junk() -> None:
    timeline = SessionTimeline(
        started_at="2026-10-06T18:00:00+00:00",
        duration=12.5,
        recorded={"alice": 4.0},
        turns=[Turn("alice", 0.5, 1.5, 0.0, 1000)],
    )
    restored = SessionTimeline.from_json(timeline.to_json())
    assert restored is not None
    assert restored.turns_for("alice") == [Turn("alice", 0.5, 1.5, 0.0, 1000)]
    assert restored.recorded == {"alice": 4.0}
    assert restored.started_at == timeline.started_at

    assert SessionTimeline.from_json("not json") is None
    assert SessionTimeline.from_json("[1, 2]") is None
    # A turn that is not a turn is dropped; the rest of the document survives.
    partial = SessionTimeline.from_json(
        '{"turns": [{"participant": "a"}, 5,'
        ' {"participant": "b", "start": 0, "end": 1, "offset": 0, "samples": 1}]}'
    )
    assert partial is not None
    assert [turn.participant for turn in partial.turns] == ["b"]


def test_timeline_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    timeline = SessionTimeline(started_at="now", duration=1.0, turns=[Turn("a", 0, 1, 0, 1)])
    path = tmp_path / "timeline.json"
    assert timeline.write(path)
    assert [item.name for item in tmp_path.iterdir()] == ["timeline.json"]
    assert SessionTimeline.load(path) == timeline
    assert SessionTimeline.load(tmp_path / "absent.json") is None




def test_ai_requests_are_held_to_the_configured_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """Two meetings overlapping must not have one starve the other's model."""
    from jitsi_audio_bridge import ai_client

    monkeypatch.setattr(config_module, "SEARCH_PATHS", (tmp_path / "nothing.ini",))
    config = load_config()
    audio = tmp_path / "participant-x.wav"
    recorder = OpusParticipantRecorder(audio)
    for packet in _encode_tone(frames=5):
        recorder.write_packet(packet)
    recorder.close()

    live = 0
    peak = 0
    lock = threading.Lock()

    class Response:
        status_code = 200
        def raise_for_status(self) -> None: ...
        def json(self) -> dict:
            return {"text": "hello"}

    def slow_post(*args: object, **kwargs: object) -> Response:
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lock:
            live -= 1
        return Response()

    monkeypatch.setattr(ai_client.requests, "post", slow_post)

    def run() -> None:
        ai_client.transcribe_audio(audio, config.whisper)

    def both() -> int:
        nonlocal peak
        peak = 0
        threads = [threading.Thread(target=run) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return peak

    # The default matches a single GPU, which queues one at a time.
    ai_client.set_ai_limits(1, 3)
    assert both() == 1

    # Split across machines, or a device that takes more, they may run at once.
    ai_client.set_ai_limits(3, 3)
    assert both() == 3
    ai_client.set_ai_limits(2, 3)
    assert both() == 2
    ai_client.set_ai_limits(1, ai_client.DEFAULT_MAX_ATTEMPTS)




def test_retries_double_their_wait_and_stop_at_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """A model being evicted takes seconds, then takes them all at once."""
    from jitsi_audio_bridge import ai_client

    monkeypatch.setattr(config_module, "SEARCH_PATHS", (tmp_path / "nothing.ini",))
    config = load_config()
    assert config.ai.max_attempts == 5
    ai_client.set_ai_limits(config.ai.max_concurrent_requests, config.ai.max_attempts)
    audio = tmp_path / "participant-x.wav"
    audio.write_bytes(b"not really audio, but never sent")

    waits: list[float] = []
    attempts = 0

    class Response:
        status_code = 503
        def raise_for_status(self) -> None: ...
        def json(self) -> dict:
            return {}

    def always_503(*args: object, **kwargs: object) -> Response:
        nonlocal attempts
        attempts += 1
        return Response()

    monkeypatch.setattr(ai_client.requests, "post", always_503)
    monkeypatch.setattr(ai_client.time, "sleep", waits.append)

    assert ai_client.transcribe_audio(audio, config.whisper) == ""
    assert attempts == 5
    assert waits == [1.0, 2.0, 4.0, 8.0]

    # The schedule is capped, so a large limit cannot sleep for hours.
    assert ai_client.backoff_seconds(12) == 30.0
    assert ai_client.backoff_seconds(1) == 0.0



def test_a_failure_body_is_quoted_in_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None, caplog
) -> None:
    """The service's reason is the only place it appears; keep it in ours."""
    from jitsi_audio_bridge import ai_client

    monkeypatch.setattr(config_module, "SEARCH_PATHS", (tmp_path / "nothing.ini",))
    config = load_config()
    ai_client.set_ai_limits(1, 2)
    audio = tmp_path / "participant-x.wav"
    audio.write_bytes(b"payload")

    class Response:
        status_code = 503
        text = '{\n  "error": "all slots are busy",\n  "cuda": "out of memory"\n}'
        def json(self) -> dict:
            return {}

    monkeypatch.setattr(ai_client.requests, "post", lambda *a, **k: Response())
    monkeypatch.setattr(ai_client.time, "sleep", lambda _: None)

    with caplog.at_level("WARNING"):
        assert ai_client.transcribe_audio(audio, config.whisper) == ""
    assert "all slots are busy" in caplog.text
    # One line, and bounded: a service that answers with a novel must not
    # flood the journal.
    line = next(record.getMessage() for record in caplog.records if "503" in record.getMessage())
    assert "\n" not in line
    assert len(ai_client.describe_failure(Response())) < 240

    ai_client.set_ai_limits(1, ai_client.DEFAULT_MAX_ATTEMPTS)


def test_a_client_error_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """A 4xx says the request is wrong; asking again changes nothing."""
    from jitsi_audio_bridge import ai_client

    monkeypatch.setattr(config_module, "SEARCH_PATHS", (tmp_path / "nothing.ini",))
    config = load_config()
    ai_client.set_ai_limits(config.ai.max_concurrent_requests, config.ai.max_attempts)
    audio = tmp_path / "participant-x.wav"
    audio.write_bytes(b"payload")

    calls = 0

    class Response:
        status_code = 400
        def raise_for_status(self) -> None:
            raise ai_client.requests.HTTPError("400 Client Error", response=self)
        def json(self) -> dict:
            return {}

    def bad_request(*args: object, **kwargs: object) -> Response:
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr(ai_client.requests, "post", bad_request)
    monkeypatch.setattr(ai_client.time, "sleep", lambda _: None)

    assert ai_client.transcribe_audio(audio, config.whisper) == ""
    assert calls == 1



def test_a_participant_whose_turns_all_failed_is_transcribed_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """Every turn failing is a service problem; one request may still work."""
    monkeypatch.setattr(config_module, "SEARCH_PATHS", (tmp_path / "nothing.ini",))
    config = load_config()

    recording = tmp_path / "participant-alice.wav"
    recorder = OpusParticipantRecorder(recording)
    for packet in _encode_tone(frames=50):
        recorder.write_packet(packet)
    recorder.close()

    timeline = SessionTimeline(
        started_at="now",
        duration=1.0,
        recorded={"alice": 1.0},
        turns=[Turn("alice", 0.0, 0.4, 0.0, 10), Turn("alice", 0.5, 1.0, 0.4, 10)],
    )

    def refuses_turns(path: str | Path, endpoint: object) -> str:
        return "" if "-00" in Path(path).name else "said something at length"

    monkeypatch.setattr(daemon_module, "transcribe_audio", refuses_turns)
    lines = daemon_module.transcribe_recordings(
        [recording], {"id_to_name": {}}, timeline, config
    )
    # One line, unattributed in time because it is the whole recording, and
    # still attributed to the right speaker.
    assert lines == [(None, "participant-alice", "said something at length")]


def test_lost_turns_are_given_a_second_chance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """An intermittent service is usually back by the end of the session."""
    monkeypatch.setattr(config_module, "SEARCH_PATHS", (tmp_path / "nothing.ini",))
    monkeypatch.setattr(daemon_module, "RETRY_PASS_PAUSE_SECONDS", 0.0)
    config = load_config()

    recording = tmp_path / "participant-alice.wav"
    recorder = OpusParticipantRecorder(recording)
    for packet in _encode_tone(frames=50):
        recorder.write_packet(packet)
    recorder.close()

    timeline = SessionTimeline(
        started_at="now",
        duration=1.0,
        recorded={"alice": 1.0},
        turns=[Turn("alice", 0.0, 0.4, 0.0, 10), Turn("alice", 0.5, 1.0, 0.4, 10)],
    )

    attempts: dict[str, int] = {}

    def flaky(path: str | Path, endpoint: object) -> str:
        name = Path(path).name
        attempts[name] = attempts.get(name, 0) + 1
        # The first request for each turn is refused; the retry pass gets them.
        return "" if attempts[name] == 1 else f"text for {name}"

    monkeypatch.setattr(daemon_module, "transcribe_audio", flaky)
    lines = daemon_module.transcribe_recordings(
        [recording], {"id_to_name": {}}, timeline, config
    )
    assert [text for _, _, text in lines] == [
        "text for participant-alice-0000.wav",
        "text for participant-alice-0001.wav",
    ]
    assert [start for start, _, _ in lines] == [0.0, 0.5]



def test_slice_wav_cuts_the_named_window(tmp_path: Path) -> None:
    source = tmp_path / "participant-alice.wav"
    recorder = OpusParticipantRecorder(source)
    for packet in _encode_tone(frames=50):                 # one second of audio
        recorder.write_packet(packet)
    recorder.close()

    half = slice_wav(source, 0.5, 0.2, tmp_path / "turns" / "alice-0001.wav")
    with wave.open(str(half), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getnframes() == pytest.approx(0.2 * 16000, abs=2)
    with wave.open(str(source), "rb") as handle:
        assert handle.getnframes() == pytest.approx(16000, abs=2)

    with pytest.raises(OpusError):
        slice_wav(source, 5.0, 0.2, tmp_path / "turns" / "past-the-end.wav")


def test_participant_id_comes_from_the_file_name() -> None:
    assert participant_id_from_path(Path("/x/participant-8aa1c4ba-a0.wav")) == "8aa1c4ba-a0"
    assert participant_id_from_path(Path("/x/alice_audio.wav")) is None


def test_render_transcript_orders_by_time_and_leaves_untimed_last() -> None:
    rendered = render_transcript(
        [
            (12.0, "Bob", "second"),
            (None, "Zoe", "no timeline for this one"),
            (3.0, "Alice", "first"),
        ]
    )
    assert rendered == (
        "[00:00:03] Alice: first"
        "\n\n[00:00:12] Bob: second"
        "\n\n[Zoe]: no timeline for this one"
    )


def test_new_settings_default_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 clean_env: None) -> None:
    monkeypatch.setattr(config_module, "SEARCH_PATHS", (tmp_path / "nothing.ini",))
    config = load_config()
    assert config.storage.capture_timeline is True
    assert config.transcript.interleave is True
    assert config.transcript.merge_gap_seconds == 1.0
    # One at a time: what a single GPU queues.
    assert config.ai.max_concurrent_requests == 1


def test_a_concurrency_below_one_is_rejected(tmp_path: Path, clean_env: None) -> None:
    path = tmp_path / "config.ini"
    path.write_text("[ai]\nmax_concurrent_requests = 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_concurrent_requests"):
        load_config(path)


def test_a_malformed_merge_gap_is_rejected(tmp_path: Path, clean_env: None) -> None:
    path = tmp_path / "config.ini"
    path.write_text("[transcript]\nmerge_gap_seconds = soon\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="merge_gap_seconds"):
        load_config(path)



def test_each_connection_records_into_its_own_part(tmp_path: Path) -> None:
    """wave.open truncates, so a reconnect must not reuse the first file."""
    first = _free_wav_path(tmp_path, "d6ae7ffe-a0")
    assert first.name == "participant-d6ae7ffe-a0.wav"
    first.write_bytes(b"audio")

    second = _free_wav_path(tmp_path, "d6ae7ffe-a0")
    assert second.name == "participant-d6ae7ffe-a0-2.wav"
    second.write_bytes(b"more audio")

    third = _free_wav_path(tmp_path, "d6ae7ffe-a0")
    assert third.name == "participant-d6ae7ffe-a0-3.wav"
    # Another participant is unaffected by any of it.
    assert _free_wav_path(tmp_path, "bb7b6e09-a0").name == "participant-bb7b6e09-a0.wav"


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
# Uploading the meeting's video to an S3-compatible endpoint
# --------------------------------------------------------------------------

#: A fixed past moment to hang recording timestamps on.
_MEETING_EPOCH = 1_760_000_000.0


def _s3_config(tmp_path: Path, **overrides: object):
    """The default configuration with an [s3] section pointed at a test tree.

    Only the endpoint and the bucket turn the feature on, so a test that wants
    it off simply does not pass them.
    """
    base = load_config()
    settings: dict[str, object] = {
        "endpoint": "http://127.0.0.1:1",
        "bucket": "meetings-recordings",
        "prefix": "",
        "access_key": "test-access-key",
        "secret_key": "test-secret-key",
        "jibri_dir": tmp_path / "jibri",
        "wait_seconds": 0.0,
        "settle_seconds": 0.0,
    }
    settings.update(overrides)
    return replace(
        base,
        storage=replace(base.storage, recordings_dir=tmp_path / "recordings"),
        s3=replace(base.s3, **settings),  # type: ignore[arg-type]
    )


#: The moment the meetings in these tests started, on both clocks.
_MEETING_STARTED_AT = datetime.fromtimestamp(_MEETING_EPOCH, UTC).isoformat(timespec="seconds")


def _meeting_dir(
    tmp_path: Path,
    room: str = "Standup",
    session: str = "abcd",
    *,
    started_at: str | None = _MEETING_STARTED_AT,
) -> Path:
    """A meeting directory as post-processing finds it.

    With a timeline by default: it is what tells the video search when the
    meeting began, and a real one has it — the tests that deliberately do
    without pass ``started_at=None``.
    """
    meeting = tmp_path / "recordings" / session
    meeting.mkdir(parents=True, exist_ok=True)
    (meeting / "metadata.json").write_text(
        json.dumps({"room_name": room, "participants": []}), encoding="utf-8"
    )
    if started_at is not None:
        SessionTimeline(started_at=started_at, duration=600.0).write(
            meeting / TIMELINE_FILENAME
        )
    return meeting


def _jibri_recording(
    root: Path,
    room: str,
    when: float = _MEETING_EPOCH,
    *,
    session: str = "jibri-session",
    filename: str | None = None,
    meeting_url: str | None = None,
) -> Path:
    """One directory of Jibri's recordings tree, as Jibri leaves it."""
    directory = root / session
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime(when))
    path = directory / (filename or f"{room}_{stamp}.mp4")
    path.write_bytes(b"\x00" * 32)
    os.utime(path, (when, when))
    if meeting_url is not False:
        (directory / "metadata.json").write_text(
            json.dumps({"meeting_url": meeting_url or f"https://jitsi.example.com/{room}"}),
            encoding="utf-8",
        )
    return path


class _StubS3:
    """Enough of an S3 endpoint to prove the client is wired up correctly.

    A mocked client would hide the things that actually break: the address a
    request goes to, the shape of the path, and whether it arrived signed at
    all.  A real boto3 client talking to this is the smallest test that
    catches those.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.requests: list[tuple[str, str, str]] = []
        self._parts: dict[str, dict[int, bytes]] = {}
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:  # noqa: ARG002 - quiet
                pass

            def _query(self) -> dict[str, list[str]]:
                return urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query, keep_blank_values=True
                )

            def _key(self) -> str:
                path = urllib.parse.urlparse(self.path).path.lstrip("/")
                return path.split("/", 1)[1] if "/" in path else path

            def _note(self, method: str) -> None:
                stub.requests.append(
                    (method, self.path, self.headers.get("Authorization", ""))
                )

            def _body(self) -> bytes:
                return self.rfile.read(int(self.headers.get("Content-Length") or 0))

            def _reply(
                self,
                status: int,
                body: bytes = b"",
                length: int | None = None,
                etag: str | None = None,
            ) -> None:
                self.send_response(status)
                if etag is not None:
                    self.send_header("ETag", etag)
                self.send_header("Content-Length", str(len(body) if length is None else length))
                self.end_headers()
                if body and self.command != "HEAD":
                    self.wfile.write(body)

            def do_PUT(self) -> None:
                self._note("PUT")
                query = self._query()
                body = self._body()
                if "partNumber" in query:
                    parts = stub._parts.setdefault(query.get("uploadId", [""])[0], {})
                    parts[int(query["partNumber"][0])] = body
                    # A real server returns each part's ETag in a header, and
                    # the client needs them back to complete the upload.
                    self._reply(200, etag=f'"part-{query["partNumber"][0]}"')
                    return
                stub.objects[self._key()] = body
                self._reply(200, etag='"whole"')

            def do_POST(self) -> None:
                self._note("POST")
                query = self._query()
                self._body()
                if "uploads" in query:
                    upload_id = f"upload-{len(stub._parts)}"
                    stub._parts[upload_id] = {}
                    body = (
                        "<InitiateMultipartUploadResult>"
                        f"<UploadId>{upload_id}</UploadId>"
                        "</InitiateMultipartUploadResult>"
                    ).encode()
                else:
                    parts = stub._parts.pop(query.get("uploadId", [""])[0], {})
                    stub.objects[self._key()] = b"".join(
                        parts[number] for number in sorted(parts)
                    )
                    body = (
                        b"<CompleteMultipartUploadResult>"
                        b"<ETag>\"stub\"</ETag></CompleteMultipartUploadResult>"
                    )
                self._reply(200, body)

            def do_HEAD(self) -> None:
                self._note("HEAD")
                stored = stub.objects.get(self._key())
                self._reply(
                    200 if stored is not None else 404, length=len(stored or b"")
                )

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def test_a_recording_is_matched_by_room_and_by_time() -> None:
    root = Path(tempfile.mkdtemp())
    _jibri_recording(root, "Standup", _MEETING_EPOCH - 7200, session="old")
    latest = _jibri_recording(root, "Standup", _MEETING_EPOCH + 60, session="new")
    _jibri_recording(root, "Retrospective", _MEETING_EPOCH + 30, session="other")

    found = s3_upload.find_recording(
        root,
        "Standup",
        not_before=_MEETING_EPOCH,
        settled_before=_MEETING_EPOCH + 120,
    )
    assert found == latest


def test_a_recording_still_being_written_is_left_alone() -> None:
    """An ffmpeg flushing its last frames must not be uploaded half-made."""
    root = Path(tempfile.mkdtemp())
    _jibri_recording(root, "Standup", _MEETING_EPOCH + 100)

    assert (
        s3_upload.find_recording(
            root,
            "Standup",
            not_before=_MEETING_EPOCH,
            settled_before=_MEETING_EPOCH + 99,
        )
        is None
    )


def test_a_recording_from_before_the_meeting_is_not_this_meeting_s() -> None:
    """The previous meeting in the same room left its own recording behind."""
    root = Path(tempfile.mkdtemp())
    _jibri_recording(root, "Standup", _MEETING_EPOCH - 3600)

    assert (
        s3_upload.find_recording(
            root,
            "Standup",
            not_before=_MEETING_EPOCH,
            settled_before=_MEETING_EPOCH + 600,
        )
        is None
    )


def test_the_room_is_read_from_jibris_own_metadata_first() -> None:
    """A name Jibri was told to use still matches the room it recorded."""
    root = Path(tempfile.mkdtemp())
    recording = _jibri_recording(
        root, "Standup", _MEETING_EPOCH, filename="recording_123.mp4", meeting_url=None
    )
    # The filename says nothing; only the metadata knows the room.
    directory = recording.parent
    (directory / "metadata.json").write_text(
        json.dumps({"meeting_url": "https://jitsi.example.com/Standup?jwt=abc#config"}),
        encoding="utf-8",
    )

    assert (
        s3_upload.find_recording(
            root,
            "Standup",
            not_before=_MEETING_EPOCH - 60,
            settled_before=_MEETING_EPOCH + 600,
        )
        == recording
    )


def test_rooms_are_compared_without_case_or_punctuation() -> None:
    root = Path(tempfile.mkdtemp())
    recording = _jibri_recording(root, "Daily-Standup", _MEETING_EPOCH)

    assert (
        s3_upload.find_recording(
            root,
            "daily standup",
            not_before=_MEETING_EPOCH - 60,
            settled_before=_MEETING_EPOCH + 600,
        )
        == recording
    )


def test_a_recording_another_meeting_already_took_is_not_taken_again(
    tmp_path: Path,
) -> None:
    config = _s3_config(tmp_path)
    recording = _jibri_recording(config.s3.jibri_dir, "Standup", _MEETING_EPOCH)
    earlier = _meeting_dir(tmp_path, session="earlier")
    s3_upload.write_claim(
        earlier, {"source": str(recording), "bucket": "b", "key": "k"}
    )

    claimed = s3_upload.claimed_recordings(config.storage.recordings_dir)
    assert claimed == {recording}
    assert (
        s3_upload.find_recording(
            config.s3.jibri_dir,
            "Standup",
            not_before=_MEETING_EPOCH - 60,
            settled_before=_MEETING_EPOCH + 600,
            claimed=claimed,
        )
        is None
    )


def test_the_object_key_keeps_the_room_and_jibris_filename() -> None:
    assert (
        s3_upload.object_key(
            "meetings", "Daily Standup", Path("/x/Daily Standup_2026-01-02-03-04-05.mp4")
        )
        == "meetings/Daily_Standup/Daily_Standup_2026-01-02-03-04-05.mp4"
    )
    assert s3_upload.object_key("", "Standup", Path("/x/s.mp4")) == "Standup/s.mp4"
    assert s3_upload.object_key("/a/b/", "Room", Path("/x/s.mp4")) == "a/b/Room/s.mp4"


def test_a_room_name_cannot_climb_out_of_the_prefix() -> None:
    """A room name reaches us from the meeting; a key is a path."""
    key = s3_upload.object_key("meetings", "../../etc", Path("/x/../../passwd.mp4"))
    assert key.startswith("meetings/")
    assert ".." not in key
    assert key.count("/") == 2


def test_the_upload_writes_the_object_and_a_claim_beside_the_transcript(
    tmp_path: Path,
) -> None:
    pytest.importorskip("boto3")
    stub = _StubS3()
    try:
        config = _s3_config(tmp_path, endpoint=stub.url, prefix="videos")
        meeting = _meeting_dir(tmp_path)
        recording = _jibri_recording(config.s3.jibri_dir, "Standup", _MEETING_EPOCH + 10)

        assert s3_upload.upload_meeting_video(meeting, config, wait=False)

        key = f"videos/Standup/{recording.name}"
        assert stub.objects[key] == recording.read_bytes()
        methods = [method for method, _, _ in stub.requests]
        assert methods == ["PUT", "HEAD"]
        # Signed, at the path the endpoint gave, with the configured credentials.
        assert all(auth.startswith("AWS4-HMAC-SHA256") for _, _, auth in stub.requests)
        assert all(path == f"/meetings-recordings/{key}" for _, path, _ in stub.requests)

        claim = s3_upload.read_claim(meeting / s3_upload.CLAIM_FILENAME)
        assert claim["bucket"] == "meetings-recordings"
        assert claim["key"] == key
        assert claim["source"] == str(recording)
        assert claim["size"] == len(recording.read_bytes())
        assert claim["confirmed_size"] == claim["size"]
        assert claim["url"] == f"{stub.url}/meetings-recordings/{key}"
    finally:
        stub.close()


def test_a_recording_larger_than_the_multipart_threshold_is_uploaded_in_parts(
    tmp_path: Path,
) -> None:
    """What every real recording does: boto3 multiparts anything over 8 MiB."""
    pytest.importorskip("boto3")
    stub = _StubS3()
    try:
        config = _s3_config(tmp_path, endpoint=stub.url, prefix="videos")
        meeting = _meeting_dir(tmp_path)
        recording = _jibri_recording(config.s3.jibri_dir, "Standup", _MEETING_EPOCH + 10)
        recording.write_bytes(b"mp4" * ((9 << 20) // 3))  # 9 MiB, so two parts

        assert s3_upload.upload_meeting_video(meeting, config, wait=False)

        key = f"videos/Standup/{recording.name}"
        assert stub.objects[key] == recording.read_bytes()
        methods = [method for method, _, _ in stub.requests]
        assert methods.count("POST") == 2, methods  # initiated, then completed
        posts = [path for method, path, _ in stub.requests if method == "POST"]
        assert any("uploads" in path for path in posts)
        assert sum(1 for method, _, _ in stub.requests if method == "PUT") >= 2
    finally:
        stub.close()


def test_an_uploaded_recording_is_not_uploaded_twice(tmp_path: Path) -> None:
    pytest.importorskip("boto3")
    stub = _StubS3()
    try:
        config = _s3_config(tmp_path, endpoint=stub.url)
        meeting = _meeting_dir(tmp_path)
        _jibri_recording(config.s3.jibri_dir, "Standup", _MEETING_EPOCH + 10)

        assert s3_upload.upload_meeting_video(meeting, config, wait=False)
        assert s3_upload.upload_meeting_video(meeting, config, wait=False)
        assert [method for method, _, _ in stub.requests] == ["PUT", "HEAD"]
    finally:
        stub.close()


def test_nothing_is_uploaded_when_no_endpoint_is_configured(tmp_path: Path) -> None:
    config = _s3_config(tmp_path, endpoint="", bucket="")
    meeting = _meeting_dir(tmp_path)
    _jibri_recording(config.s3.jibri_dir, "Standup", _MEETING_EPOCH + 10)

    assert not s3_upload.upload_meeting_video(meeting, config, wait=False)
    assert not (meeting / s3_upload.CLAIM_FILENAME).exists()


def test_a_meeting_with_no_recording_claims_nothing(tmp_path: Path, caplog) -> None:
    config = _s3_config(tmp_path, jibri_dir=tmp_path / "jibri-is-not-there")
    meeting = _meeting_dir(tmp_path)

    with caplog.at_level("INFO"):
        assert not s3_upload.upload_meeting_video(meeting, config, wait=False)
    assert not (meeting / s3_upload.CLAIM_FILENAME).exists()
    assert "no recording" in caplog.text


def test_a_recording_that_is_not_there_yet_is_waited_for(
    tmp_path: Path, monkeypatch
) -> None:
    """Jibri finalizes the recording after the meeting ends, not with it."""
    pytest.importorskip("boto3")
    stub = _StubS3()
    try:
        config = _s3_config(tmp_path, endpoint=stub.url, wait_seconds=30.0)
        meeting = _meeting_dir(tmp_path)
        monkeypatch.setattr(s3_upload, "POLL_SECONDS", 0.01)

        # The recording appears only once the daemon has started looking,
        # which is what a Jibri taking its time over the finalize looks like.
        def appear() -> None:
            time.sleep(0.05)
            _jibri_recording(config.s3.jibri_dir, "Standup", _MEETING_EPOCH + 10)

        writer = threading.Thread(target=appear)
        writer.start()
        try:
            assert s3_upload.upload_meeting_video(meeting, config)
        finally:
            writer.join()
        assert (meeting / s3_upload.CLAIM_FILENAME).exists()
    finally:
        stub.close()


def test_a_broken_endpoint_costs_the_video_and_nothing_else(
    tmp_path: Path, monkeypatch
) -> None:
    config = _s3_config(tmp_path)
    meeting = _meeting_dir(tmp_path)
    _jibri_recording(config.s3.jibri_dir, "Standup", _MEETING_EPOCH + 10)

    def explode(s3: object):
        raise OSError("connection refused")

    monkeypatch.setattr(s3_upload, "_client", explode)
    assert not s3_upload.upload_meeting_video(meeting, config, wait=False)
    assert not (meeting / s3_upload.CLAIM_FILENAME).exists()


def test_the_local_recording_is_kept_unless_the_endpoint_confirms_it(
    tmp_path: Path,
) -> None:
    pytest.importorskip("boto3")
    stub = _StubS3()
    try:
        config = _s3_config(
            tmp_path, endpoint=stub.url, delete_after_upload=True
        )
        meeting = _meeting_dir(tmp_path)
        recording = _jibri_recording(config.s3.jibri_dir, "Standup", _MEETING_EPOCH + 10)

        assert s3_upload.upload_meeting_video(meeting, config, wait=False)
        assert not recording.exists(), "a confirmed upload should free the disk"
        # The directory Jibri made is left behind; only the recording was ours.
        assert recording.parent.is_dir()
    finally:
        stub.close()


def test_a_local_recording_that_grew_after_the_upload_is_kept(
    tmp_path: Path, monkeypatch
) -> None:
    """The endpoint's copy is not the whole recording, so the file stays."""
    config = _s3_config(tmp_path, delete_after_upload=True)
    meeting = _meeting_dir(tmp_path)
    recording = _jibri_recording(config.s3.jibri_dir, "Standup", _MEETING_EPOCH + 10)

    class ShortOne:
        def upload_file(self, path: str, bucket: str, key: str) -> None:
            pass

        def head_object(self, **kwargs: object) -> dict:
            return {"ContentLength": 4}  # fewer bytes than the file holds

    monkeypatch.setattr(s3_upload, "_client", lambda s3: ShortOne())
    assert s3_upload.upload_meeting_video(meeting, config, wait=False)
    assert recording.exists()


def test_the_log_says_what_was_there_instead(tmp_path: Path, caplog) -> None:
    config = _s3_config(tmp_path)
    _jibri_recording(config.s3.jibri_dir, "Retrospective", _MEETING_EPOCH)
    _jibri_recording(config.s3.jibri_dir, "Old", _MEETING_EPOCH - 86400, session="old")
    meeting = _meeting_dir(tmp_path, room="Standup")

    with caplog.at_level("INFO"):
        assert not s3_upload.upload_meeting_video(meeting, config, wait=False)
    assert "Retrospective" in caplog.text
    assert "outside the meeting" in caplog.text


def test_the_meeting_start_comes_from_the_timeline_when_there_is_one(
    tmp_path: Path,
) -> None:
    meeting = _meeting_dir(tmp_path)
    timeline = SessionTimeline(
        started_at="2026-01-02T03:04:05+00:00", duration=60.0, turns=[]
    )
    expected = datetime.fromisoformat("2026-01-02T03:04:05+00:00").timestamp()
    assert s3_upload.meeting_started_epoch(meeting, timeline) == expected
    # Without one, the directory's own timestamp stands in for the meeting's.
    assert s3_upload.meeting_started_epoch(meeting, None) == pytest.approx(
        meeting.stat().st_mtime
    )


def test_the_upload_command_needs_an_endpoint(tmp_path: Path) -> None:
    config = _s3_config(tmp_path, endpoint="", bucket="")
    assert daemon_module.upload_directory_video(tmp_path, config) == 2
    assert daemon_module.upload_directory_video(tmp_path / "nope", config) == 2


def test_the_upload_command_reports_what_it_did(tmp_path: Path, monkeypatch) -> None:
    config = _s3_config(tmp_path)
    monkeypatch.setattr(
        daemon_module, "upload_meeting_video", lambda *a, **k: True
    )
    assert daemon_module.upload_directory_video(tmp_path, config) == 0
    monkeypatch.setattr(daemon_module, "upload_meeting_video", lambda *a, **k: False)
    assert daemon_module.upload_directory_video(tmp_path, config) == 1


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
