# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests for the Opus encoder and the test-environment tooling.

The encoder lives in the package because both the test suite and the tools
need to produce real Opus packets; testing it here keeps the round-trip
guarantee in one place. Nothing in this file touches the network.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import wave
from datetime import datetime
from pathlib import Path

import pytest

from jitsi_audio_bridge.audio import OpusDecoder, OpusEncoder, OpusError, parse_metadata
from tools import recover_meeting
from tools.send_meeting import (
    DEFAULT_RATE,
    ENCODABLE_RATES,
    FRAME_MS,
    Participant,
    build_parser,
    build_participants,
    frame_samples,
    load_wav_frames,
    media_json_info,
    media_json_media,
    media_json_ping,
    media_json_session_end,
    media_json_start,
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


# --------------------------------------------------------------------------
# Media-json sender
# --------------------------------------------------------------------------


def _participant() -> Participant:
    return Participant(
        identifier="alice", name="Alice", email="alice@example.com", sample_rate=48000
    )


def test_protocol_defaults_to_binary_and_accepts_media_json() -> None:
    assert build_parser().parse_args([]).protocol == "binary"
    assert build_parser().parse_args(["--protocol", "media-json"]).protocol == "media-json"


def test_media_json_tags_are_distinct_from_endpoint_ids() -> None:
    participant = Participant(identifier="alice", name="Alice", email="alice@example.com")
    assert participant.source_tag == "alice-audio"
    assert participant.endpoint_id == "endpoint-alice"
    assert participant.source_tag != participant.endpoint_id


def test_media_json_start_announces_the_format_and_endpoint() -> None:
    event = media_json_start(_participant(), 3)
    assert event["event"] == "start"
    assert event["sequenceNumber"] == "3"
    start = event["start"]
    assert start["tag"] == "alice-audio"
    assert start["mediaFormat"]["encoding"] == "opus"
    assert start["mediaFormat"]["sampleRate"] == 48000
    assert start["customParameters"]["endpointId"] == "endpoint-alice"


def test_media_json_media_carries_an_opus_packet_as_base64() -> None:
    packet = b"\x01\x02\x03\xff"
    event = media_json_media(_participant(), 2, 42, 1234567, packet)
    assert event["event"] == "media"
    assert event["sequenceNumber"] == "2"
    media = event["media"]
    assert media["tag"] == "alice-audio"
    # The numeric fields inherited from VoxImplant are strings on the wire.
    assert media["chunk"] == "42"
    assert media["timestamp"] == "1234567"
    assert base64.b64decode(media["payload"], validate=True) == packet


def test_media_json_ping_and_session_end_shapes() -> None:
    assert media_json_ping(7) == {"event": "ping", "id": 7}
    assert media_json_session_end() == {"event": "session-end"}


def test_media_json_info_names_the_bridge() -> None:
    info = media_json_info()
    assert info["event"] == "info"
    assert info["application"] == "jitsi-videobridge"


# --------------------------------------------------------------------------
# Recovering a meeting whose people went missing
# --------------------------------------------------------------------------


def _broken_meeting(root: Path, *, room: str = "1ff2f1ff-4bab", with_parts: bool = True):
    """A meeting directory as this daemon leaves one that lost its people."""
    from dataclasses import replace

    from jitsi_audio_bridge.config import load_config
    from jitsi_audio_bridge.timeline import SessionTimeline

    recordings = root / "recordings"
    meeting = recordings / "53f025af"
    meeting.mkdir(parents=True)
    (meeting / "metadata.json").write_text(
        json.dumps({"room_name": room, "participants": {}, "meeting_id": "53f025af"}),
        encoding="utf-8",
    )
    (meeting / "transcript.txt").write_text(
        "[participant-27edee51-a0]: buongiorno a tutti\n", encoding="utf-8"
    )
    (meeting / "summary.md").write_text("## Punti chiave\n- uno\n", encoding="utf-8")
    SessionTimeline(started_at="2026-10-13T09:47:00+00:00", duration=800.0).write(
        meeting / "timeline.json"
    )

    jibri = root / "jibri-recordings" / "908c74d7"
    jibri.mkdir(parents=True)
    if with_parts:
        (jibri / "metadata.json").write_text(
            json.dumps(
                {
                    "meeting_url": f"https://jitsi.example.com/{room}",
                    "participants": [
                        {"user": {"name": "annachiara.gallo"}},
                        {"user": {"name": "Sezen Tosunoğlu", "email": "sezen@example.com"}},
                    ],
                }
            ),
            encoding="utf-8",
        )
    recording = jibri / f"{room}_2026-10-13-09-47-21.mp4"
    recording.write_bytes(b"\x00" * 4096)
    # Recorded when the meeting was, not when this test ran: a file written
    # seconds ago is one the daemon leaves alone until it settles.
    stopped = datetime.fromisoformat("2026-10-13T10:00:00+00:00").timestamp()
    for path in jibri.iterdir():
        os.utime(path, (stopped, stopped))

    base = load_config()
    config = replace(
        base,
        storage=replace(base.storage, recordings_dir=recordings),
        s3=replace(base.s3, jibri_dir=root / "jibri-recordings"),
    )
    return meeting, config


def test_recovery_finds_the_people_jibri_saw(tmp_path: Path) -> None:
    meeting_dir, config = _broken_meeting(tmp_path)

    meeting = recover_meeting.inspect(meeting_dir, config)

    assert meeting.room_name == "1ff2f1ff-4bab"
    assert meeting.participants == [] and meeting.recipients == []
    assert meeting.transcript == meeting_dir / "transcript.txt"
    assert meeting.summary == meeting_dir / "summary.md"
    assert meeting.jibri_metadata is not None
    assert meeting.recording is not None and meeting.recording.suffix == ".mp4"

    found = recover_meeting.people_from_jibri(meeting.jibri_metadata)
    assert found["recipients"] == ["sezen@example.com"]
    assert "Sezen Tosunoğlu (sezen@example.com)" in found["participants"]
    # Nothing was written by looking.
    assert json.loads((meeting_dir / "metadata.json").read_text())["participants"] == {}


def test_recovery_keeps_the_names_and_the_file_it_replaced(tmp_path: Path) -> None:
    meeting_dir, config = _broken_meeting(tmp_path)
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert recover_meeting.recover(meeting, config, write=True) == 0

    written = json.loads((meeting_dir / "metadata.json").read_text(encoding="utf-8"))
    # Jibri's file, kept as it was: the participant with an address, and the
    # one with only a name, both.
    assert [entry["user"].get("email") for entry in written["participants"]] == [
        None, "sezen@example.com",
    ]
    backup = meeting_dir / recover_meeting.BACKUP_NAME
    assert backup.is_file()
    assert json.loads(backup.read_text(encoding="utf-8"))["participants"] == {}


def test_recovery_sends_the_mail_to_the_recovered_people(tmp_path: Path) -> None:
    sent: list = []
    meeting_dir, config = _broken_meeting(tmp_path)

    def fake_mail(recipients, room, summary, transcript, summary_path, smtp, **kwargs):
        sent.append((recipients, room, kwargs.get("started_at"), kwargs.get("language")))
        return True

    meeting = recover_meeting.inspect(meeting_dir, config)
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(recover_meeting, "send_meeting_email", fake_mail)
        monkeypatch.setattr(recover_meeting, "detect_language", lambda text, endpoint: "Italian")
        assert recover_meeting.recover(meeting, config, mail=True) == 0
    finally:
        monkeypatch.undo()

    recipients, room, started_at, language = sent[0]
    assert recipients == ["sezen@example.com"]
    assert room == "1ff2f1ff-4bab"
    assert started_at == "2026-10-13T09:47:00+00:00"
    assert language == "Italian"


def test_recovery_says_so_when_there_is_nothing_to_recover(tmp_path: Path) -> None:
    """No transcript means this tool is the wrong tool, and it says which is right."""
    meeting_dir, config = _broken_meeting(tmp_path)
    (meeting_dir / "transcript.txt").unlink()
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert recover_meeting.recover(meeting, config, mail=True) == 1


def test_recovery_without_a_recording_leaves_the_meeting_alone(tmp_path: Path) -> None:
    meeting_dir, config = _broken_meeting(tmp_path)
    for path in (config.s3.jibri_dir).rglob("*"):
        if path.is_file():
            path.unlink()
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert meeting.jibri_metadata is None and meeting.recording is None
    assert recover_meeting.recover(meeting, config, write=True) == 0
    assert json.loads((meeting_dir / "metadata.json").read_text())["participants"] == {}


def _broken_meeting_without_a_recording(tmp_path: Path, log_text: str):
    """A meeting that lost its people, with no recording to take them from."""
    meeting_dir, config = _broken_meeting(tmp_path)
    for path in (config.s3.jibri_dir).rglob("*"):
        if path.is_file():
            path.unlink()
    log = tmp_path / "jicofo.log"
    log.write_text(log_text, encoding="utf-8")
    return meeting_dir, config, log


def test_recovery_falls_back_to_the_people_jicofo_saw(tmp_path: Path) -> None:
    """No module, no recording -- but Jicofo logged who was in the call."""
    meeting_dir, config, log = _broken_meeting_without_a_recording(tmp_path, JICOFO_LOG)
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert recover_meeting.recover(meeting, config, write=True, jicofo_logs=[log]) == 0

    written = json.loads((meeting_dir / "metadata.json").read_text(encoding="utf-8"))
    assert written["source"] == "jicofo_attribution"
    assert written["room_name"] == "1ff2f1ff-4bab"
    assert [entry["email"] for entry in written["participants"]] == [
        "sezen@example.com",
        "patrick@example.com",
    ]
    # The endpoint is kept as the participant id: that is what the transcript's
    # labels are built from, so an attribution still has something to match.
    assert [entry["id"] for entry in written["participants"]] == ["76e96806", "e53be326"]

    backup = meeting_dir / recover_meeting.BACKUP_NAME
    assert json.loads(backup.read_text(encoding="utf-8"))["participants"] == {}


def test_the_jicofo_fallback_addresses_the_mail(tmp_path: Path) -> None:
    """The addresses are the whole point: the mail reaches the people."""
    sent: list = []
    meeting_dir, config, log = _broken_meeting_without_a_recording(tmp_path, JICOFO_LOG)

    def fake_mail(recipients, room, summary, transcript, summary_path, smtp, **kwargs):
        sent.append((recipients, room))
        return True

    meeting = recover_meeting.inspect(meeting_dir, config)
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(recover_meeting, "send_meeting_email", fake_mail)
        monkeypatch.setattr(
            recover_meeting, "detect_language", lambda text, endpoint: "English"
        )
        assert recover_meeting.recover(meeting, config, mail=True, jicofo_logs=[log]) == 0
    finally:
        monkeypatch.undo()

    assert sent[0] == (["patrick@example.com", "sezen@example.com"], "1ff2f1ff-4bab")


def test_a_recovered_mail_can_carry_the_recording(tmp_path: Path, monkeypatch) -> None:
    """`--video` archives the recording the meeting could not find, and the mail
    that goes out with it links there.

    The link is the whole point of uploading before mailing: a URL that stops
    working is worse than no URL.  Which is why the recovery asks for the link
    the upload produced rather than reporting the upload and sending a mail
    without it.
    """
    from dataclasses import replace

    sent: dict = {}
    meeting_dir, config = _broken_meeting(tmp_path)
    config = replace(
        config,
        s3=replace(config.s3, endpoint="http://127.0.0.1:1", bucket="meetings"),
    )
    log = tmp_path / "jicofo.log"
    log.write_text(JICOFO_LOG, encoding="utf-8")

    def fake_mail(recipients, room, summary, transcript, summary_path, smtp, **kwargs):
        sent.update(kwargs, recipients=recipients)
        return True

    link = recover_meeting.VideoLink(
        url="https://minio.example.com/meetings/k.mp4",
        expires_at="2026-10-22T10:00:00+00:00",
    )
    monkeypatch.setattr(recover_meeting, "send_meeting_email", fake_mail)
    monkeypatch.setattr(recover_meeting, "detect_language", lambda text, endpoint: "English")
    monkeypatch.setattr(recover_meeting, "prepare_video", lambda *a, **k: link)

    meeting = recover_meeting.inspect(meeting_dir, config)
    assert recover_meeting.recover(meeting, config, video=True, mail=True, jicofo_logs=[log]) == 0

    assert sent["video_url"] == link.url
    assert sent["video_until"] == link.expires_at


def test_a_meeting_that_already_names_people_keeps_them(tmp_path: Path) -> None:
    """Jicofo is the last resort, not the first: a recording's names win."""
    meeting_dir, config = _broken_meeting(tmp_path)
    log = tmp_path / "jicofo.log"
    log.write_text(JICOFO_LOG, encoding="utf-8")
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert recover_meeting.recover(meeting, config, write=True, jicofo_logs=[log]) == 0

    written = json.loads((meeting_dir / "metadata.json").read_text(encoding="utf-8"))
    assert [entry["user"].get("email") for entry in written["participants"]] == [
        None,
        "sezen@example.com",
    ]


def assert_metadata_names_somebody(meeting_dir: Path, written: str) -> None:
    """The invariant the recovery has to keep: nobody is not a recovery.

    A metadata.json the daemon reads as an empty meeting is exactly the state
    these tools exist to get out of, so whatever is written has to leave at
    least one person in it -- an address to mail, or a name to attribute.
    """
    assert isinstance(json.loads(written), dict), written
    info = parse_metadata(meeting_dir)
    assert info["recipients"] or info["id_to_name"], (
        f"the metadata names nobody: {written}"
    )


def test_the_recent_meetings_list_is_newest_first(tmp_path: Path) -> None:
    meeting_dir, config = _broken_meeting(tmp_path)
    older = config.storage.recordings_dir / "older-meeting"
    older.mkdir()
    os.utime(older, (1_600_000_000, 1_600_000_000))

    listed = recover_meeting.recent_meetings(config, 10)

    assert listed[0] == meeting_dir and older in listed
    # The bridge's own hidden directories are not meetings.
    hidden = config.storage.recordings_dir / ".session-metadata"
    hidden.mkdir()
    assert hidden not in recover_meeting.recent_meetings(config, 10)


#: Two lines from a real Jicofo log, which is where the endpoint ids the audio
#: is tagged with meet the identities the deployment gave those participants.
JICOFO_LOG = (
    "Jicofo 2026-10-13 09:46:56.403 INFO: [223] "
    "[room=1ff2f1ff@conference.example.com meeting_id=53f025af] "
    "JitsiMeetConferenceImpl.onMemberJoined#873: Member joined:76e96806 "
    "stats-id=sezen@example.com region=null audioMuted=true role=PARTICIPANT\n"
    "Jicofo 2026-10-13 09:46:58.930 INFO: [225] "
    "[room=1ff2f1ff@conference.example.com meeting_id=53f025af] "
    "JitsiMeetConferenceImpl.onMemberJoined#873: Member joined:e53be326 "
    "stats-id=patrick@example.com region=null audioMuted=true role=PARTICIPANT\n"
    "Jicofo 2026-10-13 09:47:01.100 INFO: [227] "
    "[room=other@conference.example.com meeting_id=another-meeting] "
    "JitsiMeetConferenceImpl.onMemberJoined#873: Member joined:deadbeef "
    "stats-id=somebody@example.com region=null\n"
)


def _meeting_with_labels(tmp_path: Path):
    from dataclasses import replace

    from jitsi_audio_bridge.config import load_config

    recordings = tmp_path / "recordings"
    meeting = recordings / "53f025af"
    meeting.mkdir(parents=True)
    (meeting / "metadata.json").write_text(
        json.dumps(
            {
                "room_name": "1ff2f1ff",
                "participants": [
                    {"user": {"name": "Sezen Tosunoğlu", "email": "sezen@example.com"}},
                    {"user": {"name": "Patrick Barsanti", "email": "patrick@example.com"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    (meeting / "transcript.txt").write_text(
        "[participant-76e96806-a0]: buongiorno\n[participant-e53be326-a0]: ciao\n",
        encoding="utf-8",
    )
    (meeting / "summary.md").write_text(
        "## Punti chiave\n- participant-76e96806-a0 ha aperto la riunione\n", encoding="utf-8"
    )
    log = tmp_path / "jicofo.log"
    log.write_text(JICOFO_LOG, encoding="utf-8")

    base = load_config()
    config = replace(
        base, storage=replace(base.storage, recordings_dir=recordings)
    )
    return meeting, config, log


def test_the_endpoint_ids_come_from_jicofo_for_this_meeting_only(tmp_path: Path) -> None:
    lines = tmp_path / "jicofo.log"
    lines.write_text(JICOFO_LOG, encoding="utf-8")

    found = recover_meeting.endpoints_from_jicofo([lines], "53f025af")

    assert found == {"76e96806": "sezen@example.com", "e53be326": "patrick@example.com"}
    assert "deadbeef" not in found, "another meeting's participants leaked in"


def test_attribution_names_the_labels_and_leaves_the_words(tmp_path: Path) -> None:
    meeting_dir, config, log = _meeting_with_labels(tmp_path)
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert recover_meeting.recover(
        meeting, config, attribute=True, jicofo_logs=[log]
    ) == 0

    transcript = (meeting_dir / "transcript.txt").read_text(encoding="utf-8")
    assert transcript == "[Sezen Tosunoğlu]: buongiorno\n[Patrick Barsanti]: ciao\n"
    summary = (meeting_dir / "summary.md").read_text(encoding="utf-8")
    assert "Sezen Tosunoğlu ha aperto la riunione" in summary
    # What was said is untouched, and the previous text is kept.
    assert (meeting_dir / "transcript.txt.before-attribution").read_text(
        encoding="utf-8"
    ).startswith("[participant-76e96806-a0]:")


def test_attribution_does_nothing_without_the_flag(tmp_path: Path) -> None:
    meeting_dir, config, log = _meeting_with_labels(tmp_path)
    original = (meeting_dir / "transcript.txt").read_text(encoding="utf-8")
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert recover_meeting.recover(meeting, config, jicofo_logs=[log]) == 0

    assert (meeting_dir / "transcript.txt").read_text(encoding="utf-8") == original
    assert not (meeting_dir / "transcript.txt.before-attribution").exists()


def test_attribution_says_so_when_the_log_has_no_such_meeting(tmp_path: Path) -> None:
    """Rotated away, or the meeting ran on another host: no mapping, no edit."""
    meeting_dir, config, _ = _meeting_with_labels(tmp_path)
    empty = tmp_path / "empty.log"
    empty.write_text("nothing about any meeting here\n", encoding="utf-8")
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert recover_meeting.recover(
        meeting, config, attribute=True, jicofo_logs=[empty]
    ) == 0
    assert (meeting_dir / "transcript.txt").read_text(encoding="utf-8").startswith(
        "[participant-76e96806-a0]:"
    )


def test_a_compressed_log_reads_too(tmp_path: Path) -> None:
    import gzip as gzip_module

    meeting_dir, config, log = _meeting_with_labels(tmp_path)
    rotated = tmp_path / "jicofo.log.2.gz"
    with gzip_module.open(rotated, "wt", encoding="utf-8") as handle:
        handle.write(JICOFO_LOG)

    assert recover_meeting.endpoints_from_jicofo([rotated], "53f025af") == {
        "76e96806": "sezen@example.com",
        "e53be326": "patrick@example.com",
    }


#: Jicofo's identity for a participant, without an address: a deployment that
#: does not authenticate users has only this.
JICOFO_LOG_NAMES_ONLY = (
    "Jicofo 2026-10-13 09:47:01.100 INFO: [227] "
    "[room=1ff2f1ff@conference.example.com meeting_id=53f025af] "
    "JitsiMeetConferenceImpl.onMemberJoined#873: Member joined:27edee51 "
    "stats-id=annachiara.gallo region=null\n"
)


def test_a_name_without_an_address_is_still_a_name(tmp_path: Path) -> None:
    """What Jicofo calls somebody is a name, not an address to send mail to.

    Written into the address field it would read as a participant with neither
    an address nor a name -- the daemon drops those -- and the recovery would
    have written a metadata document naming nobody, which is the failure it
    exists to repair.
    """
    meeting_dir, config, log = _broken_meeting_without_a_recording(
        tmp_path, JICOFO_LOG_NAMES_ONLY
    )
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert recover_meeting.recover(meeting, config, write=True, jicofo_logs=[log]) == 0

    written = (meeting_dir / "metadata.json").read_text(encoding="utf-8")
    assert "annachiara.gallo" in written
    assert_metadata_names_somebody(meeting_dir, written)


def test_whatever_is_written_names_somebody(tmp_path: Path) -> None:
    """The invariant: a written metadata.json never reads as an empty meeting."""
    meeting_dir, config, log = _broken_meeting_without_a_recording(tmp_path, JICOFO_LOG)
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert recover_meeting.recover(meeting, config, write=True, jicofo_logs=[log]) == 0

    written = (meeting_dir / "metadata.json").read_text(encoding="utf-8")
    assert_metadata_names_somebody(meeting_dir, written)
    # Which is to say: the addresses are there for the mail.  The join line
    # carries no display name, so the endpoint ids map to the addresses rather
    # than to names -- which is what the attribution substitutes anyway.
    info = parse_metadata(meeting_dir)
    assert info["recipients"] == ["patrick@example.com", "sezen@example.com"]
    endpoints = {
        entry["id"] for entry in json.loads(written)["participants"]
    }
    assert endpoints == {"76e96806", "e53be326"}, written


def test_a_document_that_names_nobody_is_not_written(tmp_path: Path, monkeypatch) -> None:
    """The guard, for the day something upstream starts producing empty names."""
    meeting_dir, config, log = _broken_meeting_without_a_recording(tmp_path, JICOFO_LOG)
    before = (meeting_dir / "metadata.json").read_text(encoding="utf-8")
    meeting = recover_meeting.inspect(meeting_dir, config)
    monkeypatch.setattr(
        recover_meeting,
        "metadata_from_jicofo",
        lambda mapping, room, meeting_id: json.dumps(
            {"room_name": room, "meeting_id": meeting_id, "participants": []}
        ),
    )

    assert recover_meeting.recover(meeting, config, write=True, jicofo_logs=[log]) == 0

    assert (meeting_dir / "metadata.json").read_text(encoding="utf-8") == before
    assert not (meeting_dir / recover_meeting.BACKUP_NAME).exists()


def _labelled_transcript(meeting_dir: Path) -> Path:
    """A transcript whose labels are the endpoints Jicofo logged."""
    path = meeting_dir / "transcript.txt"
    path.write_text(
        "[participant-76e96806-a0]: buongiorno a tutti\n"
        "[participant-e53be326-a0]: buongiorno\n",
        encoding="utf-8",
    )
    return path


def test_an_address_is_not_a_name_in_a_transcript(tmp_path: Path) -> None:
    """The label a reader sees: the meeting went to a person, not to an inbox.

    Jicofo logs the address a participant authenticated as and nothing else,
    so a meeting recovered from its log alone has an address per person --
    which is what a transcript label must not become.  The address stays in
    the metadata, which is what the mail uses.
    """
    assert recover_meeting.name_from_address("michael@amarulasolutions.com") == "michael"
    assert recover_meeting.name_from_address("annachiara.gallo") == "annachiara.gallo"
    assert recover_meeting.name_from_address("") == ""

    meeting_dir, config, log = _broken_meeting_without_a_recording(tmp_path, JICOFO_LOG)
    transcript = _labelled_transcript(meeting_dir)
    meeting = recover_meeting.inspect(meeting_dir, config)
    assert recover_meeting.recover(
        meeting, config, write=True, attribute=True, jicofo_logs=[log]
    ) == 0

    named = transcript.read_text(encoding="utf-8")
    assert "[sezen@example.com]" not in named
    assert "[sezen]: buongiorno a tutti" in named
    assert "[patrick]: buongiorno" in named
    # And the mail still has the addresses to send to.
    assert parse_metadata(meeting_dir)["recipients"] == [
        "patrick@example.com",
        "sezen@example.com",
    ]


def test_a_name_from_the_recording_names_the_labels(tmp_path: Path) -> None:
    """The recording knows what the log does not: what people are called.

    The meeting's metadata is the log's -- an address per person, no names --
    and it is the only thing the meeting will ever say, because a meeting that
    names somebody is never rewritten.  The names are in the file beside the
    recording, so a label is a name and the mail still has the addresses.
    """
    meeting_dir, config, log = _broken_meeting_without_a_recording(tmp_path, JICOFO_LOG)
    transcript = _labelled_transcript(meeting_dir)
    # What the first recovery wrote: the people, from the log, with no names
    # in it (the log has none).
    (meeting_dir / "metadata.json").write_text(
        recover_meeting.metadata_from_jicofo(
            {"76e96806": "sezen@example.com", "e53be326": "patrick@example.com"},
            "1ff2f1ff-4bab",
            "53f025af",
        ),
        encoding="utf-8",
    )
    recorded = _jibri_metadata_file(tmp_path, {
        "meeting_url": "https://jitsi.example.com/1ff2f1ff-4bab",
        "participants": [
            {"user": {"name": "Sezen Tosunoğlu", "email": "sezen@example.com"}},
            {"user": {"name": "Patrick Barsanti", "email": "patrick@example.com"}},
        ],
    })
    meeting = recover_meeting.inspect(meeting_dir, config)
    meeting.jibri_metadata = recorded

    assert recover_meeting.recover(
        meeting, config, attribute=True, jicofo_logs=[log]
    ) == 0

    named = transcript.read_text(encoding="utf-8")
    assert "[Sezen Tosunoğlu]: buongiorno a tutti" in named
    assert "[Patrick Barsanti]: buongiorno" in named
    # The metadata is left as it was: it names people, so nothing rewrites it,
    # and it is what the mail is addressed from.
    assert parse_metadata(meeting_dir)["recipients"] == [
        "patrick@example.com",
        "sezen@example.com",
    ]


def test_a_timed_transcript_is_repaired_too(tmp_path: Path) -> None:
    """The shape the daemon writes when the timeline interleaves a meeting.

    Its labels are not in brackets -- the offset is, and the speaker follows
    the bracket and a colon.  A meeting attributed once already, to the
    addresses from the log, has to be repairable in both shapes, or the
    summary ends up fixed and the transcript does not.
    """
    meeting_dir, config, log = _broken_meeting_without_a_recording(tmp_path, JICOFO_LOG)
    transcript = meeting_dir / "transcript.txt"
    transcript.write_text(
        "[00:00:00] sezen@example.com: buongiorno a tutti\n\n"
        "[00:01:30] patrick@example.com: buongiorno\n\n"
        # Not a label: mid-line, where somebody said it out loud.
        "write to me at sezen@example.com: I read it\n",
        encoding="utf-8",
    )
    meeting = recover_meeting.inspect(meeting_dir, config)

    assert recover_meeting.recover(
        meeting, config, write=True, attribute=True, jicofo_logs=[log]
    ) == 0

    assert transcript.read_text(encoding="utf-8") == (
        "[00:00:00] sezen: buongiorno a tutti\n\n"
        "[00:01:30] patrick: buongiorno\n\n"
        "write to me at sezen@example.com: I read it\n"
    )


def test_a_second_attribution_replaces_what_the_first_one_wrote(tmp_path: Path) -> None:
    """A meeting attributed twice: the endpoint ids are gone by then.

    The first run wrote what it had -- the addresses from the log -- and the
    second has the names from the recording.  What the first run left in the
    transcript is a label like any other, and it is only replaced in the
    brackets the transcript uses: an address somebody spoke into the meeting
    stays as they said it.
    """
    meeting_dir, config, log = _broken_meeting_without_a_recording(tmp_path, JICOFO_LOG)
    transcript = _labelled_transcript(meeting_dir)
    meeting = recover_meeting.inspect(meeting_dir, config)
    assert recover_meeting.recover(
        meeting, config, write=True, attribute=True, jicofo_logs=[log]
    ) == 0
    assert "[sezen]" in transcript.read_text(encoding="utf-8")
    # Said out loud in the meeting, not a label: this must survive.
    transcript.write_text(
        transcript.read_text(encoding="utf-8")
        + "the address is sezen@example.com, write to me\n",
        encoding="utf-8",
    )

    recording = _jibri_metadata_file(tmp_path, {
        "meeting_url": "https://jitsi.example.com/1ff2f1ff-4bab",
        "participants": [
            {"user": {"name": "Sezen Tosunoğlu", "email": "sezen@example.com"}},
            {"user": {"name": "Patrick Barsanti", "email": "patrick@example.com"}},
        ],
    })
    again = recover_meeting.inspect(meeting_dir, config)
    again.jibri_metadata = recording
    assert recover_meeting.recover(again, config, attribute=True, jicofo_logs=[log]) == 0

    named = transcript.read_text(encoding="utf-8")
    assert "[Sezen Tosunoğlu]" in named
    assert "[sezen@example.com]" not in named
    assert "the address is sezen@example.com, write to me" in named
    # The file the first run kept is not overwritten by the second.
    kept = meeting_dir / "transcript.txt.before-attribution"
    assert "[participant-" in kept.read_text(encoding="utf-8")
    # And what the first run wrote is itself replaced, not renamed into
    # "[Sezen Tosunoğlu]" on top of an address left behind.
    assert named.count("Sezen") == 1


def _jibri_metadata_file(root: Path, document: dict) -> Path:
    """One recording's metadata, somewhere the search does not look."""
    directory = root / "elsewhere" / "jibri-session"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metadata.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    (directory / "1ff2f1ff-4bab_2026-10-13-09-47-21.mp4").write_bytes(b"\x00" * 64)
    return path


def test_the_recording_can_be_named_when_the_search_misses_it(
    tmp_path: Path, capsys
) -> None:
    """--jibri-metadata, as it is actually used: the recording is on the host,
    the search did not find it, and the file beside it has the names."""
    meeting_dir, config, _log = _broken_meeting_without_a_recording(tmp_path, JICOFO_LOG)
    recorded = _jibri_metadata_file(tmp_path, {
        "meeting_url": "https://jitsi.example.com/1ff2f1ff-4bab",
        "participants": [
            {"user": {"name": "Sezen Tosunoğlu", "email": "sezen@example.com"}},
            {"user": {"name": "Patrick Barsanti", "email": "patrick@example.com"}},
        ],
    })
    settings = tmp_path / "config.ini"
    settings.write_text(
        f"[storage]\nrecordings_dir = {config.storage.recordings_dir}\n", encoding="utf-8"
    )

    status = recover_meeting.main([
        str(meeting_dir),
        "--config", str(settings),
        "--jibri-metadata", str(recorded),
        "--write",
    ])

    assert status == 0
    assert "Sezen Tosunoğlu" in capsys.readouterr().out
    assert parse_metadata(meeting_dir)["recipients"] == [
        "patrick@example.com",
        "sezen@example.com",
    ]


def test_a_jibri_metadata_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    """A recording's metadata is read, not believed: it is another program's file."""
    for raw in ("not json at all", "[1, 2]", '"a string"', "null"):
        path = tmp_path / "metadata.json"
        path.write_text(raw, encoding="utf-8")
        assert recover_meeting.people_from_jibri(path) is None, raw
