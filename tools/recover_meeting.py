# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Recover a meeting whose people went missing, without transcribing it again.

The bridge mails a meeting from three things: the audio, the transcript it
made of it, and the metadata that says who was there.  When the metadata
arrived empty -- a companion module that wrote nobody, a control frame that
carried no participants -- the meeting is mailed anyway: attributed by
`participant-<id>`, titled with the placeholder room, and sent to the fallback
address instead of to the people who were in the call.  The audio and the
transcript are fine; only the names are missing.

This finds the names.  Every Jibri recording has a ``metadata.json`` beside it
with the call URL and the participants it could see, names and addresses
included, and that file can be matched to a meeting directory by room and by
time.  The transcript does not have to be made again -- which matters, because
that is the part that costs a GPU and minutes -- so recovery reads what is on
disk, reports what it found, and then, only when asked:

    --write    adopt the participants into the meeting's metadata.json,
               keeping the old file beside it as metadata.json.before-recovery
    --mail     send the summary and transcript to the recovered recipients
    --video    upload the meeting's recording, if [s3] is configured

Examples:

    # What is in this meeting, and who should have received it?
    python3 -m tools.recover_meeting /srv/recordings/53f025af-6544-43da-965a-801494ebad1a

    # The ten most recent, to find the one that was mailed to the wrong place.
    python3 -m tools.recover_meeting --list

    # Recover it: keep the names, and send the mail that should have gone out.
    python3 -m tools.recover_meeting /srv/recordings/53f025af-... --write --mail

Nothing here transcodes, transcribes or summarises.  A meeting whose
transcript is missing as well cannot be recovered by this tool: that is what
``jitsi-audio-bridge --process-dir`` is for, and it costs what it costs.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from jitsi_audio_bridge import __version__
from jitsi_audio_bridge.ai_client import detect_language
from jitsi_audio_bridge.audio import (
    CORRECTED_TRANSCRIPT_NAME,
    METADATA_FILENAME,
    TIMELINE_FILENAME,
    parse_metadata,
)
from jitsi_audio_bridge.config import Config, ConfigError, load_config
from jitsi_audio_bridge.daemon import session_started_at, write_metadata
from jitsi_audio_bridge.mailer import send_meeting_email
from jitsi_audio_bridge.s3_upload import find_meeting_metadata, upload_meeting_video
from jitsi_audio_bridge.timeline import SessionTimeline

logger = logging.getLogger("tools.recover_meeting")

#: Written beside the metadata it replaces, so that a recovery can be undone
#: by hand and so that what the bridge had is never simply gone.
BACKUP_NAME = METADATA_FILENAME + ".before-recovery"

SUMMARY_NAME = "summary.md"
TRANSCRIPT_NAME = "transcript.txt"


@dataclass
class Meeting:
    """One meeting directory, and what can be said about it."""

    path: Path
    room_name: str
    participants: list[str] = field(default_factory=list)
    recipients: list[str] = field(default_factory=list)
    transcript: Path | None = None
    summary: Path | None = None
    timeline: SessionTimeline | None = None
    #: Jibri's metadata for the recording that belongs to this meeting.
    jibri_metadata: Path | None = None
    recording: Path | None = None

    @property
    def has_people(self) -> bool:
        return bool(self.recipients or self.participants)

    @property
    def started_at(self) -> str | None:
        return session_started_at(self.path, self.timeline)

    @property
    def recoverable(self) -> bool:
        """Whether there is a mail to send and somebody to send it to."""
        return bool(self.transcript and self.summary)


def inspect(meeting_dir: Path, config: Config) -> Meeting:
    """Read a meeting directory and look for the people Jibri knew about."""
    meeting = Meeting(path=meeting_dir, room_name="")
    info = parse_metadata(meeting_dir)
    meeting.room_name = info["room_name"]
    meeting.participants = list(info["participants"])
    meeting.recipients = list(info["recipients"])

    corrected = meeting_dir / CORRECTED_TRANSCRIPT_NAME
    plain = meeting_dir / TRANSCRIPT_NAME
    for candidate in (corrected, plain):
        if candidate.is_file():
            meeting.transcript = candidate
            break
    summary = meeting_dir / SUMMARY_NAME
    meeting.summary = summary if summary.is_file() else None
    meeting.timeline = SessionTimeline.load(meeting_dir / TIMELINE_FILENAME)

    # The recording is looked for whatever the metadata says, because it is
    # the video of the meeting as much as it is the source of the names --
    # and it is the room and the time that find it, both of which are here.
    if meeting.room_name:
        jibri = find_meeting_metadata(meeting_dir, config, meeting.room_name)
        if jibri is not None:
            meeting.jibri_metadata = jibri
            meeting.recording = next(
                (path for path in jibri.parent.iterdir()
                 if path.is_file() and path.name != jibri.name),
                None,
            )
    return meeting


def people_from_jibri(jibri_metadata: Path, meeting_dir: Path) -> dict | None:
    """Jibri's metadata as this daemon would read it, or ``None``.

    Parsed the way the daemon parses its own, so that what a recovery reports
    is what the mail will use -- and written nowhere: the caller decides.
    """
    try:
        raw = jibri_metadata.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("cannot read %s: %s", jibri_metadata, exc)
        return None

    # parse_metadata reads a file, so this reads the same content through the
    # same code path by way of a temporary directory rather than a second
    # implementation of the participant parsing.
    scratch = meeting_dir / ".recovery-scratch"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / METADATA_FILENAME).write_text(raw, encoding="utf-8")
        return parse_metadata(scratch)
    except OSError as exc:
        logger.error("cannot read what Jibri wrote: %s", exc)
        return None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def report(meeting: Meeting) -> None:
    """Say what the meeting holds, in the order somebody would ask."""
    print(f"\n{meeting.path}")
    print(f"  room        {meeting.room_name}")
    print(f"  started     {meeting.started_at or '(unknown)'}")
    print(f"  transcript  {meeting.transcript.name if meeting.transcript else '(missing)'}")
    print(f"  summary     {meeting.summary.name if meeting.summary else '(missing)'}")
    if meeting.participants:
        print(f"  people      {', '.join(meeting.participants)}")
    else:
        print("  people      (none in the metadata)")
    print(f"  recipients  {', '.join(meeting.recipients) or '(none; the fallback address)'}")

    if meeting.recording is not None:
        size = meeting.recording.stat().st_size / (1 << 20)
        print(f"  recording   {meeting.recording.name} ({size:.1f} MiB)")
    elif meeting.jibri_metadata is not None:
        print(f"  recording   {meeting.jibri_metadata.parent.name} (no media file)")
    else:
        print("  recording   (none found for this room and time)")


def recover(
    meeting: Meeting,
    config: Config,
    *,
    write: bool = False,
    mail: bool = False,
    video: bool = False,
    language: str | None = None,
) -> int:
    """Do the recovery.  Returns a process exit status."""
    if not meeting.recoverable:
        logger.error(
            "%s has no transcript or no summary; this tool recovers the people of a "
            "meeting that was processed, it does not process one.  Use "
            "`jitsi-audio-bridge --process-dir %s`.",
            meeting.path,
            meeting.path,
        )
        return 1

    info = parse_metadata(meeting.path)
    participants = meeting.participants
    recipients = meeting.recipients

    if not meeting.has_people and meeting.jibri_metadata is not None:
        found = people_from_jibri(meeting.jibri_metadata, meeting.path)
        if found is None:
            return 1
        participants, recipients = found["participants"], found["recipients"]
        if not participants and not recipients:
            logger.warning(
                "Jibri's metadata for this meeting names nobody either; "
                "there is nothing to recover from it"
            )
        else:
            print(f"\nrecovered from {meeting.jibri_metadata.parent.name}:")
            print(f"  people      {', '.join(participants)}")
            print(f"  recipients  {', '.join(recipients) or '(nobody with an address)'}")
            if write:
                backup = meeting.path / BACKUP_NAME
                current = meeting.path / METADATA_FILENAME
                if current.is_file() and not backup.exists():
                    shutil.copy2(current, backup)
                    print(f"  kept        {backup.name}")
                written = write_metadata(
                    meeting.path, meeting.jibri_metadata.read_text(encoding="utf-8")
                )
                if not written:
                    return 1
                print(f"  wrote       {METADATA_FILENAME}")
    elif not meeting.has_people:
        logger.warning(
            "the meeting names nobody, and no recording of it was found to take "
            "them from: the room may have been recorded on another host, or the "
            "recording is by now outside the window this searches"
        )

    if not recipients:
        logger.warning("no recipients: the mail would go to the fallback address")

    if video and config.s3.enabled:
        print("\nrecording:")
        upload_meeting_video(meeting.path, config, wait=False)
    elif video:
        logger.warning("[s3] endpoint and bucket are unset; nothing to upload")

    if mail:
        if language is None:
            text = meeting.transcript.read_text(encoding="utf-8") if meeting.transcript else ""
            language = detect_language(text, config.ollama)
        print(f"\nsending '{meeting.summary.name}' and '{meeting.transcript.name}'"
              f" to {', '.join(recipients) or 'the fallback address'} ({language})")
        sent = send_meeting_email(
            recipients,
            info["room_name"],
            meeting.summary.read_text(encoding="utf-8"),
            meeting.transcript,
            meeting.summary,
            config.smtp,
            started_at=meeting.started_at,
            language=language,
        )
        print("  sent" if sent else "  not sent; see the log")
        return 0 if sent else 1

    if not write and (participants or recipients):
        print("\nnothing was changed: pass --write to keep the names, --mail to send")
    return 0


def recent_meetings(config: Config, limit: int) -> list[Path]:
    """The newest meeting directories, newest first."""
    root = config.storage.recordings_dir
    try:
        candidates = [path for path in root.iterdir() if path.is_dir()]
    except OSError as exc:
        logger.error("cannot read %s: %s", root, exc)
        return []
    # Hidden directories are the bridge's own (.session-metadata); a meeting
    # directory is named after its session id.
    meetings = [path for path in candidates if not path.name.startswith(".")]
    meetings.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return meetings[:limit]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m tools.recover_meeting",
        description=(
            "Recover a meeting's people from Jibri's metadata and re-send its mail, "
            "without transcribing it again."
        ),
    )
    parser.add_argument(
        "meeting",
        nargs="?",
        metavar="DIR",
        help="the meeting directory to recover",
    )
    parser.add_argument(
        "--list",
        nargs="?",
        const=10,
        type=int,
        metavar="N",
        help="list the N most recent meetings and exit (default 10)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"keep the recovered participants in {METADATA_FILENAME}, backing up the old file",
    )
    parser.add_argument(
        "--mail",
        action="store_true",
        help="send the summary and transcript to the recovered recipients",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="upload the meeting's recording, when [s3] is configured",
    )
    parser.add_argument(
        "--language",
        metavar="NAME",
        help="the language the summary is written in, skipping the detection call",
    )
    parser.add_argument("--config", metavar="PATH", help="configuration file to read")
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
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

    if args.list is not None:
        for path in recent_meetings(config, args.list):
            info = parse_metadata(path)
            timeline = SessionTimeline.load(path / TIMELINE_FILENAME)
            when = session_started_at(path, timeline) or "?"
            print(
                f"{path.name:40}  {when:16}  {info['room_name'][:36]:36}  "
                f"{len(info['recipients'])} recipient(s)"
            )
        return 0

    if args.meeting is None:
        build_parser().print_usage(sys.stderr)
        logger.error("a meeting directory is needed, or --list to find one")
        return 2

    meeting_dir = Path(args.meeting)
    if not meeting_dir.is_dir():
        logger.error("not a directory: %s", meeting_dir)
        return 2

    meeting = inspect(meeting_dir, config)
    report(meeting)
    return recover(
        meeting,
        config,
        write=args.write,
        mail=args.mail,
        video=args.video,
        language=args.language,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
