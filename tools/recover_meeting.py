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

    # The recording's metadata was not found by the search, but it is here:
    # take the people from it and name the labels in the transcript.
    python3 -m tools.recover_meeting /srv/recordings/53f025af-... \
        --jibri-metadata /srv/jibri-recordings/908c74d7/metadata.json \
        --write --attribute

Nothing here transcodes, transcribes or summarises.  A meeting whose
transcript is missing as well cannot be recovered by this tool: that is what
``jitsi-audio-bridge --process-dir`` is for, and it costs what it costs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
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
    name_from_address,
    parse_metadata,
    parse_metadata_document,
)
from jitsi_audio_bridge.config import Config, ConfigError, load_config
from jitsi_audio_bridge.daemon import (
    match_directory_owner,
    session_started_at,
    write_metadata,
)
from jitsi_audio_bridge.mailer import send_meeting_email
from jitsi_audio_bridge.s3_upload import VideoLink, find_meeting_metadata, prepare_video
from jitsi_audio_bridge.timeline import SessionTimeline

logger = logging.getLogger("tools.recover_meeting")

#: Written beside the metadata it replaces, so that a recovery can be undone
#: by hand and so that what the bridge had is never simply gone.
BACKUP_NAME = METADATA_FILENAME + ".before-recovery"

SUMMARY_NAME = "summary.md"
TRANSCRIPT_NAME = "transcript.txt"

#: Jicofo logs every participant arriving, with the id the bridge's audio is
#: tagged with and the identity the deployment gave them:
#:
#:     [room=... meeting_id=53f025af-...] JitsiMeetConferenceImpl.onMemberJoined:
#:         Member joined:76e96806 stats-id=sezen@amarulasolutions.com ...
#:
#: The first is the endpoint id, which is what ``participant-76e96806-a0.wav``
#: is named after; the second is an address where the deployment has one, and
#: whatever the token called them otherwise.  This is the only place the two
#: are written down together -- the Prosody module is the other, and it lives
#: in memory -- so it is what a meeting whose participants were lost has left.
_JICOFO_JOIN = re.compile(
    r"Member joined\s*:(?P<endpoint>[0-9a-zA-Z_-]+)\s+stats-id=(?P<who>\S+)"
)

#: Where Jicofo writes.  What is read is this file and every rotation of it:
#: the line that says who arrived is the one this needs, and by the time a
#: meeting is worth recovering that line can be a rotation or two back --
#: Jicofo rotates when it restarts as well as by the day, and a deployment
#: that has been upgraded since the meeting has the meeting in neither of the
#: two files somebody would look in first.
JICOFO_LOG = Path("/var/log/jitsi/jicofo.log")


def default_jicofo_logs() -> list[Path]:
    """Every Jicofo log on this host, oldest first.

    Oldest first so that the last line read for a participant is the last one
    written about them: somebody who disconnected and came back has two, and
    the identity they came back under is the one that counts.
    """
    rotated = sorted(
        JICOFO_LOG.parent.glob(f"{JICOFO_LOG.name}*"), key=_rotation_order, reverse=True
    )
    return rotated or [JICOFO_LOG]


def _rotation_order(path: Path) -> int:
    """0 for the log in use, 1 for the first rotation, and so on upwards."""
    suffix = path.name[len(JICOFO_LOG.name):].lstrip(".").split(".")[0]
    return int(suffix) if suffix.isdigit() else 0


def _log_lines(path: Path) -> list[str]:
    """A log file's lines, rotated-and-compressed ones included."""
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                return handle.readlines()
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        logger.warning("cannot read %s: %s", path, exc)
        return []


def endpoints_from_jicofo(log_paths: list[Path], meeting_id: str) -> dict[str, str]:
    """Endpoint id -> the identity Jicofo logged for it, for one meeting.

    Scoped by the meeting id, which is on the same line as the join, so a log
    holding a week of meetings yields this one's participants and nobody
    else's.  Later lines win, which is what a participant who reconnected
    wants: the last identity under which they were in the call.
    """
    names: dict[str, str] = {}
    marker = f"meeting_id={meeting_id}"
    for path in log_paths:
        for line in _log_lines(path):
            if marker not in line:
                continue
            found = _JICOFO_JOIN.search(line)
            if found:
                names[found.group("endpoint")] = found.group("who")
    return names


def label_samples(meeting_dir: Path, limit: int = 3) -> list[tuple[str, str]]:
    """What the meeting's files call their speakers, a few of them.

    For the case where none of the labels matched: what the tool knows a
    person as is not what the file calls them, and the difference between the
    two is the whole of the problem.  In the order they appear.
    """
    samples: list[tuple[str, str]] = []
    for name in (TRANSCRIPT_NAME, CORRECTED_TRANSCRIPT_NAME, SUMMARY_NAME):
        path = meeting_dir / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        found: list[str] = []
        # Both shapes a label comes in: what follows the offset on a line of
        # the interleaved transcript, and the bracket the summary uses.
        for line in text.splitlines():
            match = re.match(r"(?:\[[0-9:.]+\] )?([^:\n]{1,60}): ", line)
            if match:
                found.append(match.group(1))
            found.extend(m.group(0) for m in re.finditer(r"\[[^\]\n]{1,60}\]", line))
        unique = [label for label in dict.fromkeys(found) if label.strip("[] ")]
        if unique:
            samples.append((name, " ".join(unique[:limit])))
    return samples


def previous_labels(mapping: dict[str, str], names: dict[str, str]) -> dict[str, str]:
    """Labels an earlier recovery already substituted, back to the endpoint.

    A meeting can be attributed twice -- the first time with what was known
    then, the second with the names a recording turned out to hold -- and by
    then the endpoint ids are gone from the transcript: what is in their place
    is whatever the first run wrote.  So the words that run could have written
    are labels too, and the second run has to recognise them.

    Only for endpoints whose label this run would change: a label that is
    already the name stays as it is.
    """
    replaced: dict[str, str] = {}
    for endpoint, who in mapping.items():
        for previous in (who, name_from_address(who)):
            if previous and previous != names[endpoint]:
                replaced.setdefault(previous, endpoint)
    return replaced


def attribute_files(
    meeting_dir: Path,
    names: dict[str, str],
    *,
    write: bool,
    replaced: dict[str, str] | None = None,
) -> list[tuple[str, str, int]]:
    """Replace the labels the daemon wrote with the names of the people.

    The transcript is rendered from the recordings' names, so its labels are
    ``[participant-76e96806-a0]`` and the summary, written by the model from
    that transcript, carries the same words.  Only those labels change: the
    spoken text is untouched, which is the difference between attributing a
    meeting and rewriting one.

    Returns ``(file, label, replacements)`` for everything it changed, or would
    have.
    """
    report: list[tuple[str, str, int]] = []
    for name in (TRANSCRIPT_NAME, CORRECTED_TRANSCRIPT_NAME, SUMMARY_NAME):
        path = meeting_dir / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("cannot read %s: %s", path, exc)
            continue

        changed = text
        counts: list[tuple[str, str, int]] = []
        for endpoint, who in sorted(names.items()):
            # The label is the recording's name, which is the tag the audio
            # arrived under: the endpoint id and then the source, as in
            # ``participant-76e96806-a0``.  Matching the whole of it, and not
            # the id alone, is what keeps "-a0" from surviving the rename --
            # and refusing to match a bare id keeps one id from matching
            # another that begins with it.
            label = f"participant-{endpoint}"
            pattern = re.compile(rf"{re.escape(label)}(?:-[0-9a-zA-Z_]+)?")
            changed, count = pattern.subn(who, changed)
            counts.append((name, label, count))
        for previous, endpoint in sorted((replaced or {}).items()):
            # Only where a label can be, and nowhere else: a meeting is about
            # work, and somebody dictating their address into it must not have
            # that address rewritten.  There are two shapes, and a meeting
            # written by this daemon has both: the summary and the corrected
            # transcript put the speaker in brackets -- "[michael@example.com]
            # detailed the need" -- while the interleaved transcript puts the
            # offset in them and the speaker after, "\[00:01:30] michael@example.com: text".
            # Matching only the first is what left a repaired summary beside
            # an unrepaired transcript.
            bracket = re.compile(rf"\[{re.escape(previous)}\]")
            changed, count = bracket.subn(f"[{names[endpoint]}]", changed)
            counts.append((name, previous, count))
            speaker = re.compile(
                rf"(?m)^(\[[0-9:.]+\] )?{re.escape(previous)}(?=: )"
            )
            changed, speaker_count = speaker.subn(rf"\g<1>{names[endpoint]}", changed)
            counts.append((name, f"{previous} (as a speaker)", speaker_count))
        if changed == text:
            continue
        if write:
            backup = path.with_suffix(path.suffix + ".before-attribution")
            if not backup.exists():
                shutil.copy2(path, backup)
                print(f"  kept        {backup.name}")
            path.write_text(changed, encoding="utf-8")
            print(f"  named       {name}")
        report.extend(counts)
    return report


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


def _as_metadata(raw: str) -> dict | None:
    """Read a metadata document the way the daemon reads its own.

    The document is looked at before it is written anywhere, so that what a
    recovery would put in the meeting directory can be judged first -- and
    refused, if it names nobody.
    """
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("the metadata is not readable: %s", exc)
        return None
    if not isinstance(document, dict):
        logger.error("the metadata is not a JSON object: %.60r", raw)
        return None
    return parse_metadata_document(document)


def people_from_jibri(jibri_metadata: Path) -> dict | None:
    """Jibri's metadata as this daemon would read it, or ``None``.

    Parsed the way the daemon parses its own, so that what a recovery reports
    is what the mail will use -- and written nowhere: the caller decides.
    """
    try:
        raw = jibri_metadata.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("cannot read %s: %s", jibri_metadata, exc)
        return None

    return _as_metadata(raw)


def names_for_labels(meeting: Meeting, mapping: dict[str, str]) -> tuple[dict[str, str], set[str]]:
    """What each endpoint is called, and which of them have no name at all.

    Every source that holds a name is asked: the meeting's own metadata, and
    the recording's, which is where the names are when the deployment
    authenticates its users -- Jibri writes what Mattermost told it.  What the
    meeting already says wins, because a name recovered earlier was recovered
    from a source that knew more.

    A name that is only the address shortened is not one of those: it is what
    a reader is given when nothing knows better, and anything that does know
    better is preferred to it.  The second half of the answer is the point of
    the function -- somebody no source names is somebody whose label is their
    address, and that is worth saying before the transcript is rewritten
    rather than after.
    """
    known = dict(parse_metadata(meeting.path)["id_to_name"])
    recorded: dict[str, str] = {}
    if meeting.jibri_metadata is not None:
        found = people_from_jibri(meeting.jibri_metadata) or {}
        recorded = found.get("id_to_name", {})

    names: dict[str, str] = {}
    invented: set[str] = set()
    for endpoint, who in mapping.items():
        shortened = name_from_address(who)
        better = [
            candidate
            for candidate in (
                known.get(who),
                known.get(endpoint),
                recorded.get(who),
                recorded.get(endpoint),
            )
            if candidate and candidate not in (who, shortened)
        ]
        if better:
            names[endpoint] = better[0]
        else:
            # Nothing anywhere calls this person anything: what the address
            # says, which is what the deployment itself calls them.
            names[endpoint] = shortened
            invented.add(endpoint)
    return names, invented


def metadata_from_jicofo(mapping: dict[str, str], room_name: str, meeting_id: str) -> str:
    """The metadata document the people Jicofo saw amount to.

    Jicofo's log is the only place the two id spaces are written down
    together: the endpoint id the audio is tagged with -- which is what the
    transcript's labels are built from -- and the identity the deployment gave
    that endpoint.  Where the deployment authenticates users, that identity is
    the address the mail has to reach, and it is what is left when the
    companion module wrote nobody and no recording can be found to take names
    from.

    The join line carries no display name, so the identity stands in for one:
    this invents nothing Jitsi did not say.  Written in the shape the module
    writes, which is the shape the daemon reads, with the endpoint as the
    participant id so that an attribution later has the ids it matches labels
    against.
    """
    document = {
        "room_name": room_name,
        "meeting_id": meeting_id,
        "source": "jicofo_attribution",
        "participants": [
            _jicofo_participant(endpoint, who)
            for endpoint, who in sorted(mapping.items())
        ],
    }
    return json.dumps(document)


def _jicofo_participant(endpoint: str, identity: str) -> dict[str, str]:
    """One participant, in the fields the daemon actually reads.

    A deployment that authenticates users gives Jicofo an address, and that
    goes in ``email``, because it is what the mail has to reach.  One that
    does not -- or a user with no address of their own -- gives it a name, and
    a name is only kept from a participant that has one beside an id, so the
    endpoint id goes with it in ``name``.

    The distinction is the whole difference between a metadata document that
    names somebody and one that names nobody.  A name written into ``email``
    reads as a participant with neither an address nor a name: the daemon
    drops it, and the meeting is mailed as though nobody had been in it --
    which is the failure this fallback exists to repair.
    """
    if "@" in identity:
        return {"id": endpoint, "email": identity}
    return {"id": endpoint, "name": identity}


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
    attribute: bool = False,
    jicofo_logs: list[Path] | None = None,
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

    # Who each recording belongs to, from the only place the two id spaces are
    # written down together.  Jicofo logs an address where the deployment has
    # one.  Looked up once, because it is also the last source of people when
    # neither the metadata nor a recording names any.
    logs = jicofo_logs or default_jicofo_logs()
    mapping = endpoints_from_jicofo(logs, meeting.path.name)

    if not meeting.has_people and meeting.jibri_metadata is not None:
        found = people_from_jibri(meeting.jibri_metadata)
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
                    match_directory_owner(backup)
                    print(f"  kept        {backup.name}")
                written = write_metadata(
                    meeting.path, meeting.jibri_metadata.read_text(encoding="utf-8")
                )
                if not written:
                    return 1
                print(f"  wrote       {METADATA_FILENAME}")
    elif not meeting.has_people and mapping:
        # No recording to take names from, but Jicofo saw who was in the call.
        # An address and an endpoint id are enough to address the mail and to
        # match the labels; a display name is the one thing this cannot give.
        document = metadata_from_jicofo(mapping, info["room_name"], meeting.path.name)
        found = _as_metadata(document)
        if found is None:
            return 1
        participants, recipients = found["participants"], found["recipients"]
        if not participants and not recipients:
            # What was built reads as nobody -- which is what this whole
            # recovery exists to repair, and must not be the thing that
            # writes it.  The meeting is left exactly as it was.
            logger.warning(
                "Jicofo logged %d participant(s) for %s but the document they amount "
                "to names nobody; leaving %s as it is",
                len(mapping),
                meeting.path.name,
                METADATA_FILENAME,
            )
        else:
            print("\nrecovered from the Jicofo log:")
            print(f"  people      {', '.join(participants) or '(nobody with an address)'}")
            print(f"  recipients  {', '.join(recipients) or '(nobody with an address)'}")
            if write:
                backup = meeting.path / BACKUP_NAME
                current = meeting.path / METADATA_FILENAME
                if current.is_file() and not backup.exists():
                    shutil.copy2(current, backup)
                    match_directory_owner(backup)
                    print(f"  kept        {backup.name}")
                if not write_metadata(meeting.path, document):
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

    if mapping:
        names, unnamed = names_for_labels(meeting, mapping)
        print("\nendpoint id   who Jicofo saw       (this is what the labels are named after)")
        for endpoint, who in sorted(mapping.items()):
            note = ""
            if names[endpoint] == who:
                note = "  (no name anywhere: this is the address)"
            elif endpoint in unnamed:
                note = "  (no name anywhere: the address, shortened)"
            print(f"  {endpoint:12}  {names[endpoint]}{note}")
        if unnamed and meeting.jibri_metadata is None:
            print(
                "\n  No recording was found for this meeting, so the names Jibri\n"
                "  writes beside one were not available.  If it is on this host,\n"
                "  point at it with --jibri-metadata and the labels will be names."
            )
        if attribute:
            print("\nattributing:")
            changed = attribute_files(
                meeting.path,
                names,
                write=True,
                replaced=previous_labels(mapping, names),
            )
            for path_name, label, count in changed:
                if count:
                    print(f"  {path_name:24}  {label} -> {count} label(s)")
            if not any(count for _, _, count in changed):
                # Naming what was looked for is no help when the file's labels
                # look like something else entirely: a corrected transcript is
                # a model's rendering of the first one, and it keeps them only
                # roughly.  So show what is actually in there.
                print("  nothing to rename: no label matches those ids")
                for name, sample in label_samples(meeting.path):
                    print(f"  {name:24}  its labels look like: {sample}")
        else:
            print("\n  pass --attribute to name the labels in the transcript and summary")
    elif attribute:
        logger.warning(
            "no join lines for meeting %s in %s; nothing to attribute",
            meeting.path.name,
            ", ".join(str(path) for path in logs) or "the Jicofo logs",
        )
        logger.warning(
            "if the meeting is in a log that has been rotated away or moved, "
            "name it with --jicofo-log"
        )

    link: VideoLink | None = None
    if video and config.s3.enabled:
        print("\nrecording:")
        # The same upload the daemon does before a mail, for the same reason:
        # a link is only worth putting in the mail if it works when the mail
        # is read.  The recording is already finished here, so the wait is
        # only there for the case where it is not.
        link = prepare_video(meeting.path, config, wait_seconds=config.s3.link_wait_seconds)
        print(f"  {'uploaded     ' + link.url if link else 'not uploaded; see the log'}")
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
            video_url=link.url if link else None,
            video_until=link.expires_at if link else None,
        )
        print("  sent" if sent else "  not sent; see the log")
        return 0 if sent else 1

    if not write and not attribute and (participants or recipients):
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
        "--attribute",
        action="store_true",
        help=(
            "name the speaker labels in the transcript and summary, using the endpoint "
            "ids Jicofo logged for this meeting"
        ),
    )
    parser.add_argument(
        "--jibri-metadata",
        metavar="PATH",
        help=(
            "a recording's metadata file to take the people and the display names "
            "from, when this meeting's recording is not found by the search"
        ),
    )
    parser.add_argument(
        "--jicofo-log",
        action="append",
        metavar="PATH",
        help="a Jicofo log to read the endpoint ids from; repeatable, and gzip is fine",
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
    if args.jibri_metadata is not None:
        source = Path(args.jibri_metadata)
        if not source.is_file():
            logger.error("not a file: %s", source)
            return 2
        # Named rather than searched for: a recording the search did not find
        # still holds the names -- the file is beside it, and both are here.
        meeting.jibri_metadata = source
        meeting.recording = next(
            (
                path
                for path in source.parent.iterdir()
                if path.is_file() and path.name != source.name
            ),
            None,
        )
    report(meeting)
    return recover(
        meeting,
        config,
        write=args.write,
        mail=args.mail,
        video=args.video,
        attribute=args.attribute,
        jicofo_logs=[Path(path) for path in args.jicofo_log] if args.jicofo_log else None,
        language=args.language,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
