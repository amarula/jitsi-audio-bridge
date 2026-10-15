# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""What Jicofo's log knows about a meeting's participants.

The bridge hears one thing about a participant: the endpoint id its audio
arrives under, which is what the recordings are named after.  What the
deployment calls that person -- and, where it authenticates its users, the
address it knows them by -- is in the room, and the Prosody module is what
carries it across.  That module's record is memory, and memory does not
survive the room being recreated, so a meeting can arrive with nobody in it.

Jicofo's log is the other place the two are written down together, and it is
written to disk:

    [room=... meeting_id=53f025af-...] JitsiMeetConferenceImpl.onMemberJoined:
        Member joined:76e96806 stats-id=sezen@amarulasolutions.com region=null

The first is the endpoint id the audio is tagged with; the second is the
identity the deployment authenticated, which is an address where it has one
and whatever the token called them otherwise.  There is no display name in
it -- that is what the Prosody module reads out of the presence, and nothing
else has it -- so a name taken from here is the address shortened, which is
what the deployment itself calls that person.
"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where Jicofo writes.  What is read is this file and every rotation of it:
#: by the time a meeting is worth recovering, the line that says who arrived
#: can be a rotation or two back -- Jicofo rotates when it restarts as well as
#: by the day, and a deployment upgraded since the meeting has it in neither
#: of the two files somebody would look in first.
JICOFO_LOG = Path("/var/log/jitsi/jicofo.log")

_JICOFO_JOIN = re.compile(
    r"Member joined\s*:(?P<endpoint>[0-9a-zA-Z_-]+)\s+stats-id=(?P<who>\S+)"
)


def default_logs() -> list[Path]:
    """Every Jicofo log on this host, oldest first.

    Oldest first so that the last line read for a participant is the last one
    written about them: somebody who disconnected and came back has two, and
    the identity they came back under is the one that counts.
    """
    found = JICOFO_LOG.parent.glob(f"{JICOFO_LOG.name}*")
    return sorted(found, key=_rotation_order, reverse=True) or [JICOFO_LOG]


def _rotation_order(path: Path) -> int:
    """0 for the log in use, 1 for the first rotation, and so on upwards."""
    suffix = path.name[len(JICOFO_LOG.name):].lstrip(".").split(".")[0]
    return int(suffix) if suffix.isdigit() else 0


def log_lines(path: Path) -> list[str]:
    """A log file's lines, rotated-and-compressed ones included."""
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                return handle.readlines()
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        # Readable-by-nobody is the usual case: the daemon runs as its own
        # user, and these logs belong to root.  Said once per file per
        # meeting, and it is the whole reason a name found by the recovery
        # tool is not found by the daemon.
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
        for line in log_lines(path):
            if marker not in line:
                continue
            found = _JICOFO_JOIN.search(line)
            if found:
                names[found.group("endpoint")] = found.group("who")
    return names
