# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Session timing: who spoke when, captured while the call is happening.

A transcript built from per-participant files is a sequence of monologues,
because a WAV holds samples and nothing else.  The JVB tells us plenty while
it sends: every ``media`` event carries an arrival moment and, when the
exporter sets them, a voice-activity flag and an audio level.  This module
turns that stream into speaking turns on a shared, session-relative clock —
which is the only thing that has to be captured live, since a packet that
arrived without a timestamp cannot be given one afterwards.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Silence between two runs of the same speaker that still counts as one turn.
#: Speech pauses inside a sentence are well under this; a change of speaker
#: usually is not, and the exporter's own flag settles the rest.
DEFAULT_MERGE_GAP_SECONDS = 0.7

#: A run longer than this is closed and a new turn opened, so one participant
#: talking for ten minutes does not become a single request.
DEFAULT_MAX_TURN_SECONDS = 30.0

#: Runs shorter than this are a click or comfort noise, not speech.
DEFAULT_MIN_TURN_SECONDS = 0.3

#: A packet this loud is speech.  16-bit PCM full scale is 1.0; conversational
#: speech sits around 0.1 and line silence below 0.005, so the threshold has
#: an order of magnitude of room on either side.
DEFAULT_SPEECH_LEVEL = 0.02


def format_offset(seconds: float) -> str:
    """Render a session offset as ``HH:MM:SS``, for transcript prefixes."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


@dataclass
class Turn:
    """One participant speaking without interruption."""

    participant: str
    #: Seconds since the session started — the shared clock.
    start: float
    end: float
    #: Where this turn begins inside that participant's own WAV.
    offset: float
    samples: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict[str, Any]:
        return {
            "participant": self.participant,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "offset": round(self.offset, 3),
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Turn | None:
        """Build a turn from decoded JSON, or ``None`` if it is not one."""
        if not isinstance(raw, dict):
            return None
        participant = raw.get("participant")
        if not isinstance(participant, str) or not participant:
            return None
        try:
            start = float(raw["start"])
            end = float(raw["end"])
            offset = float(raw["offset"])
            samples = int(raw["samples"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (start, end, offset)):
            return None
        return cls(participant, start, max(start, end), max(0.0, offset), max(0, samples))


class TurnTracker:
    """Turn per-packet observations into speaking turns.

    Fed one packet at a time, per participant, with the moment it arrived on
    the session clock, where it landed in that participant's recording, how
    long it is, and how loud it was.  Turns extend while packets keep coming
    within *merge_gap* of the last *speech* — silence packets do not hold a
    turn open, or a participant's comfort noise would make one turn out of the
    whole meeting — and are closed by a longer pause, by *max_turn*, or by the
    end of the session.
    """

    def __init__(
        self,
        *,
        merge_gap: float = DEFAULT_MERGE_GAP_SECONDS,
        max_turn: float = DEFAULT_MAX_TURN_SECONDS,
        min_turn: float = DEFAULT_MIN_TURN_SECONDS,
        speech_level: float = DEFAULT_SPEECH_LEVEL,
    ) -> None:
        self.merge_gap = merge_gap
        self.max_turn = max_turn
        self.min_turn = min_turn
        self.speech_level = speech_level
        self._open: dict[str, Turn] = {}
        self._last_voice_end: dict[str, float] = {}
        self._last_end: dict[str, float] = {}
        self._turns: list[Turn] = []

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def is_speech(self, level: float | None, vad: bool | None) -> bool:
        """Whether a packet counts as speech.

        The exporter's own flag wins where it set one; otherwise the level
        decides, and a packet with neither (an old sender, or the legacy
        framing) counts as speech: a turn that is too long is a smaller error
        than a silent meeting.
        """
        if vad is not None:
            return vad
        if level is None:
            return True
        return level >= self.speech_level

    def add(
        self,
        participant: str,
        *,
        session_offset: float,
        file_offset: float,
        duration: float,
        level: float | None = None,
        vad: bool | None = None,
    ) -> None:
        """Record one packet.  Never raises: timing is never worth a meeting.

        The session clock here is the arrival clock: it is what makes turns
        comparable across participants.  A stream that arrives faster than it
        plays — a sender replaying a recording, or a bridge that buffered —
        would otherwise stack every packet on one instant and collapse the
        meeting into a handful of milliseconds, so a stream's own packets are
        laid end to end when they overlap.  For a live stream, whose packets
        arrive 20 ms of audio apart, this changes nothing.
        """
        if not participant or duration <= 0:
            return
        previous_end = self._last_end.get(participant)
        if previous_end is not None and session_offset < previous_end:
            session_offset = previous_end
        self._last_end[participant] = session_offset + duration
        end = session_offset + duration
        if self.is_speech(level, vad):
            self._extend(participant, session_offset, end, file_offset, duration)
            self._last_voice_end[participant] = end
            return

        last_voice = self._last_voice_end.get(participant)
        if last_voice is not None and session_offset - last_voice > self.merge_gap:
            self._close(participant)

    def _extend(
        self, participant: str, start: float, end: float, file_offset: float, duration: float
    ) -> None:
        turn = self._open.get(participant)
        last_voice = self._last_voice_end.get(participant)
        if turn is None or (last_voice is not None and start - last_voice > self.merge_gap):
            self._close(participant)
            turn = Turn(
                participant=participant,
                start=start,
                end=end,
                offset=file_offset,
                samples=int(duration * 1000) or 1,
            )
            self._open[participant] = turn
        else:
            turn.end = end
            turn.samples += int(duration * 1000) or 1

        if turn.duration >= self.max_turn:
            # Closed here, not at the next packet: the next one reopens a turn
            # a frame later, which is exactly the split we want.
            self._close(participant)

    def _close(self, participant: str) -> None:
        turn = self._open.pop(participant, None)
        if turn is None:
            return
        if turn.duration >= self.min_turn:
            self._turns.append(turn)

    def finish(self) -> list[Turn]:
        """Close every open turn and return them all, in speaking order."""
        for participant in list(self._open):
            self._close(participant)
        self._turns.sort(key=lambda turn: (turn.start, turn.participant))
        return self._turns


def merge_turns(turns: list[Turn], limit: int) -> list[Turn]:
    """Reduce one participant's turns to *limit* by joining the closest pairs.

    A long meeting can hold more turns than are worth transcribing one by one,
    and the alternative — falling back to a whole-file monologue — loses the
    ordering the timeline exists to provide.  Merging keeps it: a merged turn
    starts where its first part did, and the closest gaps go first, so what is
    lost is resolution, not sequence.
    """
    if limit < 1:
        limit = 1
    while len(turns) > limit:
        closest_gap, index = min(
            (turns[position + 1].start - turns[position].end, position)
            for position in range(len(turns) - 1)
        )
        del closest_gap
        first, second = turns[index], turns[index + 1]
        turns[index] = Turn(
            participant=first.participant,
            start=first.start,
            end=max(first.end, second.end),
            offset=first.offset,
            samples=first.samples + second.samples,
        )
        del turns[index + 1]
    return turns


@dataclass
class SessionTimeline:
    """What was captured for one session: the shared clock, and who spoke."""

    #: Wall clock of the connection, so a session offset can be turned into an
    #: absolute time by anyone who needs one.
    started_at: str = ""
    duration: float = 0.0
    #: Seconds of audio recorded per participant, turns or not — a recording
    #: with no turns is worth telling apart from one that never arrived.
    recorded: dict[str, float] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)

    def turns_for(self, participant: str) -> list[Turn]:
        return [turn for turn in self.turns if turn.participant == participant]

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "duration": round(self.duration, 3),
            "recorded": {name: round(value, 3) for name, value in self.recorded.items()},
            "turns": [turn.as_dict() for turn in self.turns],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> SessionTimeline | None:
        """Parse a timeline, or ``None`` when the document is not usable.

        Unusable means the whole document: a session whose timeline was lost
        falls back to the per-participant transcript, which is what happened
        before any of this existed, so degrading is always safe.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        turns = [
            turn
            for turn in (Turn.from_dict(item) for item in data.get("turns") or [])
            if turn
        ]
        recorded: dict[str, float] = {}
        for name, value in (data.get("recorded") or {}).items():
            if isinstance(name, str) and isinstance(value, (int, float)):
                recorded[name] = float(value)

        started_at = data.get("started_at")
        duration = data.get("duration")
        return cls(
            started_at=started_at if isinstance(started_at, str) else "",
            duration=float(duration) if isinstance(duration, (int, float)) else 0.0,
            recorded=recorded,
            turns=turns,
        )

    @classmethod
    def load(cls, path: str | Path) -> SessionTimeline | None:
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError:
            return None
        return cls.from_json(raw)

    def write(self, path: str | Path) -> bool:
        """Write atomically, the way the control frame is written.

        The daemon rewrites this while the meeting runs, so a reader — the
        post-processing step of a previous connection, or a human — must never
        see half a document.
        """
        target = Path(path)
        temporary: str | None = None
        try:
            handle_fd, temporary = tempfile.mkstemp(
                dir=target.parent, prefix=".timeline-", suffix=".tmp"
            )
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                handle.write(self.to_json())
            os.replace(temporary, target)
        except OSError:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
            return False
        return True


def utc_now() -> str:
    """The timestamp a timeline records as its start."""
    return datetime.now(UTC).isoformat(timespec="seconds")
