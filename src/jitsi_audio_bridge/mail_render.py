# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Turn a finished meeting into the values the HTML mail renders.

Two pieces of text arrive here and neither has a reliable structure:

``summary_text``
    The model's answer, stripped and nothing else (see ``ai_client``).  The
    prompt asks for three named sections and, since this module was written,
    for markdown headings and bullets — but a model does not always comply,
    and the headings are written in the meeting's own language.  So the
    parser matches *shapes* (any heading style, any bullet marker), never
    words, and falls back to one section holding the whole text.
``transcript_text``
    Rendered by ``daemon.render_transcript``, so the shape is our own:
    ``[HH:MM:SS] Speaker: text`` for timed turns and ``[Speaker]: text`` for
    untimed ones, which may be mixed in one file.  The speaker can be a
    display name from ``metadata.json`` — untrusted — or a raw file stem like
    ``participant-michael-a0`` when nothing named it.

Nothing here reads a file, the environment or the network; the text is handed
in by the caller, which is what keeps this testable without a meeting.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

#: How many colour classes ``summary_email.css`` defines for speaker chips.
#: Keep in step with the ``--sp<n>-bg`` entries in its theme block.
CHIP_COLOURS = 6

#: ``[00:03:12] ...`` — the offset ``timeline.format_offset`` writes.
_TIMESTAMP = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")

#: A turn opens with a bracketed label.  The bracket is closed at the *first*
#: ``]``: a display name containing one would otherwise swallow the text, and
#: a mis-split speaker is better than a lost line.
_TURN = re.compile(r"^\[([^\]]*)\](.*)$")

#: Any of the heading shapes a model reaches for: ``## Title``, ``**Title**``,
#: ``Title:`` alone on a line.  The last is capped in length and forbidden
#: from containing sentence punctuation, so ordinary prose ending in a colon
#: is not mistaken for a heading.
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$")
_BOLD_HEADING = re.compile(r"^\s*\*\*(.+?)\*\*:?\s*$")
_LABEL_HEADING = re.compile(r"^\s*([^.!?,;]{1,80}):\s*$")

#: ``- item``, ``* item``, ``• item``, ``1. item``, ``1) item``.
_BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+(.*\S)\s*$")

#: Sentence-ending punctuation.  A line not ending in one is assumed to have
#: been wrapped by the model and is joined to the line above.
_TERMINAL = ".!?:;"


@dataclass(frozen=True)
class Turn:
    """One speaking turn, ready to render."""

    timestamp: str
    speaker: str
    text: str
    #: Index into the stylesheet's speaker palette.
    chip: int


@dataclass(frozen=True)
class Item:
    """A bullet or a paragraph inside a summary section."""

    text: str
    bullet: bool


@dataclass(frozen=True)
class Section:
    """One heading of the model's summary, with whatever sat under it."""

    title: str
    items: list[Item] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.items


@dataclass(frozen=True)
class Recording:
    """The meeting's recording, when one was archived and linked."""

    url: str
    label: str
    expires: str = ""


def chip_index(speaker: str) -> int:
    """A stable colour index for *speaker*.

    ``hash()`` is salted per process, so a speaker would change colour every
    time the daemon restarted — and every meeting would disagree with the last
    about who was which colour.  A digest is stable across processes, hosts
    and versions.
    """
    digest = hashlib.sha256(speaker.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % CHIP_COLOURS


def assign_chips(speakers: list[str]) -> dict[str, int]:
    """Give every speaker in one meeting their own colour.

    The hash alone is not enough: six colours and four speakers collide often
    enough that two people would regularly share one, which defeats the point
    of highlighting them at all.  Each speaker takes their hash's colour when
    it is free and the next free one when it is not, so colours stay stable
    across meetings *and* distinct within one — up to the palette size, after
    which they necessarily repeat.
    """
    assigned: dict[str, int] = {}
    used: set[int] = set()
    for speaker in speakers:
        if speaker in assigned:
            continue
        preferred = chip_index(speaker)
        choice = preferred
        for offset in range(CHIP_COLOURS):
            candidate = (preferred + offset) % CHIP_COLOURS
            if candidate not in used:
                choice = candidate
                break
        assigned[speaker] = choice
        used.add(choice)
    return assigned


def _split_label(rest: str) -> tuple[str, str] | None:
    """``"Alice: hello"`` -> ``("Alice", "hello")``, or ``None`` if unseparated.

    The ``None`` matters: without it a line the model never wrote — a
    timestamp with nothing after it, or no colon at all — would file the words
    as a speaker name and drop them from the mail.
    """
    head, separator, tail = rest.partition(":")
    if not separator:
        return None
    return head.strip(), tail.lstrip()


def parse_transcript(text: str) -> list[Turn]:
    """Split a rendered transcript back into turns.

    A turn is opened by a line starting with a bracketed label.  Any following
    line that is not blank and does not open a turn belongs to it — Whisper
    output may contain newlines of its own, and treating those as new turns
    would invent speakers.  Blank lines are dropped: they are this format's
    separator, and a paragraph break inside one turn is not worth the risk of
    splitting the turn instead.
    """
    turns: list[Turn] = []
    for line in text.splitlines():
        match = _TURN.match(line)
        if match is not None:
            label, rest = match.group(1), match.group(2)
            if _TIMESTAMP.match(label.strip()):
                timestamp = label.strip()
                split = _split_label(rest.strip())
                # Not our format: keep the words as text, attributed to nobody.
                speaker, body = split if split is not None else ("", rest.strip())
            else:
                # ``[Alice]: hello`` puts the name in the bracket, so what
                # follows is the separator and then the text — not another
                # name.  Anything else is not this format, and the text is
                # kept rather than dropped.
                timestamp = ""
                speaker = label.strip()
                body = rest.lstrip()
                body = body[1:].lstrip() if body.startswith(":") else body.strip()
            turns.append(Turn(timestamp=timestamp, speaker=speaker, text=body, chip=0))
            continue

        if not line.strip():
            continue
        if turns:
            previous = turns[-1]
            joined = f"{previous.text} {line.strip()}".strip()
            turns[-1] = Turn(
                timestamp=previous.timestamp,
                speaker=previous.speaker,
                text=joined,
                chip=previous.chip,
            )
    return _with_chips(turns)


def _with_chips(turns: list[Turn]) -> list[Turn]:
    """Colour the turns once every speaker is known, not as they arrive."""
    chips = assign_chips([turn.speaker for turn in turns])
    return [
        Turn(
            timestamp=turn.timestamp,
            speaker=turn.speaker,
            text=turn.text,
            chip=chips[turn.speaker],
        )
        for turn in turns
    ]


def _heading_of(line: str) -> str | None:
    """The heading *line* is, if it is one."""
    for pattern in (_MD_HEADING, _BOLD_HEADING, _LABEL_HEADING):
        match = pattern.match(line)
        if match is not None:
            title = match.group(1).strip()
            if title:
                return title
    return None


def parse_summary(text: str, fallback_title: str) -> list[Section]:
    """Split the model's summary into sections, without trusting its words.

    A summary that matches nothing still produces one section holding all of
    it: the mail must never come out with an empty body because a model chose
    an unfamiliar shape.
    """
    sections: list[Section] = []
    current: list[Item] = []
    title: str | None = None

    def flush() -> None:
        nonlocal current, title
        if title is not None or current:
            sections.append(Section(title=title or "", items=list(current)))
        current = []
        title = None

    for line in text.splitlines():
        if not line.strip():
            continue

        # Bullets are tested before headings, because a bullet whose text ends
        # in a colon — "- Action items:" — otherwise reads as a heading.
        bullet = _BULLET.match(line)
        if bullet is not None:
            current.append(Item(text=bullet.group(1), bullet=True))
            continue

        heading = _heading_of(line)
        if heading is not None:
            flush()
            title = heading
            continue

        stripped = line.strip()
        if current and not current[-1].text.endswith(tuple(_TERMINAL)):
            # The line above did not finish its sentence, so this is the rest
            # of it rather than a new paragraph.
            current[-1] = Item(text=f"{current[-1].text} {stripped}", bullet=current[-1].bullet)
        else:
            current.append(Item(text=stripped, bullet=False))

    flush()

    if not sections:
        return [Section(title=fallback_title, items=[])]
    if len(sections) == 1 and not sections[0].title:
        return [Section(title=fallback_title, items=sections[0].items)]
    return sections


def build_context(
    *,
    heading: str,
    room_name: str,
    when: str,
    participant_count: int,
    introduction: str,
    summary_text: str,
    transcript_text: str,
    attachments: list[str],
    sign_off: str,
    footer: str,
    preview_turns: int,
    recording: Recording | None = None,
) -> dict[str, object]:
    """Everything ``summary_email.html.j2`` needs, already counted and sliced.

    Section numbers are assigned here rather than in the template so that they
    stay correct whatever the model produced: its sections take 1..N, and the
    transcript and attachments follow.  The mock-up this design came from
    showed four fixed numbers, which only holds for a two-section summary.
    """
    sections = [
        section
        for section in parse_summary(summary_text, heading)
        if not section.is_empty
    ]
    if not sections:
        # Every heading came back empty. Show what the model wrote rather than
        # mailing a summary whose sections have nothing under them.
        sections = [Section(title=heading, items=[Item(text=summary_text.strip(), bullet=False)])]

    turns = parse_transcript(transcript_text)
    preview = turns[:preview_turns] if preview_turns >= 0 else turns
    hidden = len(turns) - len(preview)

    numbered = []
    for number, section in enumerate(sections, start=1):
        # "entries", not "items": Jinja resolves attributes before keys, so
        # ``section.items`` on a dict hands the template the dict's own
        # method rather than the list.
        numbered.append({"number": number, "title": section.title, "entries": section.items})
    transcript_number = len(sections) + 1
    attachments_number = transcript_number + 1

    return {
        "heading": heading,
        "room_name": room_name,
        "when": when,
        "participant_count": participant_count,
        "introduction": introduction,
        "sections": numbered,
        "turns": preview,
        "hidden_turns": hidden,
        "transcript_number": transcript_number,
        "attachments_number": attachments_number,
        "attachments": attachments,
        "recording": recording,
        "sign_off": sign_off,
        "footer": footer,
        "preheader": _preheader(room_name, when, summary_text, len(turns)),
    }


def _preheader(room_name: str, when: str, summary_text: str, turn_count: int) -> str:
    """The line a mail client shows beside the subject.

    Built from the summary's own first words rather than a fixed string, so it
    says something about *this* meeting instead of repeating the subject.
    """
    first = ""
    for line in summary_text.splitlines():
        candidate = line.strip().lstrip("#*- \t")
        if candidate and _heading_of(line) is None:
            first = candidate
            break
    parts = [part for part in (room_name, when) if part]
    lead = " — ".join(parts)
    if first:
        return f"{lead}: {first[:140]}"
    return f"{lead}: {turn_count} transcript turns."
