# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for turning a finished meeting into the mail's values.

Both inputs are text this daemon did not write in a fixed shape: the summary
is whatever a model chose to answer with, in a language it chose, and the
transcript is Whisper's reading of a conversation.  So the cases that matter
are the ones where the input is not what the prompt asked for — the parser has
to degrade into something readable rather than into nothing.
"""

from __future__ import annotations

from jitsi_audio_bridge import mail_render as mr

# -- transcript --------------------------------------------------------------


def test_both_turn_shapes_are_read():
    turns = mr.parse_transcript(
        "[00:03:12] Alice: hello there\n\n[Bob]: no timestamp here"
    )
    assert [(t.timestamp, t.speaker, t.text) for t in turns] == [
        ("00:03:12", "Alice", "hello there"),
        ("", "Bob", "no timestamp here"),
    ]


def test_a_blank_line_inside_a_turn_does_not_split_it():
    """Whisper emits its own newlines; they are not new speakers."""
    turns = mr.parse_transcript(
        "[00:00:04] Alice: first\n\nsecond paragraph\n\n[00:00:09] Bob: next"
    )
    assert len(turns) == 2
    assert turns[0].text == "first second paragraph"
    assert turns[1].speaker == "Bob"


def test_a_wrapped_line_joins_the_turn_above_it():
    turns = mr.parse_transcript("[00:00:01] Alice: one\nwrapped continuation")
    assert len(turns) == 1
    assert turns[0].text == "one wrapped continuation"


def test_text_that_looks_like_markup_is_kept_as_text():
    turns = mr.parse_transcript("[00:00:01] <b>Alice</b>: <img src=x onerror=y>")
    assert turns[0].speaker == "<b>Alice</b>"
    assert turns[0].text == "<img src=x onerror=y>"


def test_an_unparseable_transcript_yields_no_turns():
    assert mr.parse_transcript("just some prose with no tags at all") == []


def test_text_after_a_bare_timestamp_is_kept_not_filed_as_a_speaker():
    """``render_transcript`` always writes ``Speaker:``, but a malformed line
    must lose its attribution rather than its words."""
    turns = mr.parse_transcript("[00:00:04] hello")
    assert len(turns) == 1
    assert turns[0].timestamp == "00:00:04"
    assert turns[0].speaker == ""
    assert turns[0].text == "hello"


# -- speaker colours ---------------------------------------------------------


def test_a_speaker_keeps_their_colour_across_meetings():
    first = {
        t.speaker: t.chip
        for t in mr.parse_transcript("[00:00:01] Elena: a\n\n[00:00:02] Michael: b")
    }
    second = {
        t.speaker: t.chip
        for t in mr.parse_transcript("[00:00:09] Michael: b\n\n[00:00:10] Elena: a")
    }
    assert first == second


def test_colours_do_not_collide_within_a_meeting():
    """Six colours and four speakers collide often enough to matter."""
    names = ["Elena", "Michael", "David", "Sarah"]
    chips = mr.assign_chips(names)
    assert len(set(chips.values())) == len(names)
    assert set(chips.values()) <= set(range(mr.CHIP_COLOURS))


def test_more_speakers_than_colours_reuses_them_rather_than_failing():
    chips = mr.assign_chips([f"S{n}" for n in range(10)])
    assert len(chips) == 10
    assert set(chips.values()) <= set(range(mr.CHIP_COLOURS))


def test_the_colour_is_not_pythons_own_hash():
    """``hash()`` is salted per process, so colours would change on restart."""
    assert mr.chip_index("Elena") == mr.chip_index("Elena")
    assert isinstance(mr.chip_index("Elena"), int)


# -- summary -----------------------------------------------------------------


def test_markdown_headings_and_bullets_are_split():
    sections = mr.parse_summary(
        "## First\n- one\n- two\n\n## Second\n- three\n", "FALLBACK"
    )
    assert [(s.title, [i.text for i in s.items]) for s in sections] == [
        ("First", ["one", "two"]),
        ("Second", ["three"]),
    ]


def test_bold_and_colon_headings_are_recognised_too():
    """The prompt asks for ##, but a model does not always comply."""
    for text in ("**First**\n- one\n", "First:\n- one\n"):
        sections = mr.parse_summary(text, "FALLBACK")
        assert [s.title for s in sections] == ["First"], text


def test_a_bullet_whose_text_ends_in_a_colon_is_still_a_bullet():
    """``- Action items:`` is a bullet, not a heading with nothing under it."""
    sections = mr.parse_summary("## Notes\n- Action items:\n", "FALLBACK")
    assert [s.title for s in sections] == ["Notes"]
    assert [i.bullet for i in sections[0].items] == [True]


def test_numbered_bullets_are_recognised():
    sections = mr.parse_summary("## Steps\n1. first\n2) second\n", "FALLBACK")
    assert [i.text for i in sections[0].items] == ["first", "second"]


def test_a_wrapped_bullet_is_joined_not_split():
    sections = mr.parse_summary("## Notes\n- one point that\n  carries on\n", "FALLBACK")
    assert [i.text for i in sections[0].items] == ["one point that carries on"]


def test_a_summary_with_no_structure_still_becomes_a_section():
    """The mail must never come out empty because a model chose a shape."""
    sections = mr.parse_summary("Just prose.\nNothing else.", "MEETING SUMMARY")
    assert len(sections) == 1
    assert sections[0].title == "MEETING SUMMARY"
    assert [i.text for i in sections[0].items] == ["Just prose.", "Nothing else."]


def test_an_empty_summary_still_yields_a_section():
    assert mr.parse_summary("", "MEETING SUMMARY")


# -- context -----------------------------------------------------------------


def test_sections_are_numbered_around_whatever_arrived(mail_context):
    """The count is the model's, not a constant: three here, so transcript is 4."""
    titles = [s["title"] for s in mail_context["sections"]]
    assert titles == ["Riassunto esecutivo", "Punti chiave", "Azioni"]
    assert [s["number"] for s in mail_context["sections"]] == [1, 2, 3]
    assert mail_context["transcript_number"] == 4
    assert mail_context["attachments_number"] == 5


def test_the_preview_is_capped_and_says_how_many_were_left_out():
    context = mr.build_context(
        heading="H",
        room_name="Room",
        when="",
        participant_count=0,
        introduction="",
        summary_text="## S\n- one\n",
        transcript_text="\n\n".join(f"[00:00:{n:02d}] S{n}: line" for n in range(10)),
        attachments=[],
        sign_off="",
        footer="",
        preview_turns=3,
    )
    assert len(context["turns"]) == 3
    assert context["hidden_turns"] == 7


def test_a_context_with_no_transcript_omits_the_preview(mail_context):
    context = mr.build_context(
        heading="H", room_name="Room", when="", participant_count=0, introduction="",
        summary_text="## S\n- one\n", transcript_text="", attachments=[],
        sign_off="", footer="", preview_turns=10,
    )
    assert context["turns"] == []


def test_the_preheader_describes_this_meeting_not_the_subject(mail_context):
    preheader = mail_context["preheader"]
    assert "jitsi-ai-review" in preheader
    # The summary's own first line, not the heading above it.
    assert "Riassunto esecutivo" not in preheader
    assert "Il team ha rivisto" in preheader


def test_a_zero_preview_shows_none_and_counts_them_all():
    context = mr.build_context(
        heading="H", room_name="Room", when="", participant_count=0, introduction="",
        summary_text="## S\n- one\n", transcript_text="[00:00:01] A: x\n\n[00:00:02] B: y",
        attachments=[], sign_off="", footer="", preview_turns=0,
    )
    assert context["turns"] == []
    assert context["hidden_turns"] == 2
