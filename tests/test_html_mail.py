# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the mail's stylesheet inliner.

The inliner is the only thing standing between a re-brandable stylesheet and
a mail that arrives unstyled in Gmail, so the cases that matter here are the
ones where it could quietly do the wrong thing: a variable it fails to
resolve, a rule that outranks one it should lose to, or markup — comments,
entities, conditional comments — that it mangles on the way through.
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser

import pytest

from jitsi_audio_bridge import html_mail

MS = html_mail.MEDIA_QUERY_PLACEHOLDER


# -- variables ---------------------------------------------------------------


def test_variables_are_substituted():
    css = ":root { --ink: #ff0000; } .x { color: var(--ink); }"
    out = html_mail.inline(css, '<p class="x">hi</p>')
    assert 'style="color: #ff0000"' in out
    assert "var(--" not in out


def test_variables_may_chain_and_carry_a_fallback():
    css = (
        ":root { --a: var(--b); --b: #123456; }"
        " .x { color: var(--a); border-color: var(--nope, blue); }"
    )
    out = html_mail.inline(css, '<p class="x">hi</p>')
    assert "color: #123456" in out
    assert "border-color: blue" in out


def test_an_unresolvable_variable_without_a_fallback_becomes_empty():
    out = html_mail.inline(".x { color: var(--nope); }", '<p class="x">hi</p>')
    assert "color: " in out
    assert "var(--" not in out


# -- cascade -----------------------------------------------------------------


def test_rules_apply_in_source_order_not_class_order():
    css = ".a { color: red; } .b { color: blue; }"
    assert "color: blue" in html_mail.inline(css, '<p class="a b">hi</p>')
    # The order in the class attribute is not what decides it.
    assert "color: blue" in html_mail.inline(css, '<p class="b a">hi</p>')


def test_declarations_from_several_classes_are_merged():
    css = ".a { color: red; } .b { padding-top: 4px; }"
    out = html_mail.inline(css, '<p class="a b">hi</p>')
    assert "color: red" in out
    assert "padding-top: 4px" in out


def test_an_elements_own_style_wins():
    out = html_mail.inline(".x { color: red; }", '<p class="x" style="color: green">hi</p>')
    assert "color: green" in out
    assert "red" not in out


def test_an_elements_own_style_survives_alongside_new_declarations():
    out = html_mail.inline(".x { color: red; }", '<p class="x" style="font-weight: bold">hi</p>')
    assert "font-weight: bold" in out
    assert "color: red" in out


def test_tag_selectors_match():
    out = html_mail.inline("body { background-color: #fff; }", "<body><p>hi</p></body>")
    assert '<body style="background-color: #fff">' in out


def test_an_element_nothing_matches_is_left_alone():
    assert html_mail.inline(".x { color: red; }", "<p>hi</p>") == "<p>hi</p>"


# -- media queries -----------------------------------------------------------


def test_media_queries_replace_the_placeholder_rather_than_inlining():
    css = (
        ":root { --narrow: 18px; }"
        "@media only screen and (max-width: 600px) {"
        "  .x { padding-left: var(--narrow) !important; }"
        "}"
    )
    out = html_mail.inline(css, f'<style>{MS}</style><p class="x">hi</p>')
    assert "@media" in out
    # Lifted into the <style> block, with its variable already resolved:
    # a client that keeps it has no :root to resolve against.
    assert "padding-left: 18px" in out
    assert MS not in out
    # ...and not also splashed onto the element, which keeps its class so a
    # client that does honour the <style> block can still match it.
    assert '<p class="x">hi</p>' in out


# -- markup fidelity ---------------------------------------------------------


def test_conditional_comments_comments_and_entities_survive():
    markup = (
        "<!--[if mso]><table><tr><td><![endif]-->"
        '<p class="x">&bull; &amp; &#128196; &mdash; café</p>'
    )
    out = html_mail.inline(".x { color: red; }", markup)
    assert "<!--[if mso]>" in out
    assert "<![endif]-->" in out
    assert "&bull;" in out
    assert "&amp;" in out
    assert "&#128196;" in out
    assert "café" in out


def test_the_doctype_is_kept():
    out = html_mail.inline("body { margin-top: 0; }", "<!DOCTYPE html><html><body>x</body></html>")
    assert out.startswith("<!DOCTYPE html>")


# -- what it refuses to do ---------------------------------------------------


@pytest.mark.parametrize(
    "selector",
    [".a > .b", ".a .b", ".a:hover", "#id", ".a::before", "[data-x]"],
)
def test_an_unsupported_selector_is_reported_and_skipped(selector, caplog):
    with caplog.at_level(logging.WARNING):
        out = html_mail.inline(f"{selector} {{ color: red; }}", '<p class="a" id="id">hi</p>')
    assert "color" not in out
    assert "not supported" in caplog.text


def test_an_unsupported_at_rule_is_reported_and_skipped(caplog):
    with caplog.at_level(logging.WARNING):
        out = html_mail.inline("@import url(evil.css); .x { color: red; }", '<p class="x">hi</p>')
    assert "not supported" in caplog.text
    # The rule beside it is unaffected.
    assert "color: red" in out


# -- the shipped template ----------------------------------------------------


def test_the_shipped_template_renders_fully(mail_context):
    out = html_mail.render(mail_context)
    assert "var(--" not in out
    assert MS not in out
    assert "@media" in out
    assert "Meeting Summary &amp; Transcript" in out
    # The MSO scaffolding Outlook needs is still conditional.
    assert out.count("<!--[if mso]>") == 3


def test_the_template_escapes_what_the_meeting_contains(mail_context):
    """Speaker names come from the sender's metadata; turn text from Whisper.

    Both are untrusted and both are interpolated straight into the mail, so
    autoescape is the boundary that stops a participant putting markup into
    everyone else's inbox.  Asserted on the *inlined* output, because that is
    what is sent — the inliner walks the document afterwards and must not
    unescape anything on the way through.
    """
    out = html_mail.render(mail_context)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert "<script>alert(1)</script>" not in out


def test_the_sign_off_keeps_its_line_breaks_without_gaining_markup(mail_context):
    out = html_mail.render(mail_context)
    assert "Best regards,<br>Automated meeting transcription" in out


class _ClassCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.classes: set[str] = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == "class" and value:
                self.classes.update(value.split())


def test_every_class_in_the_template_has_a_rule(mail_context):
    """A class in the HTML that no rule matches is a typo.

    It would not fail the build or the send — it would just ship an element
    with no styles, which is exactly the kind of thing nobody notices until a
    customer says the mail looks broken.  Collected from the *rendered* mail,
    so a class built from a variable (``chip-{{ n }}``) is checked in the form
    it actually takes.
    """
    _, css = html_mail.load_default()
    _, rules, _ = html_mail._parse_stylesheet(css)
    styled = {name for kind, name, _ in rules if kind == "class"}

    collector = _ClassCollector()
    collector.feed(html_mail.render(mail_context))
    assert collector.classes - styled == set()


def test_the_shipped_stylesheet_avoids_padding_and_margin_shorthands():
    """Declarations merge per element, so a shorthand erases a longhand.

    ``.px`` sets the side padding and ``.band-*`` sets the vertical rhythm on
    the same cell; a ``padding:`` in either would silently wipe out the other.
    """
    _, css = html_mail.load_default()
    _, rules, _ = html_mail._parse_stylesheet(css)
    offenders = [
        f"{name}: {prop}"
        for _, name, declarations in rules
        for prop in declarations
        if prop in {"padding", "margin"}
    ]
    assert offenders == []
