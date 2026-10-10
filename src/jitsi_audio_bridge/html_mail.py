# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Turn the mail's stylesheet into styles the mail clients will actually keep.

The mail's appearance is authored in ``templates/summary_email.css`` so that a
deployment can re-brand it by editing one file.  That stylesheet can never be
sent as it is: Gmail discards linked stylesheets and strips most ``<style>``
blocks, Outlook's Word engine ignores them, and a mail that relies on either
arrives unstyled.  So the stylesheet is applied to the elements here, at send
time, and the mail goes out with everything on ``style=""`` attributes.

Only a small, deliberate subset of CSS is supported, and anything outside it
is reported rather than silently dropped:

``:root { --name: value; }``
    Theme variables, referenced anywhere as ``var(--name)`` or
    ``var(--name, fallback)``.  Resolved before anything is emitted, so no
    ``var()`` survives into the mail.
``.class``, ``tag``, and comma-separated lists of those
    Matched against each element's class list and tag name.  No combinators,
    no pseudo-classes, no attribute selectors, no ids.
``@media ... { ... }``
    Not inlined — a media query has no meaning on a single element.  Lifted
    into the mail's own ``<style>`` block instead, with its variables resolved.

Declarations are applied in source order, and an element's existing
``style=""`` attribute wins over the stylesheet, which is what CSS does.  Use
longhand properties in the stylesheet: declarations from several classes are
merged onto one element, and a shorthand would silently erase a longhand set
by another class.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where the mail's structure and stylesheet live.
TEMPLATE_DIR = Path(__file__).parent / "templates"

#: Stands in for the media queries inside the template's ``<style>`` block.
MEDIA_QUERY_PLACEHOLDER = "/*{{MEDIA_QUERIES}}*/"

_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_VAR = re.compile(r"var\(\s*(--[A-Za-z0-9_-]+)\s*(?:,\s*([^()]*?)\s*)?\)")
_DECLARATION = re.compile(r"([-\w]+)\s*:\s*(.+?)\s*$", re.DOTALL)


def _strip_comments(css: str) -> str:
    return _COMMENT.sub(" ", css)


def _parse_declarations(body: str) -> dict[str, str]:
    """``"a: 1; b: 2"`` -> ``{"a": "1", "b": "2"}``, in source order.

    A declaration that is not ``name: value`` is skipped rather than raising:
    one malformed line in a company's stylesheet should cost that declaration,
    not the mail.
    """
    found: dict[str, str] = {}
    for chunk in body.split(";"):
        match = _DECLARATION.match(chunk.strip())
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def _iter_blocks(css: str):
    """Yield ``(prelude, body)`` for every block, at any nesting depth.

    Nested blocks (the body of an ``@media``) come back as part of their
    parent's body, because the parent is consumed whole.

    A statement at-rule — ``@import url(...);`` — has no body and ends at a
    semicolon.  It is yielded with an empty body rather than being allowed to
    run on into the next rule's prelude, which would silently discard that
    rule along with it.
    """
    index = 0
    while True:
        opening = css.find("{", index)
        prelude = css[index:] if opening < 0 else css[index:opening]
        semicolon = prelude.find(";")
        if semicolon >= 0:
            statement = prelude[:semicolon].strip()
            if statement:
                yield statement, ""
            index += semicolon + 1
            continue
        if opening < 0:
            return
        depth = 1
        cursor = opening + 1
        while cursor < len(css) and depth:
            if css[cursor] == "{":
                depth += 1
            elif css[cursor] == "}":
                depth -= 1
            cursor += 1
        yield prelude.strip(), css[opening + 1 : cursor - 1]
        index = cursor


#: One rule: what it matches on, the name it matches, and its declarations.
_Rule = tuple[str, str, dict[str, str]]


def _parse_stylesheet(css: str) -> tuple[dict[str, str], list[_Rule], str]:
    """Split a stylesheet into variables, ordered rules, and media queries."""
    variables: dict[str, str] = {}
    rules: list[tuple[str, str, dict[str, str]]] = []
    media: list[str] = []

    for prelude, body in _iter_blocks(_strip_comments(css)):
        if prelude == ":root":
            variables.update(_parse_declarations(body))
        elif prelude.startswith("@"):
            if prelude.startswith("@media"):
                media.append(f"{prelude} {{{body}}}")
            else:
                logger.warning("stylesheet: %r is not supported and was ignored", prelude)
        else:
            declarations = _parse_declarations(body)
            for selector in prelude.split(","):
                selector = selector.strip()
                if selector.startswith(".") and selector[1:].replace("-", "").isalnum():
                    rules.append(("class", selector[1:], declarations))
                elif selector.isidentifier():
                    rules.append(("tag", selector.lower(), declarations))
                else:
                    logger.warning(
                        "stylesheet: selector %r is not supported and was ignored",
                        selector,
                    )
    return variables, rules, " ".join(media)


def _resolve(value: str, variables: dict[str, str]) -> str:
    """Expand ``var(--name)`` references, repeatedly, so vars may chain."""
    for _ in range(10):
        expanded = _VAR.sub(
            lambda match: variables.get(
                match.group(1), (match.group(2) or "").strip()
            ),
            value,
        )
        if expanded == value:
            return expanded
        value = expanded
    logger.warning("stylesheet: variable reference did not settle: %r", value)
    return value


class _Inliner(HTMLParser):
    """Rewrites the open tag of every element to carry its styles.

    Everything else — text, entities, comments, the doctype — is re-emitted
    untouched.  Comments matter in particular: the MSO conditional comments
    this template uses to give Outlook a fixed-width table are comments, and
    dropping or reflowing them would break the mail in Outlook.
    """

    def __init__(self, variables: dict[str, str], rules: list[tuple[str, str, dict[str, str]]]):
        super().__init__(convert_charrefs=False)
        self._variables = variables
        self._rules = rules
        self._out: list[str] = []

    # -- output ------------------------------------------------------------

    def result(self) -> str:
        return "".join(self._out)

    # -- styles ------------------------------------------------------------

    def _declarations_for(self, tag: str, classes: list[str]) -> dict[str, str]:
        merged: dict[str, str] = {}
        for kind, name, declarations in self._rules:
            if (kind == "class" and name in classes) or (kind == "tag" and name == tag):
                merged.update(declarations)
        return merged

    def _start_tag(self, tag: str, attrs: list[tuple[str, str | None]]) -> str:
        raw = self.get_starttag_text() or f"<{tag}>"
        by_name = {name.lower(): (value or "") for name, value in attrs}

        declarations = self._declarations_for(
            tag.lower(), (by_name.get("class") or "").split()
        )
        # An element's own style attribute wins, as it does in CSS.
        existing = _parse_declarations(by_name["style"]) if "style" in by_name else {}
        declarations.update(existing)
        if not declarations:
            return raw

        style = "; ".join(
            f"{name}: {_resolve(value, self._variables)}"
            for name, value in declarations.items()
        )
        if "style" in by_name:
            return re.sub(r'\sstyle="[^"]*"', f' style="{style}"', raw, count=1)
        return re.sub(r"\s*/?>$", f' style="{style}">', raw, count=1)

    # -- HTMLParser hooks --------------------------------------------------

    def handle_starttag(self, tag, attrs):
        self._out.append(self._start_tag(tag, attrs))

    def handle_startendtag(self, tag, attrs):
        self._out.append(self._start_tag(tag, attrs))

    def handle_endtag(self, tag):
        self._out.append(f"</{tag}>")

    def handle_data(self, data):
        self._out.append(data)

    def handle_entityref(self, name):
        self._out.append(f"&{name};")

    def handle_charref(self, name):
        self._out.append(f"&#{name};")

    def handle_comment(self, data):
        self._out.append(f"<!--{data}-->")

    def handle_decl(self, decl):
        self._out.append(f"<!{decl}>")

    def handle_pi(self, data):
        self._out.append(f"<?{data}>")

    def unknown_decl(self, data):
        self._out.append(f"<![{data}]>")


def inline(css: str, html: str) -> str:
    """Return *html* with *css* applied to its elements.

    The media queries in *css* are not inlined; they replace the placeholder
    in the template's ``<style>`` block, which is the only place a client will
    still honour them.
    """
    variables, rules, media = _parse_stylesheet(css)
    inliner = _Inliner(variables, rules)
    inliner.feed(html)
    inliner.close()
    return inliner.result().replace(
        MEDIA_QUERY_PLACEHOLDER, _resolve(media, variables)
    )


def load_default() -> tuple[str, str]:
    """The stock template and its stylesheet, as text."""
    return (
        (TEMPLATE_DIR / "summary_email.html").read_text(encoding="utf-8"),
        (TEMPLATE_DIR / "summary_email.css").read_text(encoding="utf-8"),
    )


def render(html: str | None = None, css: str | None = None) -> str:
    """The mail's HTML, styled and ready to attach as the text/html part.

    Both arguments default to the packaged template, so a deployment that
    wants its own branding passes a stylesheet (and only a stylesheet) read
    from wherever it keeps one.
    """
    default_html, default_css = load_default()
    markup = default_html if html is None else html
    stylesheet = default_css if css is None else css
    return inline(stylesheet, markup)
