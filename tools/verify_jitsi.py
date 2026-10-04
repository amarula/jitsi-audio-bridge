"""Check a Jitsi deployment against docs/jitsi-integration.md.

Read-only. Three sections, selected with ``--only``:

``config``
    Parse the effective Jicofo, Prosody and jitsi-meet files and check each
    documented requirement.  FAIL only on unambiguous misconfiguration, WARN
    where a deliberate choice is possible, SKIP when a file or permission is
    missing.
``probe``
    Act as the JVB against the exact URL and headers Jicofo is configured to
    use: handshake, ``ping`` → ``pong``, ``session-end``.  Sends no audio, so
    the bridge records an empty session and mails nothing.
``logs``
    Scan recent ``jitsi-videobridge2`` and ``jicofo`` journal entries for the
    connect lifecycle, or the failure modes, after a test meeting.

Examples:

    python3 -m tools.verify_jitsi                       # the config files only
    python3 -m tools.verify_jitsi --only config,probe   # also reach the bridge
    python3 -m tools.verify_jitsi --only logs --since "10 min ago"

Exit status is 0 when nothing failed, 1 when a check failed, and 2 for a usage
or discovery problem (a named file that does not exist, several domains to
choose from without ``--domain``, and so on).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

#: The one route the bridge serves; anything else is closed with 1008.
WEBSOCKET_PATH = "/transcribe"

#: The name this tool uses for the meeting id when resolving the template.
DEFAULT_SESSION_ID = "verify-jitsi"

#: Debian/Ubuntu package locations.
DEFAULT_JICOFO_CONF = Path("/etc/jitsi/jicofo/jicofo.conf")
DEFAULT_CUSTOM_JICOFO_CONF = Path("/etc/jitsi/jicofo/custom-jicofo.conf")
DEFAULT_PROSODY_MAIN = Path("/etc/prosody/prosody.cfg.lua")
DEFAULT_PROSODY_CONF_AVAIL = Path("/etc/prosody/conf.avail")
DEFAULT_PROSODY_CONF_D = Path("/etc/prosody/conf.d")
DEFAULT_MEET_DIR = Path("/etc/jitsi/meet")
DEFAULT_JVB_CONF = Path("/etc/jitsi/videobridge/jvb.conf")
DEFAULT_PLUGIN_DIRS = (
    Path("/usr/share/jitsi-meet/prosody-plugins"),
    # Modules that ship with Prosody itself, which a config may also enable.
    Path("/usr/lib/prosody/modules"),
    Path("/usr/lib/prosody/modules/share/lua/5.1"),
)

#: Log strings emitted by the upstream components (verified against
#: jitsi-videobridge's Exporter/MediaJsonSerializer and Jicofo's
#: TranscriptionConfig/JitsiMeetConferenceImpl).
JVB_POSITIVE_PATTERNS = (
    "Websocket connected: true",
    "Sending info to transcriber:",
    "Starting SSRC ",
    "Starting with url=",
    "Starting ping with interval=",
)
JVB_FAILURE_PATTERNS = (
    "Ping timeout, reconnecting websocket",
    "Failed to parse incoming websocket message",
    "Max reconnection attempts",
    "Received pong with id=",
    "Failed to send info message",
    "Websocket error",
)
JICOFO_ERROR_PATTERN = "Transcription enabled, but no URL is configured."
JICOFO_WARNING_PATTERN = "Transcriber URL template does not contain"

_RUNTIME_URL = re.compile(r"Starting with url=(\S+)")


class DiscoveryError(Exception):
    """Raised when there is no deployment to check, or the choice is ambiguous."""


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass(frozen=True)
class Check:
    """One requirement's verdict."""

    id: str
    status: Status
    summary: str
    fix: str = ""
    detail: str = ""


@dataclass
class Section:
    name: str
    checks: list[Check] = field(default_factory=list)


# --------------------------------------------------------------------------
# jitsi-meet config.js
# --------------------------------------------------------------------------


def strip_js_comments(text: str) -> str:
    """Blank out JavaScript comments, preserving every other character.

    Line and block comments are replaced by spaces (newlines are kept) so that
    offsets and line numbers survive.  The shipped config.js has the whole
    ``transcription`` block commented out, so matching against the raw file
    finds a setting that is not in effect.

    Regex literals are not understood; an unpaired ``/`` is treated as ordinary
    code.  A regex containing ``//`` before the object being looked for could
    hide it, which makes the tool report "not found" rather than a wrong PASS.
    """
    out: list[str] = []
    index, length = 0, len(text)
    quote: str | None = None
    while index < length:
        char = text[index]
        if quote:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'`":
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < length:
            following = text[index + 1]
            if following == "/":
                while index < length and text[index] != "\n":
                    out.append(" ")
                    index += 1
                continue
            if following == "*":
                out.append("  ")
                index += 2
                while index < length and not (text[index] == "*" and index + 1 < length
                                              and text[index + 1] == "/"):
                    out.append("\n" if text[index] == "\n" else " ")
                    index += 1
                if index < length:
                    out.append("  ")
                    index += 2
                continue
        out.append(char)
        index += 1
    return "".join(out)


def match_brace(text: str, start: int) -> int | None:
    """Return the index of the ``}`` matching the ``{`` at *start*, quotes aside."""
    depth = 0
    quote: str | None = None
    index = start
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\" and index + 1 < len(text):
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def find_js_object(text: str, key: str) -> str | None:
    """Return the body of the first ``key: { ... }`` object in *text*."""
    pattern = re.compile(rf"[\"']?{re.escape(key)}[\"']?\s*:\s*\{{")
    for found in pattern.finditer(text):
        end = match_brace(text, found.end() - 1)
        if end is not None:
            return text[found.end() : end]
    return None


def js_boolean(body: str, name: str) -> bool | None:
    """Return the boolean value of *name* in an object *body*, or None."""
    found = re.search(rf"[\"']?{re.escape(name)}[\"']?\s*:\s*(true|false)\b", body)
    if found is None:
        return None
    return found.group(1) == "true"


# --------------------------------------------------------------------------
# HOCON (jicofo.conf, jvb.conf)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HoconValue:
    path: str
    raw: str
    source: Path
    line: int


@dataclass
class HoconDocument:
    values: dict[str, HoconValue] = field(default_factory=dict)
    sources: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _hocon_lines(text: str) -> str:
    """Put every ``{``, ``}`` and ``,`` on its own line, outside quotes.

    The shipped files are line-oriented, but HOCON allows one-line blocks and
    inline objects (``ping { enabled = true }``).  Breaking them up first lets
    the line-based parser below handle every form.
    """
    out: list[str] = []
    index, length = 0, len(text)
    quote: str | None = None
    while index < length:
        char = text[index]
        if quote:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "#" or (char == "/" and index + 1 < length and text[index + 1] == "/"):
            while index < length and text[index] != "\n":
                out.append(" ")
                index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "*":
            out.append("  ")
            index += 2
            while index < length and not (text[index] == "*" and index + 1 < length
                                          and text[index + 1] == "/"):
                out.append("\n" if text[index] == "\n" else " ")
                index += 1
            if index < length:
                out.append("  ")
                index += 2
            continue
        if char == "{":
            out.append("{\n")
        elif char == "}":
            out.append("\n}\n")
        elif char == ",":
            out.append("\n")
        else:
            out.append(char)
        index += 1
    return "".join(out)


def _split_assignment(line: str) -> tuple[str, str] | None:
    """Split ``key = value`` (or ``key: value``) on the first separator."""
    quote: str | None = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char in "=:":
            return line[:index].strip(), line[index + 1 :].strip()
    return None


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_hocon(text: str, source: Path) -> tuple[dict[str, HoconValue], list[str], list[str]]:
    """Parse HOCON into ``path -> value``, plus include targets and notes.

    Handles nested and dotted keys, inline objects, quoted strings and the two
    comment forms.  ``${...}`` substitutions are kept verbatim; the checks
    report them as unverifiable rather than guessing.  Never raises.
    """
    values: dict[str, HoconValue] = {}
    includes: list[str] = []
    notes: list[str] = []
    stack: list[str] = []

    for lineno, raw_line in enumerate(_hocon_lines(text).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("include"):
            found = re.match(r"include\s+(?:required\s*\(\s*)?[\"']([^\"']+)[\"']", line)
            if found:
                includes.append(found.group(1))
            else:
                notes.append(f"{source}:{lineno}: unrecognised include: {line}")
            continue
        if line == "}":
            if stack:
                stack.pop()
            else:
                notes.append(f"{source}:{lineno}: unexpected '}}'")
            continue
        if line.endswith("{"):
            raw_key = line[:-1].strip().rstrip("=:").strip()
            key = _unquote(raw_key)
            if not key:
                notes.append(f"{source}:{lineno}: block without a key")
                continue
            stack.extend([key] if raw_key != key else key.split("."))
            continue
        assignment = _split_assignment(line)
        if assignment is None:
            notes.append(f"{source}:{lineno}: unparsable line: {line}")
            continue
        raw_key, raw_value = assignment
        key = _unquote(raw_key)
        segments = [key] if raw_key != key else key.split(".")
        path = ".".join([*stack, *segments])
        values[path] = HoconValue(path, raw_value, source, lineno)

    return values, includes, notes


def load_hocon(path: Path, depth: int = 0) -> HoconDocument:
    """Load a HOCON file and its plain includes, later values winning."""
    document = HoconDocument()
    if depth > 8:
        document.notes.append(f"{path}: includes nested too deeply")
        return document
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        document.notes.append(f"cannot read {path}: {exc}")
        return document

    document.sources.append(path)
    values, includes, notes = parse_hocon(text, path)
    document.values.update(values)
    document.notes.extend(notes)
    for target in includes:
        if "://" in target or target.startswith("classpath"):
            continue
        included = path.parent / target
        if not included.is_file():
            # A plain include of a missing file is legal HOCON and ignored.
            continue
        nested = load_hocon(included, depth + 1)
        document.values.update(nested.values)
        document.sources.extend(nested.sources)
        document.notes.extend(nested.notes)
    return document


def hocon_str(document: HoconDocument, path: str) -> str | None:
    value = document.values.get(path)
    return _unquote(value.raw) if value else None


def hocon_bool(document: HoconDocument, path: str) -> bool | None:
    value = hocon_str(document, path)
    if value is None:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return None


def hocon_duration(document: HoconDocument, path: str) -> float | None:
    """Parse a HOCON duration into seconds; bare numbers count as seconds."""
    value = hocon_str(document, path)
    if value is None:
        return None
    found = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", value)
    if not found:
        return None
    number = float(found.group(1))
    unit = found.group(2).lower()
    if unit in ("ms", "millisecond", "milliseconds"):
        return number / 1000
    if unit in ("m", "minute", "minutes"):
        return number * 60
    if unit in ("h", "hour", "hours"):
        return number * 3600
    return number


# --------------------------------------------------------------------------
# Prosody (Lua)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LuaBlock:
    kind: str
    name: str
    type: str | None
    body: str
    line: int


_LUA_BLOCK = re.compile(
    r"^[ \t]*(VirtualHost|Component)\s+[\"']([^\"']+)[\"']\s*(?:[\"']([^\"']+)[\"'])?",
    re.MULTILINE,
)


def lua_uncomment(text: str) -> str:
    """Blank Lua comments, preserving newlines and quoted strings."""
    out: list[str] = []
    index, length = 0, len(text)
    quote: str | None = None
    while index < length:
        char = text[index]
        if quote:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "-" and index + 1 < length and text[index + 1] == "-":
            long_open = re.match(r"\[(=*)\[", text[index + 2 :])
            if long_open:
                closer = "]" + long_open.group(1) + "]"
                end = text.find(closer, index + 2 + long_open.end())
                end = length if end < 0 else end + len(closer)
            else:
                end = text.find("\n", index)
                end = length if end < 0 else end
            out.append("".join("\n" if c == "\n" else " " for c in text[index:end]))
            index = end
            continue
        out.append(char)
        index += 1
    return "".join(out)


def find_lua_blocks(text: str) -> list[LuaBlock]:
    """Split a prosody config into its VirtualHost and Component blocks."""
    matches = list(_LUA_BLOCK.finditer(text))
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(
            LuaBlock(
                kind=match.group(1),
                name=match.group(2),
                type=match.group(3),
                body=text[match.end() : end],
                line=text.count("\n", 0, match.start()) + 1,
            )
        )
    return blocks


def lua_table(text: str, key: str) -> str | None:
    """Return the inner text of the last ``key = { ... }`` assignment.

    The last one wins because that is what Lua does, and Prosody configs are
    full of assignments an admin has replaced further down the file.
    """
    body: str | None = None
    for found in re.finditer(rf"(?<![\w.]){re.escape(key)}\s*=\s*\{{", text):
        end = match_brace(text, found.end() - 1)
        if end is not None:
            body = text[found.end() : end]
    return body


def lua_string_list(table: str | None) -> list[str]:
    """Every quoted string inside a Lua table body, in order."""
    if not table:
        return []
    return re.findall(r"[\"']([^\"']+)[\"']", table)


def lua_scalar(text: str, key: str) -> str | None:
    """The value of ``key = value``, unquoted, for strings and booleans."""
    found = re.search(
        rf"(?<![\w.]){re.escape(key)}\s*=\s*([\"'][^\"']*[\"']|true|false|[0-9.]+)", text
    )
    return _unquote(found.group(1)) if found else None


def lua_module_names(block_body: str) -> list[str]:
    return lua_string_list(lua_table(block_body, "modules_enabled"))


_ASYNC_TRANSCRIPTION = re.compile(
    r"(?:\basyncTranscription\b|\[\s*[\"']asyncTranscription[\"']\s*\])\s*=\s*(true|false)"
)


def lua_sets_async_transcription(text: str) -> bool:
    """Whether *text* assigns ``asyncTranscription = true`` (comments aside)."""
    return any(
        found.group(1) == "true"
        for found in _ASYNC_TRANSCRIPTION.finditer(lua_uncomment(text))
    )


# --------------------------------------------------------------------------
# URL template
# --------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")
_KNOWN_PLACEHOLDERS = ("MEETING_ID", "REGION")


@dataclass
class TemplateReport:
    raw: str
    resolved: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scheme: str = ""
    host: str = ""
    path: str = ""
    query: dict[str, list[str]] = field(default_factory=dict)


def analyze_template(raw: str, meeting_id: str, region: str = "") -> TemplateReport:
    """Resolve a Jicofo url-template the way Jicofo does, and inspect it."""
    resolved = raw.replace("{{MEETING_ID}}", meeting_id).replace("{{REGION}}", region)
    report = TemplateReport(raw=raw, resolved=resolved)

    for placeholder in _PLACEHOLDER.findall(raw):
        if placeholder not in _KNOWN_PLACEHOLDERS:
            report.errors.append(
                f"unknown placeholder {{{{{placeholder}}}}} survives into the URL, "
                "which Jicofo passes to java.net.URI"
            )
    if "{{MEETING_ID}}" not in raw:
        report.errors.append(
            "the template does not contain {{MEETING_ID}}, so every meeting "
            "resolves to the same URL"
        )
    if region == "" and "{{REGION}}" in raw and raw.split("{{REGION}}")[0].endswith("//"):
        report.warnings.append(
            "{{REGION}} is used in the host position and expands to an empty "
            "string for a bridge with no region, leaving a leading dot"
        )

    try:
        parts = urlsplit(resolved)
    except ValueError as exc:
        report.errors.append(f"the resolved URL cannot be parsed: {exc}")
        return report

    report.scheme, report.host, report.path = parts.scheme, parts.netloc, parts.path
    report.query = parse_qs(parts.query)

    if parts.scheme not in ("ws", "wss"):
        report.errors.append(f"the scheme is {parts.scheme or '(none)'!r}, not ws:// or wss://")
    if not parts.netloc:
        report.errors.append("the resolved URL has no host")
    if parts.path != WEBSOCKET_PATH:
        report.warnings.append(
            f"the path is {parts.path or '/'!r}; the bridge only serves {WEBSOCKET_PATH} "
            "(a reverse proxy may rewrite it — the probe confirms)"
        )
    session_values = report.query.get("sessionId")
    if not session_values:
        report.warnings.append(
            "the query has no sessionId, so every meeting records into the "
            "bridge's session_default directory"
        )
    elif meeting_id not in session_values[0]:
        report.warnings.append(
            f"sessionId is {session_values[0]!r}, which does not contain the meeting id"
        )
    return report


# --------------------------------------------------------------------------
# Deployment discovery and loading
# --------------------------------------------------------------------------


@dataclass
class Deployment:
    domain: str | None
    jicofo_conf: Path | None
    prosody_config: Path | None
    meet_config: Path | None
    jvb_conf: Path | None
    hocon: HoconDocument
    prosody_text: str | None
    meet_text: str | None
    notes: list[str] = field(default_factory=list)


def _read(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _meet_candidates(meet_dir: Path) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    if meet_dir.is_dir():
        for path in sorted(meet_dir.glob("*-config.js")):
            candidates[path.name[: -len("-config.js")]] = path
    return candidates


def _prosody_candidates(conf_avail: Path) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    if not conf_avail.is_dir():
        return candidates
    for path in sorted(conf_avail.glob("*.cfg.lua")):
        text = _read(path)
        if text is None:
            continue
        for block in find_lua_blocks(lua_uncomment(text)):
            if block.kind == "VirtualHost":
                candidates.setdefault(block.name, path)
    return candidates


def _pick_domain(args: argparse.Namespace, meet: dict[str, Path],
                 prosody: dict[str, Path]) -> str | None:
    if args.domain:
        return args.domain
    shared = sorted(set(meet) & set(prosody))
    if len(shared) == 1:
        return shared[0]
    if not shared and len(meet) == 1 and not prosody:
        return next(iter(meet))
    if not shared and len(prosody) == 1 and not meet:
        return next(iter(prosody))
    known = sorted(set(meet) | set(prosody))
    if not known:
        return None
    raise DiscoveryError(
        "cannot choose between these deployments: " + ", ".join(known) + "; pass --domain"
    )


def load_deployment(args: argparse.Namespace, require_files: bool = True) -> Deployment:
    """Find the deployment's files, read them, and note anything missing.

    *require_files* is false for runs that do not need the configuration (a
    probe given ``--url``, or a logs-only run), so they work on a host where
    Jitsi is not installed.
    """
    notes: list[str] = []

    jicofo_conf = Path(args.jicofo_conf) if args.jicofo_conf else DEFAULT_JICOFO_CONF
    if args.jicofo_conf and not jicofo_conf.is_file():
        raise DiscoveryError(f"--jicofo-conf: {jicofo_conf} does not exist")

    prosody_config: Path | None = None
    if args.prosody_config:
        prosody_config = Path(args.prosody_config)
        if not prosody_config.is_file():
            raise DiscoveryError(f"--prosody-config: {prosody_config} does not exist")

    meet_config: Path | None = Path(args.meet_config) if args.meet_config else None
    if args.meet_config and not meet_config.is_file():
        raise DiscoveryError(f"--meet-config: {meet_config} does not exist")

    jvb_conf: Path | None = Path(args.jvb_conf) if args.jvb_conf else DEFAULT_JVB_CONF
    if args.jvb_conf and not jvb_conf.is_file():
        raise DiscoveryError(f"--jvb-conf: {jvb_conf} does not exist")

    domain = args.domain
    if prosody_config is None or meet_config is None:
        meet = _meet_candidates(DEFAULT_MEET_DIR)
        prosody = _prosody_candidates(DEFAULT_PROSODY_CONF_AVAIL)
        domain = _pick_domain(args, meet, prosody)
        if domain is None and require_files and not args.jicofo_conf:
            raise DiscoveryError(
                "no Jitsi configuration found in "
                f"{DEFAULT_MEET_DIR} or {DEFAULT_PROSODY_CONF_AVAIL}; pass "
                "--meet-config/--prosody-config (or --domain) explicitly"
            )
        if prosody_config is None and domain:
            prosody_config = prosody.get(domain, DEFAULT_PROSODY_CONF_AVAIL / f"{domain}.cfg.lua")
            if not prosody_config.is_file():
                if DEFAULT_PROSODY_MAIN.is_file() and f'VirtualHost "{domain}"' in (
                    _read(DEFAULT_PROSODY_MAIN) or ""
                ):
                    prosody_config = DEFAULT_PROSODY_MAIN
                    notes.append("the site config is inline in " + str(DEFAULT_PROSODY_MAIN))
                else:
                    prosody_config = None
        if meet_config is None and domain:
            meet_config = meet.get(domain, DEFAULT_MEET_DIR / f"{domain}-config.js")
            if not meet_config.is_file():
                meet_config = None

    if not jicofo_conf.is_file():
        notes.append(f"no {jicofo_conf} (Jicofo may not be installed here)")
        jicofo_path: Path | None = None
    else:
        jicofo_path = jicofo_conf
    return Deployment(
        domain=domain,
        jicofo_conf=jicofo_path,
        prosody_config=prosody_config,
        meet_config=meet_config,
        jvb_conf=jvb_conf if jvb_conf.is_file() else None,
        hocon=load_hocon(jicofo_path) if jicofo_path else HoconDocument(),
        prosody_text=_read(prosody_config),
        meet_text=_read(meet_config),
        notes=notes,
    )


# --------------------------------------------------------------------------
# Config checks
# --------------------------------------------------------------------------


def check_jicofo(deployment: Deployment, meeting_id: str, custom_conf: Path) -> list[Check]:
    checks: list[Check] = []
    document = deployment.hocon

    if deployment.jicofo_conf is None:
        return [Check("jicofo.file", Status.SKIP, f"no {DEFAULT_JICOFO_CONF} to read")]
    checks.append(
        Check("jicofo.file", Status.PASS, f"read {deployment.jicofo_conf}",
              detail="\n".join(str(source) for source in document.sources))
    )

    raw_template = hocon_str(document, "jicofo.transcription.url-template")
    if not raw_template:
        checks.append(Check(
            "jicofo.url-template", Status.FAIL,
            "no jicofo.transcription.url-template is configured",
            fix=(
                "add to /etc/jitsi/jicofo/jicofo.conf (or custom-jicofo.conf):\n"
                'jicofo { transcription { url-template = '
                '"ws://<bridge-host>:<port>/transcribe?sessionId={{MEETING_ID}}" } }\n'
                "without it Jicofo logs \"Transcription enabled, but no URL is "
                "configured\" and never starts the transcriber (docs/jitsi-integration.md §3)"
            ),
        ))
    elif "${" in raw_template:
        checks.append(Check(
            "jicofo.url-template", Status.WARN,
            "the template is built from an environment substitution, which is "
            "resolved when Jicofo starts",
            detail=raw_template,
            fix="check the value the service actually receives, or run --only probe "
                "to exercise the resolved URL",
        ))
    else:
        report = analyze_template(raw_template, meeting_id)
        if report.errors:
            checks.append(Check(
                "jicofo.url-template", Status.FAIL, "; ".join(report.errors),
                detail=f"resolved: {report.resolved}",
                fix="see docs/jitsi-integration.md §3 for the template contract",
            ))
        else:
            checks.append(Check(
                "jicofo.url-template", Status.PASS,
                f"{raw_template}", detail=f"resolves to {report.resolved}",
            ))
        for warning in report.warnings:
            checks.append(Check(
                "jicofo.url-template.endpoint", Status.WARN, warning,
                fix="confirm this is deliberate, or align it with the bridge's route",
            ))

    if custom_conf.is_file() and custom_conf not in document.sources:
        text = _read(custom_conf) or ""
        if "transcription" in text:
            checks.append(Check(
                "jicofo.custom-conf", Status.FAIL,
                f"{custom_conf} defines transcription settings but is not included "
                "by jicofo.conf, so they have no effect",
                fix='add `include "custom-jicofo.conf"` to /etc/jitsi/jicofo/jicofo.conf '
                    "and restart jicofo",
            ))
        else:
            checks.append(Check(
                "jicofo.custom-conf", Status.WARN,
                f"{custom_conf} exists but is not included by jicofo.conf",
                fix="include it, or remove it if it is unused",
            ))

    ping_enabled = hocon_bool(document, "jicofo.transcription.ping.enabled")
    interval = hocon_duration(document, "jicofo.transcription.ping.interval")
    timeout = hocon_duration(document, "jicofo.transcription.ping.timeout")
    ping_state = "enabled" if ping_enabled is not False else "disabled"
    interval_text = f"{interval:g}" if interval is not None else "10"
    timeout_text = f"{timeout:g}" if timeout is not None else "3"
    values = (
        f"ping {ping_state}; interval {interval_text} s, timeout {timeout_text} s"
        + (" (defaults)" if ping_enabled is None and interval is None and timeout is None else "")
    )
    if timeout is not None and interval is not None and timeout >= interval:
        checks.append(Check(
            "jicofo.ping", Status.WARN, values,
            fix="a timeout at or above the interval marks pongs late; "
                "keep timeout < interval (the defaults are 10 s / 3 s)",
        ))
    else:
        checks.append(Check("jicofo.ping", Status.PASS, values))

    header_paths = sorted(
        path for path in document.values if path.startswith("jicofo.transcription.http-headers.")
    )
    if header_paths:
        names = ", ".join(path.rsplit(".", 1)[1] for path in header_paths)
        checks.append(Check(
            "jicofo.http-headers", Status.PASS,
            f"{len(header_paths)} header(s) set: {names} (values hidden)",
            detail="the probe sends these; a ${...} value is resolved at runtime and "
                   "cannot be checked here"
            if any("${" in (document.values[path].raw or "") for path in header_paths)
            else "",
        ))
    return checks


def check_prosody(
    deployment: Deployment,
    domain: str | None,
    plugin_dirs: Sequence[Path],
    read_module: Callable[[Path], str | None],
) -> list[Check]:
    checks: list[Check] = []
    text = deployment.prosody_text
    if text is None:
        return [Check(
            "prosody.file", Status.SKIP,
            f"no Prosody site config found for {domain or 'the discovered domain'}",
            fix="pass --prosody-config if the file is somewhere else",
        )]

    body = lua_uncomment(text)
    if DEFAULT_PROSODY_CONF_D.is_dir() and DEFAULT_PROSODY_MAIN.is_file():
        main_text = _read(DEFAULT_PROSODY_MAIN) or ""
        if (
            "conf.d" not in main_text
            and str(deployment.prosody_config) != str(DEFAULT_PROSODY_MAIN)
        ):
            checks.append(Check(
                "prosody.included", Status.WARN,
                f"{DEFAULT_PROSODY_MAIN} does not Include conf.d/*.cfg.lua, so "
                f"{deployment.prosody_config} may not be loaded",
                fix=f'add Include "conf.d/*.cfg.lua" to {DEFAULT_PROSODY_MAIN}',
            ))

    blocks = find_lua_blocks(body)
    virtual_hosts = [block for block in blocks if block.kind == "VirtualHost"]
    mucs = [block for block in blocks if block.kind == "Component" and block.type == "muc"]

    main_muc_name: str | None = None
    for host in virtual_hosts:
        if domain is None or host.name == domain:
            main_muc_name = lua_scalar(host.body, "main_muc") or main_muc_name
            if domain is not None and host.name == domain:
                break
    main_muc = next((block for block in mucs if block.name == main_muc_name), None)
    if main_muc is None:
        main_muc = next(
            (block for block in mucs if block.name == f"conference.{domain}"),
            mucs[0] if len(mucs) == 1 else None,
        )
    if main_muc is None:
        return [*checks, Check(
            "prosody.muc", Status.FAIL,
            "no MUC component found in the Prosody configuration",
            fix="check that the site config is the one Prosody loads",
        )]
    checks.append(Check("prosody.muc", Status.PASS, f"{main_muc.name} ({main_muc.type})"))

    modules = lua_module_names(main_muc.body)
    if "muc_meeting_id" in modules:
        checks.append(Check("prosody.muc_meeting_id", Status.PASS,
                            "enabled; {{MEETING_ID}} resolves to the room's meeting id"))
    else:
        checks.append(Check(
            "prosody.muc_meeting_id", Status.WARN,
            "muc_meeting_id is not enabled on the main MUC",
            fix='add "muc_meeting_id"; to modules_enabled; without it Jicofo generates '
                "a random meeting id of its own",
        ))

    has_module = "room_metadata" in modules or any(
        "room_metadata" in lua_module_names(host.body) for host in virtual_hosts
    )
    has_component = any(
        block.kind == "Component" and block.type == "room_metadata_component" for block in blocks
    )
    if has_module and has_component:
        checks.append(Check("prosody.room_metadata", Status.PASS,
                            "room_metadata and room_metadata_component are present"))
    elif has_component or has_module:
        checks.append(Check(
            "prosody.room_metadata", Status.WARN,
            "room metadata is only partially configured "
            f"(module: {has_module}, component: {has_component})",
            fix="compare with the stock site config; Jicofo reads the gating flags "
                "from the room metadata component",
        ))
    else:
        checks.append(Check(
            "prosody.room_metadata", Status.FAIL,
            "no room_metadata module or room_metadata_component component found",
            fix="Jicofo never sees asyncTranscription without them; restore the stock "
                "room_metadata entries in the site config",
        ))

    forcing: list[str] = []
    missing: list[str] = []
    for name in modules:
        path = next(
            (
                candidate
                for directory in plugin_dirs
                for candidate in (directory / f"mod_{name}.lua", directory / f"{name}.lua")
                if candidate.is_file()
            ),
            None,
        )
        if path is None:
            missing.append(name)
            continue
        if lua_sets_async_transcription(read_module(path) or ""):
            forcing.append(f"{name} ({path})")
    if forcing:
        checks.append(Check("prosody.force_async_transcription", Status.PASS,
                            "set by " + ", ".join(forcing)))
    else:
        available = [
            f"{path.name} ({directory})"
            for directory in plugin_dirs
            if directory.is_dir()
            for path in sorted(directory.glob("*.lua"))
            if lua_sets_async_transcription(read_module(path) or "")
        ]
        if available:
            checks.append(Check(
                "prosody.force_async_transcription", Status.WARN,
                "nothing enabled on the main MUC sets asyncTranscription, but "
                "an unenabled module does: " + ", ".join(available),
                fix='add that module name to modules_enabled on "'
                    f'{main_muc.name}"',
            ))
        else:
            checks.append(Check(
                "prosody.force_async_transcription", Status.FAIL,
                "no module sets asyncTranscription = true, so Jicofo will never "
                "start the transcriber",
                fix="install mod_force_async_transcription.lua in "
                    f"{plugin_dirs[0] if plugin_dirs else 'a plugin_paths directory'} and add "
                    f'"force_async_transcription"; to modules_enabled on "{main_muc.name}" '
                    "(docs/jitsi-integration.md §2)",
            ))
    if missing and not any(directory.is_dir() for directory in plugin_dirs):
        checks.append(Check(
            "prosody.modules.installed", Status.SKIP,
            "no plugin directory is readable here, so module files cannot be checked",
            fix="run the tool on the Jitsi host, or pass --plugin-dir",
        ))
    elif missing:
        checks.append(Check(
            "prosody.modules.installed", Status.WARN,
            "enabled module(s) have no file in the plugin paths: " + ", ".join(sorted(missing)),
            fix="check plugin_paths and that the module files are installed",
        ))
    return checks


def check_meet_config(deployment: Deployment) -> list[Check]:
    text = deployment.meet_text
    if text is None:
        return [Check(
            "meet.file", Status.SKIP,
            f"no client config found for {deployment.domain or 'the discovered domain'}",
            fix="pass --meet-config if the file is somewhere else",
        )]
    checks = [Check("meet.file", Status.PASS, f"read {deployment.meet_config}")]

    stripped = strip_js_comments(text)
    body = find_js_object(stripped, "transcription")
    if body is None:
        commented = find_js_object(text, "transcription") is not None
        checks.append(Check(
            "meet.transcription.enabled", Status.FAIL,
            "the transcription block is still commented out"
            if commented
            else "there is no transcription object in the client config",
            fix='add `transcription: { enabled: true },` to '
                f"{deployment.meet_config} and reload the clients; without it no client "
                "can set recording.isTranscribingEnabled (docs/jitsi-integration.md §1)",
        ))
    elif js_boolean(body, "enabled") is True:
        checks.append(Check("meet.transcription.enabled", Status.PASS, "enabled: true"))
    else:
        value = js_boolean(body, "enabled")
        checks.append(Check(
            "meet.transcription.enabled", Status.FAIL,
            f"transcription.enabled is {'false' if value is False else 'absent'}",
            fix="set `transcription: { enabled: true },` in the client config",
        ))
    return checks


def check_jvb(deployment: Deployment) -> list[Check]:
    if deployment.jvb_conf is None:
        return [Check("jvb.exporter", Status.SKIP, f"no {DEFAULT_JVB_CONF}")]
    document = load_hocon(deployment.jvb_conf)
    attempts = hocon_duration(document, "videobridge.exporter.max-reconnect-attempts")
    if attempts is not None and attempts <= 0:
        return [Check(
            "jvb.exporter", Status.WARN,
            f"videobridge.exporter.max-reconnect-attempts is {attempts:g}; "
            "the JVB will not retry a dropped connection",
            fix="remove the key (unlimited) or raise it",
        )]
    return [Check("jvb.exporter", Status.PASS, "no problematic exporter overrides "
                                               "(the JVB needs no configuration)")]


# --------------------------------------------------------------------------
# Live probe
# --------------------------------------------------------------------------


async def _await_pong(connection: Any, ping_id: int, timeout: float) -> tuple[bool, int]:
    """Wait for the pong with *ping_id*, counting anything else received."""
    deadline = time.monotonic() + timeout
    ignored = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, ignored
        message = await asyncio.wait_for(connection.recv(), remaining)
        try:
            event = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            ignored += 1
            continue
        if isinstance(event, dict) and event.get("event") == "pong" and event.get("id") == ping_id:
            return True, ignored
        ignored += 1


def classify_probe_error(exc: BaseException, target_uri: str) -> Check:
    """Turn a connection failure into one FAIL with the likely cause."""
    if isinstance(exc, socket.gaierror):
        summary = f"the host in {target_uri} cannot be resolved"
        fix = "check DNS from this host, or run the probe on the JVB host"
    elif isinstance(exc, ConnectionRefusedError):
        summary = f"nothing is listening at {target_uri}"
        fix = "is the bridge running? is the port right? a firewall may be dropping it"
    elif isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        summary = f"the connection to {target_uri} timed out"
        fix = "a firewall is likely dropping the packets; test from the JVB host"
    elif isinstance(exc, ssl.SSLError):
        summary = f"TLS failed for {target_uri}: {exc}"
        fix = "use ws:// on a trusted network, or fix the certificate"
    else:
        websockets = sys.modules.get("websockets")
        invalid_status = getattr(websockets, "InvalidStatus", None)
        invalid_handshake = getattr(websockets, "InvalidHandshake", None)
        if invalid_status is not None and isinstance(exc, invalid_status):
            status = getattr(getattr(exc, "response", None), "status_code", "?")
            summary = f"the server refused the WebSocket upgrade with HTTP {status}"
            fix = "check the path, and any proxy in front of the bridge"
        elif invalid_handshake is not None and isinstance(exc, invalid_handshake):
            summary = f"the WebSocket handshake failed: {exc}"
            fix = "confirm the URL points at the bridge's /transcribe route"
        else:
            summary = f"{type(exc).__name__}: {exc}"
            fix = "run the probe on the host that runs the JVB if it is a network issue"
    return Check("probe.connect", Status.FAIL, summary, fix=fix)


def _connect_kwargs(websockets_module: Any, headers: dict[str, str]) -> dict[str, Any]:
    import inspect

    if not headers:
        return {}
    parameters = inspect.signature(websockets_module.connect).parameters
    name = "additional_headers" if "additional_headers" in parameters else "extra_headers"
    return {name: headers}


def classify_close(exc: BaseException, uri: str) -> Check:
    """Turn an unexpected close during the probe into a FAIL or WARN."""
    code = getattr(getattr(exc, "rcvd", None), "code", None)
    if code == 1008:
        return Check(
            "probe.path", Status.FAIL,
            f"the bridge rejected the path with 1008 (unsupported path) for {uri}",
            fix="the URL must target /transcribe; fix the Jicofo url-template",
        )
    return Check(
        "probe.connection", Status.FAIL,
        f"the connection closed (code {code}) during the probe",
        fix="check the bridge's log for the reason",
    )


async def run_probe(uri: str, headers: dict[str, str], ping_enabled: bool,
                    ping_timeout: float, connect_timeout: float) -> list[Check]:
    """Drive one probe session and return its checks."""
    import websockets

    from tools.send_meeting import media_json_info, media_json_ping, media_json_session_end

    checks: list[Check] = []
    try:
        connection = await websockets.connect(
            uri,
            open_timeout=connect_timeout,
            ping_interval=None,
            close_timeout=3,
            **_connect_kwargs(websockets, headers),
        )
    except Exception as exc:  # noqa: BLE001 - classified into a Check
        return [classify_probe_error(exc, uri)]

    checks.append(Check(
        "probe.connect", Status.PASS, f"WebSocket handshake completed with {uri}",
        detail=f"{len(headers)} configured header(s) applied" if headers else "",
    ))
    closed_early = False
    try:
        async with connection:
            await connection.send(json.dumps(media_json_info()))
            if not ping_enabled:
                checks.append(Check(
                    "probe.pong", Status.SKIP,
                    "jicofo.transcription.ping.enabled is false, so no ping was sent",
                ))
            else:
                ping_id = 1
                await connection.send(json.dumps(media_json_ping(ping_id)))
                try:
                    answered, ignored = await _await_pong(connection, ping_id,
                                                          ping_timeout + 2)
                except TimeoutError:
                    answered, ignored = False, 0
                except websockets.ConnectionClosed as exc:
                    closed_early = True
                    checks.append(classify_close(exc, uri))
                if not closed_early:
                    if answered:
                        checks.append(Check(
                            "probe.pong", Status.PASS, "the bridge answered the ping",
                            detail=f"{ignored} other frame(s) received first" if ignored else "",
                        ))
                    else:
                        checks.append(Check(
                            "probe.pong", Status.FAIL,
                            "no pong arrived within the configured timeout",
                            fix="the endpoint is not answering media-json pings; the JVB "
                                "would drop and re-establish the connection",
                        ))
            if not closed_early:
                await connection.send(json.dumps(media_json_session_end()))
                checks.append(Check("probe.session-end", Status.PASS,
                                    "session-end sent; no audio was transmitted"))
    except websockets.ConnectionClosed as exc:
        checks.append(classify_close(exc, uri))
    except Exception as exc:  # noqa: BLE001 - reported, never a traceback
        checks.append(Check(
            "probe.session-end", Status.FAIL,
            f"{type(exc).__name__}: {exc}",
            fix="check the bridge's log for the reason",
        ))
    return checks


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------


@dataclass
class LogReport:
    source: str
    lines: int = 0
    positives: dict[str, int] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    normal_closes: int = 0
    runtime_urls: list[str] = field(default_factory=list)


def classify_jvb(lines: Iterable[str], source: str = "") -> LogReport:
    report = LogReport(source=source)
    for line in lines:
        report.lines += 1
        for pattern in JVB_POSITIVE_PATTERNS:
            if pattern in line:
                report.positives[pattern] = report.positives.get(pattern, 0) + 1
        for pattern in JVB_FAILURE_PATTERNS:
            if pattern in line:
                report.failures[pattern] = report.failures.get(pattern, 0) + 1
        if "Websocket closed with status " in line:
            if "status 1000" in line:
                report.normal_closes += 1
            else:
                report.failures["Websocket closed with status"] = (
                    report.failures.get("Websocket closed with status", 0) + 1
                )
        found = _RUNTIME_URL.search(line)
        if found:
            report.runtime_urls.append(found.group(1))
    return report


def classify_jicofo(lines: Iterable[str], source: str = "") -> LogReport:
    report = LogReport(source=source)
    for line in lines:
        report.lines += 1
        if JICOFO_ERROR_PATTERN in line:
            report.failures[JICOFO_ERROR_PATTERN] = (
                report.failures.get(JICOFO_ERROR_PATTERN, 0) + 1
            )
        if JICOFO_WARNING_PATTERN in line:
            report.failures[JICOFO_WARNING_PATTERN] = (
                report.failures.get(JICOFO_WARNING_PATTERN, 0) + 1
            )
    return report


def read_journal(unit: str, since: str, timeout: float = 30) -> tuple[list[str] | None, str]:
    """Read a unit's journal; returns (lines, error)."""
    if shutil.which("journalctl") is None:
        return None, "journalctl is not installed"
    try:
        result = subprocess.run(
            ["journalctl", "--no-pager", "-o", "short-iso", "-u", unit, "--since", since],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"journalctl failed: {exc}"
    if result.returncode != 0:
        message = (result.stderr or "").strip().splitlines()
        return None, message[-1] if message else f"journalctl exited {result.returncode}"
    lines = result.stdout.splitlines()
    if not lines:
        return None, f"no journal entries for {unit} since {since}"
    return lines, ""


def read_log_file(path: Path) -> tuple[list[str] | None, str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines(), ""
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"


def _log_checks(name: str, report: LogReport, *, error_pattern: str | None = None,
                warning_pattern: str | None = None,
                configured: str | None = None) -> list[Check]:
    checks: list[Check] = []
    problem = report.failures.get(error_pattern) if error_pattern else None
    if problem:
        checks.append(Check(
            f"{name}.problem", Status.FAIL,
            f"{error_pattern!r} appears {problem} time(s) in the window",
            fix="the running process does not have the configured transcriber URL; "
                "restart Jicofo after editing the config",
        ))
    elif warning_pattern and report.failures.get(warning_pattern):
        checks.append(Check(
            f"{name}.problem", Status.WARN,
            f"{warning_pattern!r} appears in the window",
            fix="the template needs {{MEETING_ID}}; see docs/jitsi-integration.md §3",
        ))
    elif report.failures:
        checks.append(Check(
            f"{name}.problem", Status.FAIL,
            "; ".join(f"{pattern!r} x{count}" for pattern, count in report.failures.items()),
            fix="these are the JVB's exporter failure modes; check the bridge's "
                "reachability and its log",
        ))
    else:
        checks.append(Check(f"{name}.problem", Status.PASS,
                            "no transcription errors in the window"))
        return checks

    if report.runtime_urls and configured:
        runtime = report.runtime_urls[-1]
        runtime_parts, configured_parts = urlsplit(runtime), urlsplit(configured)
        if (runtime_parts.netloc, runtime_parts.path) != (
            configured_parts.netloc, configured_parts.path
        ):
            checks.append(Check(
                f"{name}.runtime-url", Status.WARN,
                f"the running process connects to {runtime_parts.netloc}{runtime_parts.path}, "
                "not the configured target",
                fix="the process was started before the config was edited; restart it",
            ))
    return checks


# --------------------------------------------------------------------------
# Sections, rendering, CLI
# --------------------------------------------------------------------------


def plugin_dirs_for(args: argparse.Namespace, prosody_text: str | None) -> list[Path]:
    """Where Prosody looks for modules: --plugin-dir, plugin_paths, defaults."""
    candidates = [Path(path) for path in (args.plugin_dir or [])]
    if prosody_text:
        paths = lua_table(lua_uncomment(prosody_text), "plugin_paths")
        candidates.extend(Path(path) for path in lua_string_list(paths))
    candidates.extend(DEFAULT_PLUGIN_DIRS)
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


def run_config_section(deployment: Deployment, args: argparse.Namespace) -> Section:
    section = Section("config")
    plugin_dirs = plugin_dirs_for(args, deployment.prosody_text)
    # A relative include resolves against the including file, so that is where
    # the custom file an admin would edit lives.
    custom_conf = (
        deployment.jicofo_conf.parent / "custom-jicofo.conf"
        if deployment.jicofo_conf
        else DEFAULT_CUSTOM_JICOFO_CONF
    )
    section.checks.extend(check_jicofo(deployment, args.session_id, custom_conf))
    section.checks.extend(
        check_prosody(deployment, deployment.domain, plugin_dirs, _read)
    )
    section.checks.extend(check_meet_config(deployment))
    section.checks.extend(check_jvb(deployment))
    return section


def run_probe_section(deployment: Deployment, args: argparse.Namespace) -> Section:
    section = Section("probe")
    uri = args.url
    ping_enabled = True
    ping_timeout = args.ping_timeout or 3.0
    if uri is None:
        template = hocon_str(deployment.hocon, "jicofo.transcription.url-template")
        if not template:
            section.checks.append(Check(
                "probe.target", Status.SKIP,
                "no url-template to probe; pass --url to probe an endpoint anyway",
            ))
            return section
        report = analyze_template(template, args.session_id, args.region)
        uri = report.resolved
        ping_enabled = (
            hocon_bool(deployment.hocon, "jicofo.transcription.ping.enabled") is not False
        )
        configured = hocon_duration(deployment.hocon, "jicofo.transcription.ping.timeout")
        if args.ping_timeout is None and configured is not None:
            ping_timeout = configured
    headers = {
        path.rsplit(".", 1)[1]: _unquote(value.raw)
        for path, value in deployment.hocon.values.items()
        if path.startswith("jicofo.transcription.http-headers.")
    }
    section.checks.append(Check(
        "probe.target", Status.PASS, uri,
        detail="resolved from jicofo.transcription.url-template" if args.url is None
        else "from --url",
    ))
    section.checks.extend(
        asyncio.run(run_probe(uri, headers, ping_enabled, ping_timeout, args.timeout))
    )
    return section


def run_logs_section(deployment: Deployment, args: argparse.Namespace) -> Section:
    section = Section("logs")
    configured = hocon_str(deployment.hocon, "jicofo.transcription.url-template")

    for name, unit, log_path, classify, error, warning in (
        ("logs.jvb", args.jvb_unit, args.jvb_log, classify_jvb, None, None),
        ("logs.jicofo", args.jicofo_unit, args.jicofo_log, classify_jicofo,
         JICOFO_ERROR_PATTERN, JICOFO_WARNING_PATTERN),
    ):
        if log_path:
            lines, failure = read_log_file(Path(log_path))
            source = str(log_path)
        else:
            lines, failure = read_journal(unit, args.since)
            source = f"{unit} (journalctl)"
        if lines is None:
            section.checks.append(Check(
                f"{name}.source", Status.SKIP, failure,
                fix="run as root or with the systemd-journal group, or pass a log file",
            ))
            continue
        report = classify(lines, source)
        section.checks.append(Check(
            f"{name}.source", Status.PASS,
            f"{report.lines} line(s) from {source}",
        ))
        if name == "logs.jvb":
            if report.failures:
                section.checks.append(Check(
                    "logs.jvb.connection", Status.FAIL,
                    "; ".join(f"{p!r} x{c}" for p, c in report.failures.items()),
                    fix="the JVB is connecting but failing; most often the bridge is "
                        "unreachable from the JVB host or is not answering pings",
                ))
            elif report.positives:
                section.checks.append(Check(
                    "logs.jvb.connection", Status.PASS,
                    "the exporter connected: " + ", ".join(
                        f"{p!r} x{c}" for p, c in report.positives.items()
                    ),
                    detail=f"{report.normal_closes} normal close(s)",
                ))
            else:
                section.checks.append(Check(
                    "logs.jvb.connection", Status.WARN,
                    "no exporter activity in the window (absence is not evidence; "
                    "make a test call with transcription on)",
                    fix="start a meeting, enable transcription, then rerun with "
                        "--since '5 min ago'",
                ))
            if report.runtime_urls and configured:
                runtime = urlsplit(report.runtime_urls[-1])
                wanted = urlsplit(analyze_template(configured, args.session_id).resolved)
                if (runtime.netloc, runtime.path) != (wanted.netloc, wanted.path):
                    section.checks.append(Check(
                        "logs.jvb.runtime-url", Status.WARN,
                        f"the JVB connects to {runtime.netloc}{runtime.path}, not the "
                        "configured target",
                        fix="restart Jicofo and the JVB so the edit takes effect",
                    ))
        else:
            section.checks.extend(
                _log_checks(name, report, error_pattern=error, warning_pattern=warning,
                            configured=configured)
            )
    return section


def render(sections: Sequence[Section]) -> None:
    for section in sections:
        print(f"\n{section.name}")
        for check in section.checks:
            print(f"  [{check.status.value}] {check.id} — {check.summary}")
            if check.detail:
                for line in check.detail.splitlines():
                    print(f"          {line}")
            if check.fix and check.status in (Status.FAIL, Status.WARN):
                for index, line in enumerate(check.fix.splitlines()):
                    prefix = "          fix: " if index == 0 else "               "
                    print(f"{prefix}{line}")
        counts: dict[Status, int] = dict.fromkeys(Status, 0)
        for check in section.checks:
            counts[check.status] += 1
        print(
            f"  {counts[Status.PASS]} passed, {counts[Status.FAIL]} failed, "
            f"{counts[Status.WARN]} warning(s), {counts[Status.SKIP]} skipped"
        )


def exit_code(sections: Sequence[Section]) -> int:
    if any(check.status is Status.FAIL for section in sections for check in section.checks):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m tools.verify_jitsi",
        description="Check a Jitsi deployment against docs/jitsi-integration.md.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=["config", "probe", "logs", "all"],
        help="section(s) to run; repeatable (default: config)",
    )
    parser.add_argument("--domain", help="the Jitsi domain, when several are configured")
    parser.add_argument("--jicofo-conf", metavar="PATH")
    parser.add_argument("--prosody-config", metavar="PATH")
    parser.add_argument("--meet-config", metavar="PATH")
    parser.add_argument("--jvb-conf", metavar="PATH")
    parser.add_argument(
        "--plugin-dir", action="append", metavar="PATH",
        help="Prosody plugin_paths directory; repeatable",
    )
    parser.add_argument("--url", help="probe this endpoint instead of the configured template")
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID,
                        help="meeting id used to resolve the template (default: %(default)s)")
    parser.add_argument("--region", default="", help="value for {{REGION}} when resolving")
    parser.add_argument("--timeout", type=float, default=10.0, help="connect timeout (s)")
    parser.add_argument("--ping-timeout", type=float, default=None,
                        help="pong wait (s); defaults to the configured ping timeout")
    parser.add_argument("--since", default="30 min ago", help="journalctl --since window")
    parser.add_argument("--jvb-unit", default="jitsi-videobridge2")
    parser.add_argument("--jicofo-unit", default="jicofo")
    parser.add_argument("--jvb-log", metavar="PATH", help="read a log file instead of journald")
    parser.add_argument("--jicofo-log", metavar="PATH")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sections_wanted = args.only or ["config"]
    if "all" in sections_wanted:
        sections_wanted = ["config", "probe", "logs"]

    try:
        deployment = load_deployment(args, require_files="config" in sections_wanted)
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("jitsi-audio-bridge deployment check")
    print(f"  domain    {deployment.domain or '(not determined)'}")
    for label, path in (
        ("jicofo", deployment.jicofo_conf),
        ("prosody", deployment.prosody_config),
        ("client", deployment.meet_config),
        ("jvb", deployment.jvb_conf),
    ):
        print(f"  {label:<9} {path or '(not found)'}")
    for note in deployment.notes:
        print(f"  note: {note}")

    sections: list[Section] = []
    if "config" in sections_wanted:
        sections.append(run_config_section(deployment, args))
    if "probe" in sections_wanted:
        sections.append(run_probe_section(deployment, args))
    if "logs" in sections_wanted:
        sections.append(run_logs_section(deployment, args))

    render(sections)

    failures = [
        check for section in sections for check in section.checks if check.status is Status.FAIL
    ]
    if failures:
        print("\nnext steps:")
        for index, check in enumerate(failures, start=1):
            print(f"  {index}. [{check.id}] {check.summary}")
            if check.fix:
                print(f"     {check.fix.splitlines()[0]}")
        print(f"\n{len(failures)} check(s) failed")
    else:
        print("\nno failures")
    if "probe" in sections_wanted:
        print("note: the probe proves reachability from this host; run it where the JVB "
              "runs if they differ")
    return exit_code(sections)


if __name__ == "__main__":  # pragma: no cover
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
