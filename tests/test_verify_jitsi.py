"""Unit tests for tools/verify_jitsi.py.

Everything here is offline: the parsers run over fixture strings and the check
functions over parsed fixtures.  The live probe is exercised end to end by
tests/smoke_test.py, which is the only place a real socket is opened.

The two fixtures that matter most are the traps the tool exists to catch: a
``url-template`` that is only present commented out in HOCON, and the shipped
``config.js`` block that is commented out with ``//``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from tools import verify_jitsi
from tools.verify_jitsi import (
    PROSODY_MODULE_LUA,
    Check,
    Deployment,
    DiscoveryError,
    HoconDocument,
    Section,
    Status,
    add_js_property,
    add_lua_modules,
    analyze_template,
    append_to_file_text,
    check_jicofo,
    check_meet_config,
    check_prosody,
    classify_jicofo,
    classify_jvb,
    exit_code,
    find_js_object,
    find_js_var_object_span,
    find_lua_blocks,
    hocon_bool,
    hocon_duration,
    hocon_str,
    hocon_transcription_block,
    js_boolean,
    load_deployment,
    load_hocon,
    lua_module_names,
    lua_sets_async_transcription,
    lua_string_list,
    lua_table,
    lua_table_span,
    lua_uncomment,
    normalize_bridge_url,
    parse_hocon,
    propose_jicofo_fix,
    propose_meet_fix,
    propose_prosody_fixes,
    strip_js_comments,
    write_proposal,
)

# --------------------------------------------------------------------------
# JavaScript comments and the client config
# --------------------------------------------------------------------------

COMMENTED_CLIENT = """var config = {
    hosts: { domain: 'meet.example.com' },
    // transcription: {
    //     // Whether the feature should be enabled or not.
    //     enabled: false,
    // },
    url: 'https://meet.example.com/x',
};
"""

LIVE_CLIENT = """var config = {
    hosts: { domain: 'meet.example.com' },
    transcription: {
        enabled: true,
    },
    url: 'https://meet.example.com/x',
};
"""


def test_strip_js_comments_keeps_urls_and_line_count() -> None:
    stripped = strip_js_comments("var u = 'https://x/y'; // note\n/* block */\n")
    assert "'https://x/y'" in stripped
    assert "note" not in stripped
    assert "block" not in stripped
    assert stripped.count("\n") == 2


def test_commented_block_is_invisible_after_stripping() -> None:
    assert find_js_object(strip_js_comments(COMMENTED_CLIENT), "transcription") is None
    # …but still findable in the raw text, which is how the tool knows to say
    # "still commented out" rather than "missing".
    assert find_js_object(COMMENTED_CLIENT, "transcription") is not None


def test_live_transcription_object_is_found() -> None:
    body = find_js_object(strip_js_comments(LIVE_CLIENT), "transcription")
    assert body is not None
    assert js_boolean(body, "enabled") is True


def test_js_boolean_does_not_leak_from_another_object() -> None:
    text = "var c = { recording: { enabled: true }, transcription: { enabled: false } };"
    body = find_js_object(strip_js_comments(text), "transcription")
    assert body is not None
    assert js_boolean(body, "enabled") is False


def test_js_boolean_is_none_when_absent() -> None:
    assert js_boolean("var x = 1;", "enabled") is None


# --------------------------------------------------------------------------
# HOCON
# --------------------------------------------------------------------------

HOCON = """
jicofo {
  // url-template = "ws://commented.example.com/transcribe"
  # url-template = "ws://hash.example.com/transcribe"
  transcription {
    url-template = "ws://bridge.example.com:8080/transcribe?sessionId={{MEETING_ID}}"
    http-headers = { "Authorization" = "Bearer a//b#c" }
    ping { enabled = true
           interval = 10 seconds
           timeout = 300 ms }
  }
}
"""


def _document(text: str, source: Path = Path("jicofo.conf")) -> HoconDocument:
    values, includes, notes = parse_hocon(text, source)
    assert not notes, notes
    return HoconDocument(values=values, sources=[source])


def test_hocon_nested_blocks_and_commented_traps() -> None:
    document = _document(HOCON)
    template = hocon_str(document, "jicofo.transcription.url-template")
    assert template == "ws://bridge.example.com:8080/transcribe?sessionId={{MEETING_ID}}"
    # The two commented examples must not have produced values of their own.
    assert "commented.example.com" not in str(document.values)
    assert "hash.example.com" not in str(document.values)


def test_hocon_quoted_values_keep_comment_characters() -> None:
    assert hocon_str(_document(HOCON), "jicofo.transcription.http-headers.Authorization") == (
        "Bearer a//b#c"
    )


def test_hocon_booleans_and_durations() -> None:
    document = _document(HOCON)
    assert hocon_bool(document, "jicofo.transcription.ping.enabled") is True
    assert hocon_duration(document, "jicofo.transcription.ping.interval") == 10.0
    assert hocon_duration(document, "jicofo.transcription.ping.timeout") == 0.3


def test_hocon_dotted_and_one_line_forms() -> None:
    dotted = _document('jicofo.transcription.url-template = "ws://x/transcribe"')
    assert hocon_str(dotted, "jicofo.transcription.url-template") == "ws://x/transcribe"
    one_line = _document('jicofo { transcription { url-template = "ws://x/transcribe" } }')
    assert hocon_str(one_line, "jicofo.transcription.url-template") == "ws://x/transcribe"
    colon = _document('jicofo { transcription { url-template: "ws://x/transcribe" } }')
    assert hocon_str(colon, "jicofo.transcription.url-template") == "ws://x/transcribe"


def test_hocon_never_raises_on_an_unterminated_block() -> None:
    document = _document('jicofo { transcription { url-template = "ws://x/t"')
    assert hocon_str(document, "jicofo.transcription.url-template") == "ws://x/t"


def test_hocon_missing_values_are_none() -> None:
    document = _document("jicofo { conference { x = 1 } }")
    assert hocon_str(document, "jicofo.transcription.url-template") is None
    assert hocon_duration(document, "jicofo.transcription.ping.interval") is None


def test_hocon_include_chain_later_wins_and_missing_includes_are_ignored(
    tmp_path: Path,
) -> None:
    (tmp_path / "custom-jicofo.conf").write_text(
        'jicofo.transcription.url-template = "ws://custom/transcribe"\n'
    )
    main = tmp_path / "jicofo.conf"
    main.write_text(
        'jicofo.transcription.url-template = "ws://stock/transcribe"\n'
        "include \"custom-jicofo.conf\"\n"
        'include "not-installed.conf"\n'
    )
    document = verify_jitsi.load_hocon(main)
    assert hocon_str(document, "jicofo.transcription.url-template") == "ws://custom/transcribe"
    assert main in document.sources and (tmp_path / "custom-jicofo.conf") in document.sources


# --------------------------------------------------------------------------
# Prosody
# --------------------------------------------------------------------------

PROSODY = """
plugin_paths = { "/usr/share/jitsi-meet/prosody-plugins/" }

VirtualHost "meet.example.com"
    modules_enabled = {
        "room_metadata";
        "features_identity";
    }

Component "conference.meet.example.com" "muc"
    main_muc = "conference.meet.example.com"
    modules_enabled = {
        "muc_meeting_id";
        "force_async_transcription";
        -- "token_verification";
    }

Component "metadata.meet.example.com" "room_metadata_component"
    muc_component = "conference.meet.example.com"
"""

PROSODY_WITHOUT_FORCE = PROSODY.replace('        "force_async_transcription";\n', "")
PROSODY_WITHOUT_IDENTITY = PROSODY.replace('        "features_identity";\n', "")

FORCE_MODULE = """
local util = module:require 'util';
module:hook('muc-room-created', function(event)
    local room = event.room;
    if not room.jitsiMetadata then
        room.jitsiMetadata = {};
    end
    room.jitsiMetadata.asyncTranscription = true;
end, -2);
"""


def test_lua_blocks_are_found_with_their_types() -> None:
    blocks = find_lua_blocks(lua_uncomment(PROSODY))
    assert [(b.kind, b.name, b.type) for b in blocks] == [
        ("VirtualHost", "meet.example.com", None),
        ("Component", "conference.meet.example.com", "muc"),
        ("Component", "metadata.meet.example.com", "room_metadata_component"),
    ]


def test_lua_module_names_skip_commented_entries() -> None:
    muc = next(b for b in find_lua_blocks(lua_uncomment(PROSODY)) if b.type == "muc")
    assert lua_module_names(muc.body) == ["muc_meeting_id", "force_async_transcription"]


def test_lua_table_last_assignment_wins() -> None:
    text = 'plugin_paths = { "/first" }\nplugin_paths = { "/second" }\n'
    assert lua_table(text, "plugin_paths") is not None
    assert "/second" in lua_table(text, "plugin_paths")
    assert "/first" not in lua_table(text, "plugin_paths")


def test_lua_uncomment_handles_long_comments() -> None:
    text = 'x = 1\n--[[\nasyncTranscription = true\n]]\ny = 2\n'
    stripped = lua_uncomment(text)
    assert "asyncTranscription" not in stripped
    assert "y = 2" in stripped


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("room.jitsiMetadata.asyncTranscription = true", True),
        ('room["jitsiMetadata"]["asyncTranscription"] = true', True),
        ("room.jitsiMetadata.asyncTranscription = false", False),
        ("-- room.jitsiMetadata.asyncTranscription = true", False),
        ("x = 1", False),
    ],
)
def test_lua_sets_async_transcription(text: str, expected: bool) -> None:
    assert lua_sets_async_transcription(text) is expected


# --------------------------------------------------------------------------
# URL template
# --------------------------------------------------------------------------


def test_template_passes_when_it_matches_the_contract() -> None:
    report = analyze_template(
        "ws://bridge.example.com:8080/transcribe?sessionId={{MEETING_ID}}", "meeting-1"
    )
    assert not report.errors and not report.warnings
    assert report.resolved.endswith("sessionId=meeting-1")
    assert (report.scheme, report.host, report.path) == (
        "ws", "bridge.example.com:8080", "/transcribe",
    )


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("ws://bridge/transcribe?m={{MEETING_ID}}", "sessionId"),
        ("ws://bridge/transcribe?sessionId=static&m={{MEETING_ID}}", "sessionId"),
        ("ws://bridge/other?sessionId={{MEETING_ID}}", "path"),
        ("http://bridge/transcribe?sessionId={{MEETING_ID}}", None),
        ("ws://bridge/transcribe?sessionId={{MEETING_ID}}&x={{ROOM}}", None),
    ],
)
def test_template_verdicts(template: str, expected: str | None) -> None:
    report = analyze_template(template, "meeting-1")
    if expected is None:
        assert report.errors
    else:
        assert not report.errors
        assert any(expected in warning for warning in report.warnings)


def test_template_requires_the_meeting_id_placeholder() -> None:
    report = analyze_template("ws://bridge/transcribe?sessionId=x", "meeting-1")
    assert any("MEETING_ID" in error for error in report.errors)


# --------------------------------------------------------------------------
# Check policy
# --------------------------------------------------------------------------


def _deployment(*, prosody_text: str | None = None, meet_text: str | None = None,
                hocon: HoconDocument | None = None,
                jicofo_conf: Path | None = Path("jicofo.conf")) -> Deployment:
    return Deployment(
        domain="meet.example.com",
        jicofo_conf=jicofo_conf,
        prosody_config=Path("meet.example.com.cfg.lua"),
        meet_config=Path("meet.example.com-config.js"),
        jvb_conf=None,
        hocon=hocon or HoconDocument(),
        prosody_text=prosody_text,
        meet_text=meet_text,
    )


def _status(checks: list[Check], check_id: str) -> Status:
    return next(check.status for check in checks if check.id == check_id)


def test_meet_config_commented_block_is_a_failure_that_says_so() -> None:
    checks = check_meet_config(_deployment(meet_text=COMMENTED_CLIENT))
    check = next(c for c in checks if c.id == "meet.transcription.enabled")
    assert check.status is Status.FAIL
    assert "commented out" in check.summary


def test_meet_config_live_true_passes_and_false_fails() -> None:
    assert _status(
        check_meet_config(_deployment(meet_text=LIVE_CLIENT)), "meet.transcription.enabled"
    ) is Status.PASS
    disabled = LIVE_CLIENT.replace("enabled: true", "enabled: false")
    assert _status(
        check_meet_config(_deployment(meet_text=disabled)), "meet.transcription.enabled"
    ) is Status.FAIL


def test_prosody_accepts_room_metadata_on_the_virtual_host(tmp_path: Path) -> None:
    checks = check_prosody(_deployment(prosody_text=PROSODY), "meet.example.com", [], lambda p: "")
    assert _status(checks, "prosody.room_metadata") is Status.PASS


def test_prosody_force_module_found_and_missing(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "mod_force_async_transcription.lua").write_text(FORCE_MODULE)
    reader = lambda path: path.read_text()  # noqa: E731 - a one-line test reader

    checks = check_prosody(
        _deployment(prosody_text=PROSODY), "meet.example.com", [plugin_dir], reader
    )
    assert _status(checks, "prosody.force_async_transcription") is Status.PASS

    empty = tmp_path / "empty"
    empty.mkdir()
    checks = check_prosody(
        _deployment(prosody_text=PROSODY), "meet.example.com", [empty], reader
    )
    assert _status(checks, "prosody.force_async_transcription") is Status.FAIL


def test_prosody_without_a_muc_is_a_failure() -> None:
    checks = check_prosody(
        _deployment(prosody_text='VirtualHost "meet.example.com"\n'), "meet.example.com", [],
        lambda p: "",
    )
    assert _status(checks, "prosody.muc") is Status.FAIL


def test_jicofo_requires_a_url_template() -> None:
    checks = check_jicofo(_deployment(), "meeting-1", Path("/nonexistent/custom.conf"))
    assert _status(checks, "jicofo.url-template") is Status.FAIL

    document = _document(
        'jicofo { transcription { url-template = "ws://bridge/transcribe?sessionId='
        '{{MEETING_ID}}" } }'
    )
    checks = check_jicofo(
        _deployment(hocon=document), "meeting-1", Path("/nonexistent/custom.conf")
    )
    assert _status(checks, "jicofo.url-template") is Status.PASS


def test_jicofo_environment_substitution_is_a_warning_not_a_failure() -> None:
    document = _document(
        'jicofo { transcription { url-template = "${?JICOFO_TRANSCRIPTION_URL}" } }'
    )
    checks = check_jicofo(_deployment(hocon=document), "meeting-1", Path("/nonexistent"))
    assert _status(checks, "jicofo.url-template") is Status.WARN


def test_jicofo_reports_an_unincluded_custom_conf(tmp_path: Path) -> None:
    custom = tmp_path / "custom-jicofo.conf"
    custom.write_text('jicofo.transcription.url-template = "ws://bridge/transcribe"\n')
    document = _document('jicofo { transcription { url-template = "ws://other/transcribe" } }')
    checks = check_jicofo(_deployment(hocon=document), "meeting-1", custom)
    assert _status(checks, "jicofo.custom-conf") is Status.FAIL


# --------------------------------------------------------------------------
# Log classification
# --------------------------------------------------------------------------


def test_classify_jvb_counts_positives_and_failures() -> None:
    lines = [
        "jvb  INFO Websocket connected: true",
        "jvb  INFO Sending info to transcriber: {...}",
        "jvb  INFO Starting SSRC 123 for endpoint abc",
        "jvb  INFO Websocket closed with status 1000, reason: closing",
        "jvb  WARN Ping timeout, reconnecting websocket",
    ]
    report = classify_jvb(lines)
    assert report.positives["Websocket connected: true"] == 1
    assert report.positives["Starting SSRC "] == 1
    assert report.normal_closes == 1
    assert "Ping timeout, reconnecting websocket" in report.failures
    assert "Websocket closed with status" not in report.failures


def test_classify_jvb_collects_runtime_urls() -> None:
    report = classify_jvb(["INFO Starting with url=ws://bridge:8080/transcribe?sessionId=x"])
    assert report.runtime_urls == ["ws://bridge:8080/transcribe?sessionId=x"]


def test_classify_jvb_is_empty_for_no_lines() -> None:
    report = classify_jvb([])
    assert report.lines == 0 and not report.positives and not report.failures


def test_classify_jicofo_flags_the_missing_url_error() -> None:
    report = classify_jicofo(["ERROR Transcription enabled, but no URL is configured."])
    assert "Transcription enabled, but no URL is configured." in report.failures


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def _namespace(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "domain": None,
        "jicofo_conf": None,
        "prosody_config": None,
        "meet_config": None,
        "jvb_conf": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _tree(tmp_path: Path, domains: list[str]) -> None:
    meet = tmp_path / "meet"
    conf_avail = tmp_path / "conf.avail"
    meet.mkdir()
    conf_avail.mkdir()
    for domain in domains:
        (meet / f"{domain}-config.js").write_text("var config = {};\n")
        (conf_avail / f"{domain}.cfg.lua").write_text(
            f'VirtualHost "{domain}"\n    modules_enabled = {{ }}\n'
        )


def test_discovery_selects_a_single_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree(tmp_path, ["meet.example.com"])
    monkeypatch.setattr(verify_jitsi, "DEFAULT_MEET_DIR", tmp_path / "meet")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_PROSODY_CONF_AVAIL", tmp_path / "conf.avail")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_JICOFO_CONF", tmp_path / "jicofo.conf")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_PROSODY_MAIN", tmp_path / "prosody.cfg.lua")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_PROSODY_CONF_D", tmp_path / "conf.d")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_JVB_CONF", tmp_path / "jvb.conf")

    deployment = load_deployment(_namespace())
    assert deployment.domain == "meet.example.com"
    assert deployment.meet_config == tmp_path / "meet" / "meet.example.com-config.js"
    assert deployment.prosody_config == tmp_path / "conf.avail" / "meet.example.com.cfg.lua"


def test_discovery_refuses_to_guess_between_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree(tmp_path, ["a.example.com", "b.example.com"])
    monkeypatch.setattr(verify_jitsi, "DEFAULT_MEET_DIR", tmp_path / "meet")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_PROSODY_CONF_AVAIL", tmp_path / "conf.avail")
    with pytest.raises(DiscoveryError, match="--domain"):
        load_deployment(_namespace())


def test_discovery_rejects_a_missing_explicit_file(tmp_path: Path) -> None:
    with pytest.raises(DiscoveryError, match="does not exist"):
        load_deployment(_namespace(jicofo_conf=str(tmp_path / "missing.conf")))


def test_discovery_tolerates_no_configuration_when_not_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify_jitsi, "DEFAULT_MEET_DIR", tmp_path / "meet")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_PROSODY_CONF_AVAIL", tmp_path / "conf.avail")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_JICOFO_CONF", tmp_path / "jicofo.conf")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_JVB_CONF", tmp_path / "jvb.conf")
    deployment = load_deployment(_namespace(), require_files=False)
    assert deployment.jicofo_conf is None


# --------------------------------------------------------------------------
# Exit status
# --------------------------------------------------------------------------


def test_exit_code_only_fails_on_failures() -> None:
    assert exit_code([Section("x", [Check("a", Status.PASS, "")])]) == 0
    assert exit_code([Section("x", [Check("a", Status.WARN, "")])]) == 0
    assert exit_code([Section("x", [Check("a", Status.SKIP, "")])]) == 0
    assert exit_code([Section("x", [Check("a", Status.FAIL, "")])]) == 1


def test_script_runs_from_any_directory(tmp_path: Path) -> None:
    """``python3 /path/to/tools/verify_jitsi.py`` must not need the repo as cwd."""
    script = Path(verify_jitsi.__file__)
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--only" in result.stdout


# --------------------------------------------------------------------------
# Fix proposals: text surgery
# --------------------------------------------------------------------------


def test_normalize_bridge_url_builds_the_documented_template() -> None:
    template, problem = normalize_bridge_url("ws://bridge.example.com:8080")
    assert problem == ""
    assert template == "ws://bridge.example.com:8080/transcribe?sessionId={{MEETING_ID}}"
    # A trailing slash, or the route already given, must not double up.
    assert normalize_bridge_url("ws://h:1/")[0] == "ws://h:1/transcribe?sessionId={{MEETING_ID}}"
    assert normalize_bridge_url("ws://h:1/transcribe")[0] == (
        "ws://h:1/transcribe?sessionId={{MEETING_ID}}"
    )
    # A bare host takes the bridge's usual listener; a bare host:port keeps its port.
    assert normalize_bridge_url("bridge.example.com")[0] == (
        "ws://bridge.example.com:8080/transcribe?sessionId={{MEETING_ID}}"
    )
    assert normalize_bridge_url("bridge.example.com:9000")[0] == (
        "ws://bridge.example.com:9000/transcribe?sessionId={{MEETING_ID}}"
    )
    # A full template is kept verbatim.
    full = "wss://h/transcribe?sessionId={{MEETING_ID}}&x=1"
    assert normalize_bridge_url(full) == (full, "")


@pytest.mark.parametrize("value", ["", "two words", "http://h/transcribe", "ws://h/a b"])
def test_normalize_bridge_url_rejects_unusable_values(value: str) -> None:
    template, problem = normalize_bridge_url(value)
    assert template is None and problem


def test_hocon_block_parses_back_to_the_intended_settings() -> None:
    block = hocon_transcription_block("ws://bridge/transcribe?sessionId={{MEETING_ID}}")
    document = _document(block)
    assert hocon_str(document, "jicofo.transcription.url-template") == (
        "ws://bridge/transcribe?sessionId={{MEETING_ID}}"
    )
    assert hocon_bool(document, "jicofo.transcription.ping.enabled") is True
    assert hocon_duration(document, "jicofo.transcription.ping.interval") == 10.0


def test_append_to_file_text_keeps_one_separating_blank_line() -> None:
    assert append_to_file_text("", "x\n") == "x\n"
    assert append_to_file_text("a\n", "x\n") == "a\n\nx\n"
    assert append_to_file_text("a", "x\n") == "a\n\nx\n"


def test_add_lua_modules_rewrites_an_inline_table_legibly() -> None:
    original = 'Component "conference.meet.example.com" "muc"\n    modules_enabled = { "a"; }\n'
    span = lua_table_span(original, "modules_enabled")
    assert span is not None
    result = add_lua_modules(original, span, ["b"])
    assert result == (
        'Component "conference.meet.example.com" "muc"\n'
        '    modules_enabled = { "a";\n        "b";\n    }\n'
    )
    table = lua_table(result.split("\n", 1)[1], "modules_enabled")
    assert lua_string_list(table) == ["a", "b"]


def test_add_lua_modules_repairs_a_missing_separator_and_keeps_comments() -> None:
    original = 'x = {\n    "a" -- keep me\n}\n'
    span = lua_table_span(original, "x")
    assert span is not None
    result = add_lua_modules(original, span, ["b"])
    assert '"a";' in result or '"a" -- keep me' in result
    assert "-- keep me" in result
    assert '"b";' in result


def test_add_js_property_handles_multiline_empty_and_missing_comma() -> None:
    multiline = "var config = {\n    url: 'x',\n};\n"
    span = find_js_var_object_span(strip_js_comments(multiline))
    assert span is not None
    assert add_js_property(multiline, span, "transcription: { enabled: true },") == (
        "var config = {\n    url: 'x',\n    transcription: { enabled: true },\n};\n"
    )

    empty = "var config = {};\n"
    span = find_js_var_object_span(strip_js_comments(empty))
    assert span is not None
    assert add_js_property(empty, span, "a: 1,") == "var config = {\n    a: 1,\n};\n"

    no_comma = "var config = { url: 'x' };\n"
    span = find_js_var_object_span(strip_js_comments(no_comma))
    assert span is not None
    assert add_js_property(no_comma, span, "a: 1,") == (
        "var config = { url: 'x',\n    a: 1,\n};\n"
    )


def test_find_js_var_object_span_requires_exactly_one_candidate() -> None:
    assert find_js_var_object_span("var config = { a: 1 };") is not None
    assert find_js_var_object_span("var other = { a: 1 };") is None
    assert find_js_var_object_span("var config = { a: 1 }; var config = { b: 2 };") is None


def test_module_source_matches_the_documented_module() -> None:
    docs = (Path(__file__).resolve().parent.parent / "docs" / "jitsi-integration.md").read_text()
    marker = "Create `mod_force_async_transcription.lua`"
    fence = docs.split(marker, 1)[1].split("```lua", 1)[1].split("```", 1)[0]
    assert fence.strip() == PROSODY_MODULE_LUA.strip()


# --------------------------------------------------------------------------
# Fix proposals: the builders
# --------------------------------------------------------------------------


def _reader(files: dict[Path, str]):
    return lambda path: files.get(Path(path))


def test_propose_jicofo_needs_a_bridge_url() -> None:
    (proposal,) = propose_jicofo_fix(
        _deployment(), Path("/nonexistent/custom-jicofo.conf"), bridge_url=None
    )
    assert proposal.target is None
    assert "--bridge-url" in proposal.reason


def test_propose_jicofo_appends_to_the_included_custom_file() -> None:
    jicofo = 'jicofo {\n  conference { x = 1 }\n}\ninclude "custom-jicofo.conf"\n'
    custom = Path("/etc/jitsi/jicofo/custom-jicofo.conf")
    deployment = _deployment(hocon=_document(jicofo, Path("/etc/jitsi/jicofo/jicofo.conf")))
    proposals = propose_jicofo_fix(
        deployment, custom, bridge_url="ws://bridge:8080",
        read=_reader({Path("jicofo.conf"): jicofo}),
    )
    (proposal,) = proposals
    assert proposal.target == custom
    assert proposal.unit == "jicofo"
    merged = parse_hocon(proposal.new_text, custom)[0]
    assert hocon_str(HoconDocument(values=merged), "jicofo.transcription.url-template") == (
        "ws://bridge:8080/transcribe?sessionId={{MEETING_ID}}"
    )


def test_propose_jicofo_adds_the_include_when_it_is_missing() -> None:
    jicofo = "jicofo {\n  conference { x = 1 }\n}\n"
    jicofo_path = Path("/etc/jitsi/jicofo/jicofo.conf")
    custom = Path("/etc/jitsi/jicofo/custom-jicofo.conf")
    deployment = _deployment(hocon=_document(jicofo, jicofo_path), jicofo_conf=jicofo_path)
    proposals = propose_jicofo_fix(
        deployment, custom, bridge_url="ws://bridge:8080",
        read=_reader({jicofo_path: jicofo, custom: ""}),
    )
    assert [p.target for p in proposals] == [custom, jicofo_path]
    assert 'include "custom-jicofo.conf"' in proposals[1].new_text


def test_propose_jicofo_is_silent_when_a_live_template_exists() -> None:
    document = _document(
        'jicofo { transcription { url-template = "ws://bridge/transcribe?sessionId='
        '{{MEETING_ID}}" } }'
    )
    assert propose_jicofo_fix(
        _deployment(hocon=document), Path("/nonexistent"), bridge_url="ws://x:1"
    ) == []


def test_propose_jicofo_warns_about_a_loopback_address() -> None:
    jicofo_path = Path("/etc/jitsi/jicofo/jicofo.conf")
    custom = Path("/etc/jitsi/jicofo/custom-jicofo.conf")
    deployment = _deployment(hocon=HoconDocument(), jicofo_conf=jicofo_path)
    (proposal,) = propose_jicofo_fix(
        deployment, custom, bridge_url="127.0.0.1:9000",
        read=_reader({jicofo_path: 'include "custom-jicofo.conf"\n', custom: ""}),
    )
    assert any("loopback" in note for note in proposal.notes)

    (proposal,) = propose_jicofo_fix(
        deployment, custom, bridge_url="bridge.example.com",
        read=_reader({jicofo_path: 'include "custom-jicofo.conf"\n', custom: ""}),
    )
    assert not any("loopback" in note for note in proposal.notes)


def test_propose_jicofo_keeps_an_existing_custom_template(tmp_path: Path) -> None:
    custom = tmp_path / "custom-jicofo.conf"
    custom.write_text('jicofo.transcription.url-template = "ws://already/t?sessionId=x"\n')
    jicofo_path = tmp_path / "jicofo.conf"
    jicofo_path.write_text("jicofo { conference { x = 1 } }\n")
    deployment = _deployment(hocon=load_hocon(jicofo_path), jicofo_conf=jicofo_path)
    proposals = propose_jicofo_fix(deployment, custom, bridge_url="ws://bridge:8080")
    assert [p.target for p in proposals] == [jicofo_path]
    assert "already" in proposals[0].notes[0]


def test_propose_prosody_writes_the_module_and_the_site_config(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    # The MUC does not enable it yet, so both a module file and a site edit are due.
    deployment = _deployment(prosody_text=PROSODY_WITHOUT_FORCE)
    proposals = propose_prosody_fixes(deployment, [plugin_dir], read=_reader({}))
    by_name = {p.target.name: p for p in proposals}
    assert set(by_name) == {"mod_force_async_transcription.lua", "meet.example.com.cfg.lua"}
    module = by_name["mod_force_async_transcription.lua"]
    assert PROSODY_MODULE_LUA.splitlines()[1] in module.new_text
    site = by_name["meet.example.com.cfg.lua"]
    assert '"force_async_transcription";' in site.new_text
    assert site.unit == "prosody"


def test_propose_prosody_enables_an_existing_module_instead_of_adding_one(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    existing = plugin_dir / "mod_our_own.lua"
    existing.write_text("room.jitsiMetadata.asyncTranscription = true;\n")
    deployment = _deployment(prosody_text=PROSODY)
    proposals = propose_prosody_fixes(
        deployment, [plugin_dir], read=_reader({existing: existing.read_text()})
    )
    (proposal,) = proposals
    assert proposal.target.name == "meet.example.com.cfg.lua"
    assert '"our_own";' in proposal.new_text
    assert "mod_force_async_transcription" not in proposal.new_text


def test_propose_prosody_is_blocked_without_a_modules_enabled_table() -> None:
    text = (
        'VirtualHost "meet.example.com"\n'
        'Component "conference.meet.example.com" "muc"\n'
        "    main_muc = \"conference.meet.example.com\"\n"
    )
    proposals = propose_prosody_fixes(_deployment(prosody_text=text), [Path("/nonexistent")])
    blocked = [p for p in proposals if p.target is None]
    assert blocked, "the site config cannot be edited without a modules_enabled table"
    assert "global module list" in blocked[0].reason
    # The module file itself can still be proposed; only the splice is blocked.
    assert any(p.target is not None for p in proposals)


def test_propose_meet_inserts_into_the_config_object() -> None:
    (proposal,) = propose_meet_fix(_deployment(meet_text=COMMENTED_CLIENT))
    assert proposal.target is not None
    assert "commented out" in proposal.notes[0]
    assert js_boolean(
        find_js_object(strip_js_comments(proposal.new_text), "transcription"), "enabled"
    ) is True
    # The commented sample is still there, untouched.
    assert "//     enabled: false," in proposal.new_text


def test_propose_meet_flips_a_live_false_without_duplicating_the_key() -> None:
    text = "var config = {\n    transcription: { enabled: false },\n};\n"
    (proposal,) = propose_meet_fix(_deployment(meet_text=text))
    assert proposal.new_text.count("transcription:") == 1
    assert "enabled: true" in proposal.new_text


def test_propose_meet_is_silent_when_enabled_and_blocked_when_ambiguous() -> None:
    assert propose_meet_fix(_deployment(meet_text=LIVE_CLIENT)) == []
    (proposal,) = propose_meet_fix(_deployment(meet_text="var notAConfig = {};\n"))
    assert proposal.target is None and proposal.reason


# --------------------------------------------------------------------------
# Fix proposals: idempotency and "the proposal fixes the check"
# --------------------------------------------------------------------------


def test_proposals_satisfy_the_checks_they_address(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "mod_force_async_transcription.lua").write_text(PROSODY_MODULE_LUA)

    prosody_proposals = propose_prosody_fixes(
        _deployment(prosody_text=PROSODY_WITHOUT_FORCE), [plugin_dir], read=_reader({})
    )
    site = next(p for p in prosody_proposals if p.target and p.target.suffix == ".lua"
                and not p.target.name.startswith("mod_"))
    checks = check_prosody(
        _deployment(prosody_text=site.new_text), "meet.example.com", [plugin_dir],
        _reader({plugin_dir / "mod_force_async_transcription.lua": PROSODY_MODULE_LUA}),
    )
    assert _status(checks, "prosody.force_async_transcription") is Status.PASS

    meet_proposal = propose_meet_fix(_deployment(meet_text=COMMENTED_CLIENT))[0]
    checks = check_meet_config(_deployment(meet_text=meet_proposal.new_text))
    assert _status(checks, "meet.transcription.enabled") is Status.PASS


def test_running_the_builders_on_their_own_output_proposes_nothing(tmp_path: Path) -> None:
    meet_proposal = propose_meet_fix(_deployment(meet_text=COMMENTED_CLIENT))[0]
    assert propose_meet_fix(_deployment(meet_text=meet_proposal.new_text)) == []

    # Once the site config enables a module that really exists in the plugin
    # path, the Prosody builder has nothing left to propose either.
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    module = plugin_dir / "mod_force_async_transcription.lua"
    module.write_text(PROSODY_MODULE_LUA)
    reader = _reader({module: PROSODY_MODULE_LUA})
    without = PROSODY.replace('        "force_async_transcription";\n', "")
    (site_proposal,) = propose_prosody_fixes(
        _deployment(prosody_text=without), [plugin_dir], read=reader
    )
    assert '"force_async_transcription";' in site_proposal.new_text
    assert propose_prosody_fixes(
        _deployment(prosody_text=site_proposal.new_text), [plugin_dir], read=reader
    ) == []


# --------------------------------------------------------------------------
# Fix proposals: the writer
# --------------------------------------------------------------------------


def _proposal_for(target: Path, text: str = "new content\n") -> verify_jitsi.Proposal:
    return verify_jitsi.Proposal(
        check_id="test", summary="test", target=target, new_text=text
    )


def test_writer_writes_beside_the_original_and_leaves_it_alone(tmp_path: Path) -> None:
    target = tmp_path / "jicofo.conf"
    target.write_text("original\n")
    target.chmod(0o640)
    before = (target.read_bytes(), target.stat().st_mtime_ns)

    result = write_proposal(_proposal_for(target))
    assert result.path == tmp_path / "jicofo.conf.new"
    assert result.path.read_text() == "new content\n"
    assert result.path.stat().st_mode & 0o777 == 0o640
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_writer_refuses_to_clobber_without_force(tmp_path: Path) -> None:
    target = tmp_path / "jicofo.conf"
    target.write_text("original\n")
    (tmp_path / "jicofo.conf.new").write_text("earlier proposal\n")

    result = write_proposal(_proposal_for(target))
    assert result.path is None
    assert "--force-fix" in result.error
    assert (tmp_path / "jicofo.conf.new").read_text() == "earlier proposal\n"

    result = write_proposal(_proposal_for(target), force=True)
    assert result.path is not None
    assert result.path.read_text() == "new content\n"


def test_writer_follows_a_symlink_to_the_file_being_edited(tmp_path: Path) -> None:
    conf_avail = tmp_path / "conf.avail"
    conf_d = tmp_path / "conf.d"
    conf_avail.mkdir()
    conf_d.mkdir()
    real = conf_avail / "meet.example.com.cfg.lua"
    real.write_text("VirtualHost ...\n")
    link = conf_d / "meet.example.com.cfg.lua"
    link.symlink_to(real)

    result = write_proposal(_proposal_for(link))
    assert result.path == conf_avail / "meet.example.com.cfg.lua.new"
    assert not (conf_d / "meet.example.com.cfg.lua.new").exists()


def test_writer_uses_the_output_dir_and_reports_an_unwritable_one(tmp_path: Path) -> None:
    target = tmp_path / "jicofo.conf"
    target.write_text("original\n")
    staging = tmp_path / "staging"
    result = write_proposal(_proposal_for(target), output_dir=staging)
    assert result.path == staging / "jicofo.conf.new"

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        result = write_proposal(_proposal_for(target), output_dir=locked)
        assert result.path is None
        assert "sudo" in result.error or "output-dir" in result.error
    finally:
        locked.chmod(0o700)


def test_writer_reports_a_blocked_proposal() -> None:
    blocked = verify_jitsi.Proposal("x", "summary", reason="nothing to do")
    result = write_proposal(blocked)
    assert result.path is None and "nothing to do" in result.error


# --------------------------------------------------------------------------
# Fix proposals: the command line
# --------------------------------------------------------------------------


def _broken_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    meet = tmp_path / "meet"
    conf_avail = tmp_path / "conf.avail"
    plugins = tmp_path / "plugins"
    for directory in (meet, conf_avail, plugins):
        directory.mkdir()
    jicofo = tmp_path / "jicofo.conf"
    jicofo.write_text("jicofo {\n  transcription {\n    // url-template = \"ws://old/\"\n  }\n}\n")
    prosody = conf_avail / "meet.example.com.cfg.lua"
    prosody.write_text(
        'VirtualHost "meet.example.com"\n'
        '    modules_enabled = { "room_metadata"; }\n'
        'Component "conference.meet.example.com" "muc"\n'
        '    main_muc = "conference.meet.example.com"\n'
        '    modules_enabled = { "muc_meeting_id"; }\n'
        'Component "metadata.meet.example.com" "room_metadata_component"\n'
    )
    meet_config = meet / "meet.example.com-config.js"
    meet_config.write_text("var config = {\n    url: 'x',\n};\n")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_MEET_DIR", meet)
    monkeypatch.setattr(verify_jitsi, "DEFAULT_PROSODY_CONF_AVAIL", conf_avail)
    monkeypatch.setattr(verify_jitsi, "DEFAULT_JICOFO_CONF", jicofo)
    monkeypatch.setattr(verify_jitsi, "DEFAULT_JVB_CONF", tmp_path / "jvb.conf")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_PROSODY_MAIN", tmp_path / "prosody.cfg.lua")
    monkeypatch.setattr(verify_jitsi, "DEFAULT_PROSODY_CONF_D", tmp_path / "conf.d")
    return {"jicofo": jicofo, "prosody": prosody, "meet": meet_config, "plugins": plugins}


def test_main_fix_stages_proposals_and_keeps_the_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = _broken_tree(tmp_path, monkeypatch)
    staging = tmp_path / "staging"
    code = verify_jitsi.main([
        "--only", "fix", "--domain", "meet.example.com",
        "--bridge-url", "ws://bridge.example.com:8080",
        "--plugin-dir", str(tree["plugins"]),
        "--output-dir", str(staging),
    ])
    out = capsys.readouterr().out
    assert code == 1, "the deployment is still broken, so the exit status must stay 1"
    assert "config" in out and "fix" in out, "--only fix must also run the config checks"
    staged = {path.name for path in staging.iterdir()}
    assert "jicofo.conf.new" in staged
    assert "custom-jicofo.conf.new" in staged
    assert "meet.example.com.cfg.lua.new" in staged
    assert "mod_force_async_transcription.lua.new" in staged
    assert "meet.example.com-config.js.new" in staged
    assert "PROPOSED" in out
    # The originals are untouched.
    assert tree["jicofo"].read_text().count("url-template") == 1  # the comment only


def test_main_rejects_an_invalid_bridge_url(capsys: pytest.CaptureFixture[str]) -> None:
    code = verify_jitsi.main(["--only", "config", "--bridge-url", "http://bridge/transcribe"])
    assert code == 2
    assert "error: --bridge-url" in capsys.readouterr().err


def test_main_without_fix_proposes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = _broken_tree(tmp_path, monkeypatch)
    code = verify_jitsi.main(
        ["--domain", "meet.example.com", "--plugin-dir", str(tree["plugins"])]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "PROPOSED" not in out
    assert not list(tmp_path.rglob("*.new"))


# --------------------------------------------------------------------------
# Fix proposals: room metadata
# --------------------------------------------------------------------------

NO_ROOM_METADATA = (
    'plugin_paths = { "/plugins" }\n'
    'VirtualHost "meet.example.com"\n'
    '    modules_enabled = { "muc_meeting_id"; }\n'
    "\n"
    'Component "conference.meet.example.com" "muc"\n'
    '    main_muc = "conference.meet.example.com"\n'
    '    modules_enabled = { }\n'
)


def _plugin_reader(plugin_dir: Path) -> object:
    return _reader({plugin_dir / "mod_room_metadata_component.lua": "-- component\n"})


def test_lua_block_offsets_start_right_after_the_header() -> None:
    """The fixer splices at these offsets; swallowing the newline broke it."""
    text = 'VirtualHost "meet.example.com"\n    modules_enabled = { }\n'
    (block,) = find_lua_blocks(text)
    assert text[block.start :].startswith("\n    modules_enabled")


def test_propose_prosody_adds_the_room_metadata_component(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    proposals = propose_prosody_fixes(
        _deployment(prosody_text=NO_ROOM_METADATA), [plugin_dir],
        read=_plugin_reader(plugin_dir),
    )
    by_id = {p.check_id: p for p in proposals}
    site = by_id["prosody.room_metadata"]
    assert site.target is not None
    assert 'Component "metadata.meet.example.com" "room_metadata_component"' in site.new_text
    assert 'muc_component = "conference.meet.example.com"' in site.new_text
    # The deprecated module is gone upstream: only the component is proposed.
    assert '"room_metadata";' not in site.new_text
    assert 'room_metadata_component = "' not in site.new_text

    # Installing it must satisfy the check it addresses.
    checks = check_prosody(
        _deployment(prosody_text=site.new_text), "meet.example.com", [plugin_dir],
        _plugin_reader(plugin_dir),
    )
    assert _status(checks, "prosody.room_metadata") is Status.PASS


def test_propose_prosody_refuses_a_component_whose_module_is_missing(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()  # readable, but the room metadata plugins are not installed
    proposals = propose_prosody_fixes(
        _deployment(prosody_text=NO_ROOM_METADATA), [plugin_dir], read=_reader({})
    )
    blocked = next(p for p in proposals if p.check_id == "prosody.room_metadata")
    assert blocked.target is None
    assert "jitsi-meet-prosody" in blocked.reason
    # Nothing that would stop Prosody from starting was proposed.
    edits = [p for p in proposals if p.target is not None]
    assert all('Component "metadata.' not in p.new_text for p in edits)
    assert all('room_metadata_component = "' not in p.new_text for p in edits)


def test_propose_prosody_skips_room_metadata_work_when_it_is_present(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    module = plugin_dir / "mod_force_async_transcription.lua"
    module.write_text(PROSODY_MODULE_LUA)
    component = plugin_dir / "mod_room_metadata_component.lua"
    component.write_text("-- component\n")
    reader = _reader({module: PROSODY_MODULE_LUA, component: "-- component\n"})
    proposals = propose_prosody_fixes(
        _deployment(prosody_text=PROSODY), [plugin_dir], read=reader
    )
    # Everything the checker wants is present, so there is nothing to propose.
    assert proposals == []


def test_room_metadata_check_accepts_the_modern_component_only_shape() -> None:
    """Upstream removed mod_room_metadata.lua in 2026; the component is what counts."""
    component_only = (
        'VirtualHost "meet.example.com"\n'
        '    modules_enabled = { }\n'
        'Component "conference.meet.example.com" "muc"\n'
        '    main_muc = "conference.meet.example.com"\n'
        'Component "metadata.meet.example.com" "room_metadata_component"\n'
        '    muc_component = "conference.meet.example.com"\n'
    )
    checks = check_prosody(
        _deployment(prosody_text=component_only), "meet.example.com", [], lambda p: None
    )
    assert _status(checks, "prosody.room_metadata") is Status.PASS

    module_only = component_only.replace(
        'Component "metadata.meet.example.com" "room_metadata_component"\n'
        '    muc_component = "conference.meet.example.com"\n',
        "",
    ).replace('modules_enabled = { }', 'modules_enabled = { "room_metadata"; }')
    checks = check_prosody(
        _deployment(prosody_text=module_only), "meet.example.com", [], lambda p: None
    )
    assert _status(checks, "prosody.room_metadata") is Status.WARN


def test_propose_prosody_does_not_re_add_an_already_enabled_module(tmp_path: Path) -> None:
    """The module name may be enabled while its file is missing: install, do not duplicate."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    text = NO_ROOM_METADATA.replace(
        "modules_enabled = { }", 'modules_enabled = { "force_async_transcription"; }'
    )
    proposals = propose_prosody_fixes(
        _deployment(prosody_text=text), [plugin_dir], read=_plugin_reader(plugin_dir)
    )
    ids = {p.check_id for p in proposals}
    assert "prosody.force_async_transcription" in ids          # the module file is proposed
    site = next(p for p in proposals if p.check_id == "prosody.room_metadata")
    assert site.new_text.count("force_async_transcription") == 1


def test_prosody_warns_when_the_module_never_publishes_the_metadata(tmp_path: Path) -> None:
    """Setting the flag is not enough: the component only broadcasts on an event."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    silent = plugin_dir / "mod_force_async_transcription.lua"
    silent.write_text(
        "module:hook('muc-room-created', function(event)\n"
        "    event.room.jitsiMetadata.asyncTranscription = true;\n"
        "end, -2);\n"
    )
    reader = _reader({silent: silent.read_text()})
    checks = check_prosody(
        _deployment(prosody_text=PROSODY), "meet.example.com", [plugin_dir], reader
    )
    assert _status(checks, "prosody.force_async_transcription") is Status.PASS
    assert _status(checks, "prosody.force_async_transcription.publish") is Status.WARN

    # The shipped module fires the event, so it does not warn.
    publishing = plugin_dir / "mod_force_async_transcription.lua"
    publishing.write_text(PROSODY_MODULE_LUA)
    checks = check_prosody(
        _deployment(prosody_text=PROSODY), "meet.example.com", [plugin_dir],
        _reader({publishing: PROSODY_MODULE_LUA}),
    )
    assert not [c for c in checks if c.id.endswith(".publish")]


def test_prosody_requires_features_identity_for_clients_to_see_the_metadata(
    tmp_path: Path,
) -> None:
    """The identity is how clients learn the component exists at all."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    identity = plugin_dir / "mod_features_identity.lua"
    identity.write_text("-- identity\n")
    reader = _reader({identity: "-- identity\n"})

    checks = check_prosody(
        _deployment(prosody_text=PROSODY), "meet.example.com", [plugin_dir], reader
    )
    assert _status(checks, "prosody.features_identity") is Status.PASS

    checks = check_prosody(
        _deployment(prosody_text=PROSODY_WITHOUT_IDENTITY), "meet.example.com",
        [plugin_dir], reader,
    )
    check = next(c for c in checks if c.id == "prosody.features_identity")
    assert check.status is Status.FAIL
    # The client-side symptom is what an admin sees, so it is in the message.
    assert "getMetadata() stays {}" in check.summary
    assert 'modules_enabled on "meet.example.com"' in check.fix


def test_prosody_features_identity_reports_a_missing_plugin_file(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()  # readable, but the identity module is not installed
    checks = check_prosody(
        _deployment(prosody_text=PROSODY_WITHOUT_IDENTITY), "meet.example.com",
        [plugin_dir], _reader({}),
    )
    check = next(c for c in checks if c.id == "prosody.features_identity")
    assert check.status is Status.FAIL
    assert "jitsi-meet-prosody" in check.fix


def test_prosody_features_identity_is_not_checkable_without_the_component() -> None:
    """No component, nothing to advertise: the check stays out of the way."""
    without = PROSODY_WITHOUT_IDENTITY.replace(
        'Component "metadata.meet.example.com" "room_metadata_component"\n'
        '    muc_component = "conference.meet.example.com"\n',
        "",
    )
    checks = check_prosody(
        _deployment(prosody_text=without), "meet.example.com", [], lambda p: None
    )
    assert not [c for c in checks if c.id == "prosody.features_identity"]


def test_propose_prosody_advertises_the_identity_on_the_main_host(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    module = plugin_dir / "mod_force_async_transcription.lua"
    module.write_text(PROSODY_MODULE_LUA)
    identity = plugin_dir / "mod_features_identity.lua"
    identity.write_text("-- identity\n")
    reader = _reader({module: PROSODY_MODULE_LUA, identity: "-- identity\n"})

    proposals = propose_prosody_fixes(
        _deployment(prosody_text=PROSODY_WITHOUT_IDENTITY), [plugin_dir], read=reader
    )
    (site,) = proposals
    assert site.check_id == "prosody.features_identity"
    assert '"features_identity";' in site.new_text
    # On the host clients query, not on the MUC.
    host_table = lua_table(
        lua_uncomment(site.new_text).split("Component")[0], "modules_enabled"
    )
    assert "features_identity" in lua_string_list(host_table)

    checks = check_prosody(
        _deployment(prosody_text=site.new_text), "meet.example.com", [plugin_dir], reader
    )
    assert _status(checks, "prosody.features_identity") is Status.PASS

    # Running the builder again on its own output proposes nothing.
    assert propose_prosody_fixes(
        _deployment(prosody_text=site.new_text), [plugin_dir], read=reader
    ) == []


def test_propose_prosody_refuses_the_identity_when_its_module_is_missing(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()  # readable, module absent: Prosody would log a load error
    proposals = propose_prosody_fixes(
        _deployment(prosody_text=PROSODY_WITHOUT_IDENTITY), [plugin_dir], read=_reader({})
    )
    blocked = next(p for p in proposals if p.check_id == "prosody.features_identity")
    assert blocked.target is None
    assert "jitsi-meet-prosody" in blocked.reason
