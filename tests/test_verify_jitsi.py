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
from pathlib import Path

import pytest

from tools import verify_jitsi
from tools.verify_jitsi import (
    Check,
    Deployment,
    DiscoveryError,
    HoconDocument,
    Section,
    Status,
    analyze_template,
    check_jicofo,
    check_meet_config,
    check_prosody,
    classify_jicofo,
    classify_jvb,
    exit_code,
    find_js_object,
    find_lua_blocks,
    hocon_bool,
    hocon_duration,
    hocon_str,
    js_boolean,
    load_deployment,
    lua_module_names,
    lua_sets_async_transcription,
    lua_table,
    lua_uncomment,
    parse_hocon,
    strip_js_comments,
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
                hocon: HoconDocument | None = None) -> Deployment:
    return Deployment(
        domain="meet.example.com",
        jicofo_conf=Path("jicofo.conf"),
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
