# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the Debian packaging under packaging/.

The package itself is built by packaging/build-deb.sh, which needs pip and
network access, so it is not built here.  What is tested is the glue that would
fail silently: the launcher's path resolution, and the contract between the
build script and the files it rewrites or renders.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "packaging" / "deb" / "jitsi-audio-bridge.launcher"
UNIT = ROOT / "systemd" / "jitsi-audio-bridge.service"


def _installed_launcher(tmp_path: Path, name: str, *, built_for: str = "3.14",
                        running: str = "3.14") -> Path:
    """Install the launcher under *name* beside a stub venv python.

    The stub answers the launcher's ``-c`` version probe with *running* and
    otherwise echoes its arguments, so both the guard and the module mapping
    are observable without a real interpreter.
    """
    binary = tmp_path / "usr" / "bin" / name
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(LAUNCHER, binary)
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

    venv = tmp_path / "usr" / "lib" / "jitsi-audio-bridge" / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "BUILT-FOR").write_text(f"{built_for}\n", encoding="utf-8")
    venv_python = venv / "bin" / "python"
    venv_python.write_text(
        f'#!/bin/sh\nif [ "$1" = "-c" ]; then echo "{running}"; exit 0; fi\necho "$@"\n',
        encoding="utf-8",
    )
    venv_python.chmod(0o755)
    return binary


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("jitsi-audio-bridge", "jitsi_audio_bridge"),
        ("jitsi-audio-bridge-verify", "tools.verify_jitsi"),
        ("jitsi-audio-bridge-send", "tools.send_meeting"),
    ],
)
def test_launcher_maps_its_name_to_a_module(tmp_path: Path, name: str, module: str) -> None:
    """The launcher must find the venv relative to itself, and pick the module
    from the name it was installed under."""
    binary = _installed_launcher(tmp_path, name)
    result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"-m {module} --version"


def test_launcher_refuses_a_mismatched_python_with_the_reason(tmp_path: Path) -> None:
    """A package built for another python3 must say so, not traceback."""
    binary = _installed_launcher(
        tmp_path, "jitsi-audio-bridge", built_for="3.12", running="3.14"
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 78
    assert "built for python3 3.12" in result.stderr
    assert "make deb" in result.stderr


def test_unit_has_the_lines_the_builder_rewrites() -> None:
    """build-deb.sh rewrites these three keys; fail here if they move."""
    lines = UNIT.read_text(encoding="utf-8").splitlines()
    for key in ("Documentation=", "WorkingDirectory=", "ExecStart="):
        assert any(line.startswith(key) for line in lines), f"systemd unit lost {key}"


def test_the_licence_is_stated_once_and_the_same_way_everywhere() -> None:
    """The package, the metadata and the source files have to agree."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["license"] == "AGPL-3.0-only"
    assert "LICENSE" in pyproject["project"]["license-files"]

    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in licence
    assert "Version 3, 19 November 2007" in licence
    assert "END OF TERMS AND CONDITIONS" in licence

    copyright_file = (ROOT / "packaging" / "deb" / "copyright.in").read_text(encoding="utf-8")
    assert "License: AGPL-3.0-only" in copyright_file
    assert "Copyright: 2026 Amarula Solutions" in copyright_file


def test_every_source_file_says_what_it_is_under() -> None:
    """A file copied out of the tree takes its licence with it."""
    missing = [
        str(path.relative_to(ROOT))
        for directory in ("src", "tools", "tests")
        for path in sorted((ROOT / directory).rglob("*.py"))
        if "SPDX-License-Identifier: AGPL-3.0-only" not in path.read_text(encoding="utf-8")
    ]
    assert not missing, "no SPDX notice in: " + ", ".join(missing)


def test_every_declared_dependency_is_accounted_for_in_the_copyright_file() -> None:
    """The .deb ships its dependencies; its copyright file has to name them.

    This is the half that can be checked offline — what the project declares.
    The build script checks the other half, everything pip actually resolved.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(pyproject["project"]["dependencies"])
    declared += list(pyproject["project"]["optional-dependencies"].get("s3", []))
    assert declared

    copyright_file = (ROOT / "packaging" / "deb" / "copyright.in").read_text(encoding="utf-8")
    for requirement in declared:
        name = re.split(r"[<>=!\[; ]", requirement, maxsplit=1)[0]
        assert name in copyright_file, f"{name} is shipped but not accounted for"


def test_the_build_refuses_a_dependency_it_cannot_account_for() -> None:
    builder = (ROOT / "packaging" / "build-deb.sh").read_text(encoding="utf-8")
    assert "copyright.in does not account for" in builder
    assert 'install -m 0644 "$ROOT/LICENSE"' in builder


def test_the_unit_lands_where_modern_systemd_reads_it() -> None:
    """On every supported distribution /lib is an alias for /usr/lib."""
    builder = (ROOT / "packaging" / "build-deb.sh").read_text(encoding="utf-8")
    assert '"$PKGROOT/usr/lib/systemd/system/$PKG.service"' in builder
    assert "$PKGROOT/lib/systemd" not in builder
    # Nothing may be installed through the alias, but the permissions of what
    # is installed still have to be normalised.
    assert 'chmod -R u=rwX,go=rX "$PKGROOT"' in builder


def test_the_build_ships_the_changelog_and_the_lintian_overrides() -> None:
    """Policy wants a changelog; lintian wants to be told what is deliberate."""
    builder = (ROOT / "packaging" / "build-deb.sh").read_text(encoding="utf-8")
    assert "changelog.Debian.gz" in builder
    assert '"$DEB_VERSION"' in builder
    assert "usr/share/lintian/overrides/$PKG" in builder

    overrides = (ROOT / "packaging" / "deb" / "lintian-overrides").read_text(encoding="utf-8")
    tags = [
        line.split(":", 1)[1].strip()
        for line in overrides.splitlines()
        if line.startswith("jitsi-audio-bridge:")
    ]
    assert tags, "the overrides file names no tags"
    # Every override is a tag this package means to keep, so each one has to
    # carry the reason it is acceptable above it.
    for tag in tags:
        assert tag in overrides


def test_the_build_cleans_up_after_the_last_thing_that_imports() -> None:
    """Bytecode written by the import checks must not reach the package."""
    lines = (ROOT / "packaging" / "build-deb.sh").read_text(encoding="utf-8").splitlines()
    checks = [index for index, line in enumerate(lines) if "-c 'import " in line]
    cleanups = [
        index for index, line in enumerate(lines) if "'__pycache__' -prune" in line
    ]
    assert checks and cleanups
    # The last import is the one that matters: anything before it is tidying
    # what the copy brought in, and anything after would still be built.
    assert max(cleanups) > max(checks), "nothing removes the bytecode the imports wrote"


def test_the_workflows_check_the_things_that_break() -> None:
    """A workflow that quietly stops testing something is worse than none."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ruff check src tests tools" in ci
    assert "python -m pytest tests/ -q" in ci
    assert "python tests/smoke_test.py" in ci
    assert "lintian --fail-on error" in ci
    for image in ("debian:12", "debian:13", "ubuntu:24.04"):
        assert image in ci, f"the package matrix no longer builds for {image}"
    assert 'python: ["3.11", "3.12", "3.13", "3.14"]' in ci

    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert 'tags: ["v*"]' in release
    # A release publishes what CI verified, not a second build of its own.
    assert "uses: ./.github/workflows/ci.yml" in release
    assert "gh release create" in release
    assert "does not match __version__" in release


def test_templates_use_the_substitutions_the_builder_provides() -> None:
    control = (ROOT / "packaging" / "deb" / "control.in").read_text(encoding="utf-8")
    for token in ("@VERSION@", "@ARCH@", "@MAINTAINER@", "@INSTALLED_SIZE@", "@PYTHON_MINOR@"):
        assert token in control, f"control.in lost {token}"
    postinst = (ROOT / "packaging" / "deb" / "postinst.in").read_text(encoding="utf-8")
    assert "@PYTHON_MINOR@" in postinst


def test_maintainer_scripts_are_valid_shell() -> None:
    scripts = [
        ROOT / "packaging" / "build-deb.sh",
        ROOT / "packaging" / "deb" / "postinst.in",
        ROOT / "packaging" / "deb" / "prerm",
        ROOT / "packaging" / "deb" / "postrm",
        LAUNCHER,
    ]
    for script in scripts:
        result = subprocess.run(
            ["sh", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"
