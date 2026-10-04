"""Tests for the Debian packaging under packaging/.

The package itself is built by packaging/build-deb.sh, which needs pip and
network access, so it is not built here.  What is tested is the glue that would
fail silently: the launcher's path resolution, and the contract between the
build script and the files it rewrites or renders.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
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
