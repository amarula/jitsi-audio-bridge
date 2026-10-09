# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Test configuration.

Puts ``src/`` and the repository root on the import path so the suite runs from
a checkout without needing ``pip install -e .`` first. An installed copy of the
package takes precedence if one is present.

``src`` makes ``jitsi_audio_bridge`` importable; the root makes the ``tools``
test environment importable, which some tests exercise directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

for entry in (SRC, ROOT):
    if entry.is_dir() and str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
