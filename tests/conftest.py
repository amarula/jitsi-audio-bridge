"""Test configuration.

Puts ``src/`` on the import path so the suite runs from a checkout without
needing ``pip install -e .`` first. An installed copy of the package takes
precedence if one is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
