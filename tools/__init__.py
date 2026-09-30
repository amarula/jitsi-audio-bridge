"""Local test environment for jitsi-audio-bridge.

Three runnable pieces:

``stubs``
    Stub Whisper, Ollama and SMTP services, so the pipeline can run to
    completion without any real dependency.
``send_meeting``
    A sender simulator that speaks the bridge's wire protocol, standing in for
    whatever forwards audio from the Jitsi side.
``testenv``
    Brings the stubs and the bridge up together and prints how to drive them.

Run them from a checkout without installing the package first: this module puts
``src/`` on the import path.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
