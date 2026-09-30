"""Support ``python -m jitsi_audio_bridge``."""

from __future__ import annotations

from .daemon import main

if __name__ == "__main__":
    raise SystemExit(main())
