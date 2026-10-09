# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Support ``python -m jitsi_audio_bridge``."""

from __future__ import annotations

from .daemon import main

if __name__ == "__main__":
    raise SystemExit(main())
