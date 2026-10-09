# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Jitsi audio bridge.

Receives per-participant Opus audio from a Jitsi Videobridge over a WebSocket,
records each participant to a WAV file, then transcribes, summarises and emails
the result once the meeting ends.

Module layout:

``config``
    Reads ``config.ini`` and the process environment.  The only module that
    touches either.
``audio``
    Opus decoding and meeting-metadata parsing.  No network access.
``ai_client``
    HTTP clients for the local Whisper and Ollama endpoints.  No path handling.
``mailer``
    SMTP delivery of the finished summary.  No path handling.
``daemon``
    The WebSocket server and the post-processing pipeline.  The only module
    that knows about asyncio.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
