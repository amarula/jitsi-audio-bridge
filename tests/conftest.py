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

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

for entry in (SRC, ROOT):
    if entry.is_dir() and str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


#: A meeting the mail tests render: three model sections in a language that is
#: not English, both transcript shapes, and a speaker whose name is markup.
SAMPLE_SUMMARY = (
    "## Riassunto esecutivo\n"
    "Il team ha rivisto l'implementazione di jitsi-audio-bridge in Python.\n"
    "\n"
    "## Punti chiave\n"
    "- Risolto il problema del media export del JVB\n"
    "- Gestiti i disconnect del WebSocket\n"
    "\n"
    "## Azioni\n"
    "- Verificare la config di Prosody\n"
)

SAMPLE_TRANSCRIPT = (
    "[00:00:04] Elena: Welcome everyone. Let's kick off the review.\n"
    "\n"
    "[00:00:12] Michael: Thanks Elena. I've updated daemon.py.\n"
    "\n"
    "[00:00:25] David: How are we handling audio tags?\n"
    "\n"
    "[participant-tag-4]: recorded without a timeline\n"
    "\n"
    "[00:00:31] <script>alert(1)</script>: not really a name\n"
)


@pytest.fixture
def mail_context() -> dict[str, object]:
    """The template's context for :data:`SAMPLE_SUMMARY` and its transcript."""
    from jitsi_audio_bridge import mail_render

    return mail_render.build_context(
        heading="Meeting Summary & Transcript",
        room_name="jitsi-ai-review",
        when="2026-10-09 11:30",
        participant_count=4,
        introduction="Please find the automated summary for room 'jitsi-ai-review'.",
        summary_text=SAMPLE_SUMMARY,
        transcript_text=SAMPLE_TRANSCRIPT,
        attachments=["jitsi-ai-review_transcript.txt", "jitsi-audio-review_summary.md"],
        sign_off="Best regards,\nAutomated meeting transcription",
        footer="Generated entirely on-premise. No cloud dependency.",
        preview_turns=10,
        recording=mail_render.Recording(
            url="https://minio.example.com/meetings/x.mp4?X-Amz-Signature=abc",
            label="Recording",
            expires="2026-10-16",
        ),
    )
