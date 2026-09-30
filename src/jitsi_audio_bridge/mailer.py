"""SMTP delivery of the finished meeting summary and transcript.

Synchronous, like :mod:`jitsi_audio_bridge.ai_client`, and called from the
daemon's worker thread.

No path handling: the transcript is read from the path it is handed.
"""

from __future__ import annotations

import logging
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path

from .config import SmtpConfig

logger = logging.getLogger(__name__)

#: Characters that are safe in a mail attachment filename parameter.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

#: Upper bound on the room-name fragment of the filename.
_MAX_FILENAME_STEM = 80


def safe_attachment_name(room_name: str) -> str:
    """Build an attachment filename that cannot escape a directory or inject.

    ``room_name`` arrives from the JVB's metadata and is therefore not
    trustworthy: a newline in it could inject additional MIME parameters, and a
    path separator could direct the write elsewhere.
    """
    stem = _UNSAFE_FILENAME_CHARS.sub("_", room_name).strip("._")
    return f"{(stem or 'meeting')[:_MAX_FILENAME_STEM]}_transcript.txt"


def _usable_recipients(addresses: list[str] | None) -> list[str]:
    """Drop addresses that are empty or would be rejected when building headers.

    Participants come from the JVB's metadata, so an address is one more
    untrusted string: one containing a newline would otherwise raise while
    constructing the message and cost the whole summary.
    """
    usable: list[str] = []
    for address in addresses or []:
        if not isinstance(address, str):
            continue
        candidate = address.strip()
        if not candidate:
            continue
        if any(character in candidate for character in "\r\n"):
            logger.warning("ignoring recipient with an embedded newline")
            continue
        usable.append(candidate)
    return usable


def _subject_for(room_name: str) -> str:
    """Build a subject line with any embedded newlines flattened out."""
    if not isinstance(room_name, str):
        room_name = str(room_name)
    flattened = " ".join(room_name.split())
    return f"Meeting Summary: {flattened or 'Meeting'}"


def send_meeting_email(
    recipients: list[str] | None,
    room_name: str,
    summary_text: str,
    transcript_path: str | Path | None,
    smtp: SmtpConfig,
) -> bool:
    """Email the summary to *recipients*, attaching the transcript file.

    Falls back to the configured fallback recipient when the meeting recorded
    no addresses.  Returns whether the message was handed to the relay.
    """
    targets = _usable_recipients(recipients)
    if not targets and smtp.fallback_recipient:
        logger.info("no usable recipients in metadata; using the fallback recipient")
        targets = _usable_recipients([smtp.fallback_recipient])
    if not targets:
        logger.error("no recipients and no usable fallback configured; not sending")
        return False

    try:
        message = EmailMessage()
        message["Subject"] = _subject_for(room_name)
        message["From"] = smtp.sender
        message["To"] = ", ".join(targets)
        message.set_content(f"Meeting summary for '{room_name}':\n\n{summary_text}")

        if transcript_path:
            path = Path(transcript_path)
            try:
                data = path.read_bytes()
            except OSError as exc:
                # Still worth sending the summary without its attachment.
                logger.error("cannot attach %s: %s", path, exc)
            else:
                message.add_attachment(
                    data,
                    maintype="text",
                    subtype="plain",
                    filename=safe_attachment_name(room_name),
                )

        with smtplib.SMTP(smtp.host, smtp.port, timeout=60) as server:
            if smtp.use_starttls:
                server.starttls()
            if smtp.user and smtp.password:
                server.login(smtp.user, smtp.password)
            server.send_message(message)
    except (smtplib.SMTPException, OSError, ValueError) as exc:
        # ValueError covers a header the email package refuses to encode; the
        # transcript on disk is still intact, so this is logged, not raised.
        logger.error("could not send the meeting summary: %s", exc)
        return False

    logger.info("meeting summary for %r sent to %s", room_name, ", ".join(targets))
    return True
