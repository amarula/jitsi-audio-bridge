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


def attachment_stem(room_name: str) -> str:
    """Reduce a room name to something safe to build a filename from.

    ``room_name`` is derived from metadata the sender controls and is therefore
    not trustworthy: a newline in it could inject additional MIME parameters,
    and a path separator could direct an attachment name elsewhere.
    """
    stem = _UNSAFE_FILENAME_CHARS.sub("_", str(room_name)).strip("._")
    return (stem or "meeting")[:_MAX_FILENAME_STEM]


def safe_attachment_name(room_name: str) -> str:
    """The transcript's attachment filename."""
    return f"{attachment_stem(room_name)}_transcript.txt"


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


def _subject_for(room_name: str, suffix: str = "") -> str:
    """Build a subject line with any embedded newlines flattened out."""
    flattened = " ".join(str(room_name).split()) or "Meeting"
    subject = f"Meeting Summary & Transcript: {flattened}"
    # Tolerate a suffix written either as "Amarula" or "- Amarula": the
    # separator is added here, so a leading one is stripped rather than doubled.
    cleaned = " ".join(str(suffix).split()).strip("-–—").strip()
    if cleaned:
        subject = f"{subject} - {cleaned}"
    return subject


def _attach(message: EmailMessage, path: str | Path | None, filename: str, subtype: str) -> bool:
    """Attach a file if it exists.  Returns whether it was attached."""
    if not path:
        return False
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        logger.error("cannot attach %s: %s", source, exc)
        return False
    message.add_attachment(data, maintype="text", subtype=subtype, filename=filename)
    return True


def send_meeting_email(
    recipients: list[str] | None,
    room_name: str,
    summary_text: str,
    transcript_path: str | Path | None,
    summary_path: str | Path | None,
    smtp: SmtpConfig,
) -> bool:
    """Email the summary to *recipients*, attaching the transcript and summary.

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

    stem = attachment_stem(room_name)

    try:
        message = EmailMessage()
        message["Subject"] = _subject_for(room_name, smtp.subject_suffix)
        message["From"] = smtp.sender
        message["To"] = ", ".join(targets)
        message.set_content(
            f"Hello,\n\n"
            f"Please find the automated summary and raw transcript for room "
            f"'{room_name}' attached below.\n\n"
            f"{'-' * 50}\n"
            f"MEETING SUMMARY ({room_name.upper()})\n"
            f"{'-' * 50}\n\n"
            f"{summary_text}\n\n"
            f"Best regards,\n"
            f"Automated meeting transcription\n"
        )

        # Both are attached: the transcript is the record, the summary is what
        # people actually read, and the summary is also in the body so it is
        # legible without opening anything.
        attached = _attach(message, transcript_path, f"{stem}_transcript.txt", "plain")
        attached |= _attach(message, summary_path, f"{stem}_summary.md", "markdown")
        if not attached:
            logger.warning("no files were attached to the summary email")

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
