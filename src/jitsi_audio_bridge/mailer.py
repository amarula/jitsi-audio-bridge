"""SMTP delivery of the finished meeting summary and transcript.

Synchronous, like :mod:`jitsi_audio_bridge.ai_client`, and called from the
daemon's worker thread.

No path handling: the transcript is read from the path it is handed.
"""

from __future__ import annotations

import logging
import re
import smtplib
from datetime import datetime
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


def _format_when(started_at: str | None) -> str:
    """A meeting's local date and time, or "" when it is not known.

    A room keeps its name, so every meeting held in one would otherwise carry
    exactly the same subject and heading; the moment it started is what tells
    them apart in a mailbox.  A timestamp without a zone is taken at face
    value, which is what the fallback for a session without a timeline is.
    """
    if not started_at:
        return ""
    try:
        moment = datetime.fromisoformat(str(started_at))
    except ValueError:
        return ""
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    return moment.strftime("%Y-%m-%d %H:%M")


#: The mail's own words.  Only the phrases the daemon writes are here — the
#: summary and its headings come from the model, already in the meeting's
#: language — and a language that is not in the table is mailed in English
#: rather than in nothing.
_MAIL_STRINGS: dict[str, tuple[str, str, str]] = {
    "english": (
        "Meeting Summary & Transcript",
        "MEETING SUMMARY",
        "Please find the automated summary and raw transcript for room "
        "'{room}' attached below.",
    ),
    "italian": (
        "Riepilogo e trascrizione della riunione",
        "RIEPILOGO DELLA RIUNIONE",
        "In allegato il riepilogo automatico e la trascrizione della riunione "
        "'{room}'.",
    ),
    "spanish": (
        "Resumen y transcripción de la reunión",
        "RESUMEN DE LA REUNIÓN",
        "Adjunto encontrará el resumen automático y la transcripción de la "
        "sala '{room}'.",
    ),
    "french": (
        "Résumé et transcription de la réunion",
        "RÉSUMÉ DE LA RÉUNION",
        "Veuillez trouver ci-joint le résumé automatique et la transcription "
        "de la salle '{room}'.",
    ),
    "german": (
        "Zusammenfassung und Transkript des Meetings",
        "ZUSAMMENFASSUNG DES MEETINGS",
        "Im Anhang finden Sie die automatische Zusammenfassung und das "
        "Transkript des Raums '{room}'.",
    ),
}


def mail_strings(language: str | None) -> tuple[str, str, str]:
    """Subject prefix, heading and introduction for *language*.

    The language is the one the summary was written in, so the whole mail
    reads in one voice.  Anything not in the table falls back to English.
    """
    return _MAIL_STRINGS.get(str(language or "").strip().lower(), _MAIL_STRINGS["english"])


def _subject_for(
    room_name: str,
    suffix: str = "",
    started_at: str | None = None,
    language: str | None = None,
) -> str:
    """Build a subject line with any embedded newlines flattened out."""
    flattened = " ".join(str(room_name).split()) or "Meeting"
    subject = f"{mail_strings(language)[0]}: {flattened}"
    when = _format_when(started_at)
    if when:
        subject = f"{subject} ({when})"
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
    started_at: str | None = None,
    language: str | None = None,
) -> bool:
    """Email the summary to *recipients*, attaching the transcript and summary.

    Falls back to the configured fallback recipient when the meeting recorded
    no addresses.  *started_at* is the meeting's own clock, used to tell one
    meeting in a room from the next, and *language* is the one the summary was
    written in, which the subject, the heading and the introduction follow.
    Returns whether the message was handed to the relay.
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
        message["Subject"] = _subject_for(
            room_name, smtp.subject_suffix, started_at, language
        )
        message["From"] = smtp.sender
        message["To"] = ", ".join(targets)
        # The model wrote the summary in the meeting's language; the mail
        # around it is written in the same one, so it reads in one voice.
        _, heading, introduction = mail_strings(language)
        when = _format_when(started_at)
        message.set_content(
            introduction.format(room=room_name)
            + "\n\n"
            + f"{'-' * 50}\n"
            + f"{heading} ({room_name.upper()})"
            + (f" — {when}" if when else "")
            + "\n"
            + f"{'-' * 50}\n\n"
            + f"{summary_text}\n\n"
            + "Best regards,\n"
            + "Automated meeting transcription\n"
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
