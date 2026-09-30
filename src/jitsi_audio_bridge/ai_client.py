"""HTTP clients for the local Whisper and Ollama services.

These functions are synchronous on purpose.  The daemon calls them from a
worker thread via ``asyncio.to_thread`` so the event loop stays free while a
transcription runs.

Failures are logged and degrade to an empty result rather than propagating:
one unreachable participant, or a transient Ollama hiccup, should cost a
meeting its summary only if it really cannot be recovered — not take down the
connection handler that called it.

No path handling beyond reading the file it is handed.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import requests

from .config import EndpointConfig, OllamaConfig

logger = logging.getLogger(__name__)

#: Only this much of a transcript is sent when asking Ollama for the language;
#: the answer is stable long before the end of the meeting.
_LANGUAGE_SAMPLE_CHARS = 1500


def transcribe_audio(wav_path: str | Path, endpoint: EndpointConfig) -> str:
    """Transcribe one WAV file via the Whisper service.

    Returns the transcript, or ``""`` if the file could not be read or the
    service failed.

    Note that the whole file is base64-encoded into memory and into the JSON
    request body; see REVIEW.md for the memory implications on long meetings.
    """
    path = Path(wav_path)
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        logger.error("cannot read %s: %s", path, exc)
        return ""

    payload = {"audio_base64": encoded, "filename": path.name}

    try:
        response = requests.post(
            endpoint.url, json=payload, verify=endpoint.verify_tls, timeout=endpoint.timeout
        )
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:
        logger.error("whisper request for %s failed: %s", path.name, exc)
        return ""
    except ValueError as exc:  # a non-JSON body
        logger.error("whisper returned a malformed response for %s: %s", path.name, exc)
        return ""

    if not isinstance(body, dict):
        logger.error("whisper returned %s, expected a JSON object", type(body).__name__)
        return ""
    return (body.get("text") or "").strip()


def detect_language(transcript_text: str, endpoint: OllamaConfig) -> str:
    """Ask Ollama which language a transcript is written in.

    Deliberately a separate pass with no other instructions in the prompt: a
    model asked to summarise *and* name the language tends to answer in the
    language it is summarising.

    Falls back to English, saying so in the log — unlike a silent default,
    which hides every outage behind a plausible-looking answer.
    """
    # Explicit newlines rather than a triple-quoted block: the prompt text is
    # reproduced exactly, and no source line has to be unreasonably long.
    prompt = (
        "Identify the primary language spoken in the following transcript text.\n"
        "Return ONLY the English name of the language "
        "(for example: English, Italian, French, German, Spanish).\n"
        "Do NOT write explanations. Do NOT include quotes or punctuation.\n"
        "\n"
        "Transcript sample:\n"
        f"{transcript_text[:_LANGUAGE_SAMPLE_CHARS]}\n"
    )
    payload = {
        "model": endpoint.model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        response = requests.post(
            endpoint.url, json=payload, verify=endpoint.verify_tls, timeout=endpoint.timeout
        )
        response.raise_for_status()
        detected = (response.json().get("response") or "").strip().strip(".").strip('"')
    except (requests.RequestException, ValueError, AttributeError) as exc:
        logger.warning("language detection failed, assuming English: %s", exc)
        return "English"

    if not detected:
        logger.warning("language detection returned nothing, assuming English")
        return "English"
    logger.info("detected transcript language: %s", detected)
    return detected


def build_summary_prompt(transcript_text: str, room_name: str, participants: list[str],
                         language: str) -> str:
    """Build the summarisation prompt.

    Two rules carry most of the weight. The speaker-attribution rule stops the
    model from flattening the ``[Name]:`` tags into unattributed prose, which
    is the difference between minutes and a wall of text. The language rule is
    repeated against each section heading, because models otherwise translate
    the body but leave the headings in English.
    """
    participant_list = ", ".join(participants) if participants else "Not specified"
    return (
        f"You are an executive assistant. Write the meeting summary strictly in {language}.\n"
        "\n"
        "STRICT RULES FOR SPEAKER ANNOTATION:\n"
        "- The transcript contains speaker annotations like [Speaker Name]: ...\n"
        "- You MUST attribute discussion points, proposals, and action items directly "
        "to the correct speaker named in the transcript tags.\n"
        "\n"
        "STRICT LANGUAGE RULE:\n"
        f"- Target Language: {language.upper()}\n"
        f"- Write 100% of the output in {language}.\n"
        "- All section titles, headers, bullet points, and descriptions MUST be in "
        f"{language}.\n"
        "\n"
        "CONTEXT:\n"
        f"- Meeting Room / Topic: {room_name}\n"
        f"- Known Participants: {participant_list}\n"
        "\n"
        f"REQUIRED OUTPUT FORMAT (All headings MUST be translated into {language}):\n"
        f"- [Header for Executive Summary in {language}]: 2-3 concise sentences "
        "detailing purpose and core result.\n"
        f"- [Header for Key Discussion Points in {language}]: Bullet points covering "
        "key arguments, topics, and decisions attributed to speakers.\n"
        f"- [Header for Action Items & Decisions in {language}]: Bullet points "
        "explicitly listing assigned tasks and who agreed to do them.\n"
        "\n"
        "Meeting Transcript:\n"
        f"{transcript_text}\n"
    )


def generate_summary(
    transcript_text: str,
    room_name: str,
    participants: list[str],
    endpoint: OllamaConfig,
) -> str:
    """Generate a meeting summary with Ollama, in the transcript's language.

    Returns the summary, or ``""`` if Ollama could not be reached.
    """
    language = detect_language(transcript_text, endpoint)
    prompt = build_summary_prompt(transcript_text, room_name, participants, language)
    payload = {
        "model": endpoint.model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        response = requests.post(
            endpoint.url, json=payload, verify=endpoint.verify_tls, timeout=endpoint.timeout
        )
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:
        logger.error("ollama request failed: %s", exc)
        return ""
    except ValueError as exc:
        logger.error("ollama returned a malformed response: %s", exc)
        return ""

    if not isinstance(body, dict):
        logger.error("ollama returned %s, expected a JSON object", type(body).__name__)
        return ""
    return (body.get("response") or "").strip()
