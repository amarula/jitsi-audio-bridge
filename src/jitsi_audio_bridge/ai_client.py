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
_LANGUAGE_SAMPLE_CHARS = 1000


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

    Falls back to English, saying so in the log — unlike the previous silent
    default, which hid every outage behind a plausible-looking answer.
    """
    prompt = (
        "Identify the language of this transcript. Respond with ONLY the English "
        f"name of the language:\n{transcript_text[:_LANGUAGE_SAMPLE_CHARS]}"
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
        detected = (response.json().get("response") or "").strip().strip(".")
    except (requests.RequestException, ValueError, AttributeError) as exc:
        logger.warning("language detection failed, assuming English: %s", exc)
        return "English"

    if not detected:
        logger.warning("language detection returned nothing, assuming English")
        return "English"
    return detected


def generate_summary(transcript_text: str, room_name: str, endpoint: OllamaConfig) -> str:
    """Generate a meeting summary with Ollama, in the transcript's language.

    Returns the summary, or ``""`` if Ollama could not be reached.
    """
    language = detect_language(transcript_text, endpoint)
    prompt = f"""You are an executive assistant. Write a meeting summary strictly in {language}.

Room: {room_name}

Transcript:
{transcript_text}

Format:
- Executive Summary
- Key Discussion Points
- Action Items & Assigned Owners
"""
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
