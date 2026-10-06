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
import contextlib
import logging
import threading
import time
from pathlib import Path

import requests

from .config import EndpointConfig, OllamaConfig

logger = logging.getLogger(__name__)

#: Only this much of a transcript is sent when asking Ollama for the language;
#: the answer is stable long before the end of the meeting.
_LANGUAGE_SAMPLE_CHARS = 1500

#: A 5xx answer means the service is there but not serving — restarting,
#: reloading a model, momentarily overloaded — and the same request usually
#: works a moment later.  Transcription is now one request per speaking turn,
#: so a brief outage used to cost every turn in flight.  A 4xx is never
#: retried: the request itself is wrong.
_ATTEMPT_BACKOFF_SECONDS = (1.0, 3.0)

#: Whisper and Ollama usually share a machine, and often a GPU: a model loaded
#: by one can starve the other, which then answers 5xx until its own model is
#: back.  With two sessions in flight — one summarising while the other
#: transcribes — the daemon would do that to itself, so by default it asks one
#: service at a time.  See [ai] serialize_requests.
_AI_LOCK = threading.Lock()
_serialize_requests = True


def serialize_requests(enabled: bool) -> None:
    """Set whether AI requests are held one at a time (see [ai])."""
    global _serialize_requests
    _serialize_requests = enabled


def _post_json(url: str, payload: dict[str, object], endpoint: EndpointConfig) -> dict | None:
    """POST *payload* and return the JSON object, retrying a 5xx answer.

    Returns ``None`` — having logged why — when every attempt failed, which is
    the same contract the callers had before, only with fewer ways to lose a
    meeting to a service that was busy for a second.
    """
    attempts = len(_ATTEMPT_BACKOFF_SECONDS) + 1
    guard = _AI_LOCK if _serialize_requests else contextlib.nullcontext()
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            time.sleep(_ATTEMPT_BACKOFF_SECONDS[attempt - 2])
        try:
            with guard:
                response = requests.post(
                    url, json=payload, verify=endpoint.verify_tls, timeout=endpoint.timeout
                )
            if response.status_code >= 500:
                logger.warning(
                    "%s answered %d (attempt %d/%d)",
                    url,
                    response.status_code,
                    attempt,
                    attempts,
                )
                continue
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            logger.warning(
                "request to %s failed (attempt %d/%d): %s", url, attempt, attempts, exc
            )
            continue
        except ValueError as exc:  # a non-JSON body
            logger.warning(
                "%s returned a malformed response (attempt %d/%d): %s",
                url,
                attempt,
                attempts,
                exc,
            )
            continue
        if not isinstance(body, dict):
            logger.error("%s returned %s, expected a JSON object", url, type(body).__name__)
            return None
        return body

    logger.error("giving up on %s after %d attempt(s)", url, attempts)
    return None


def transcribe_audio(wav_path: str | Path, endpoint: EndpointConfig) -> str:
    """Transcribe one WAV file via the Whisper service.

    Returns the transcript, or ``""`` if the file could not be read or the
    service failed.

    The request format is fixed by the service: the audio must be base64 in a
    JSON field named ``audio_base64``, alongside ``filename``. That rules out
    ``multipart/form-data`` and means the whole file has to be resident at
    once. See REVIEW.md for the measured memory cost on long recordings and the
    ways to reduce it without changing this contract.
    """
    path = Path(wav_path)
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        logger.error("cannot read %s: %s", path, exc)
        return ""

    payload = {"audio_base64": encoded, "filename": path.name}

    body = _post_json(endpoint.url, payload, endpoint)
    if body is None:
        logger.error("whisper did not transcribe %s", path.name)
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

    body = _post_json(endpoint.url, payload, endpoint)
    if body is None:
        logger.warning("language detection failed, assuming English")
        return "English"
    detected = str(body.get("response") or "").strip().strip(".").strip('"')

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

    body = _post_json(endpoint.url, payload, endpoint)
    if body is None:
        logger.error("ollama produced no summary")
        return ""

    if not isinstance(body, dict):
        logger.error("ollama returned %s, expected a JSON object", type(body).__name__)
        return ""
    return (body.get("response") or "").strip()
