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
#: reloading a model, or waiting for the GPU its model shares — and the same
#: request usually works a moment later.  Transcription is one request per
#: speaking turn, so a brief outage used to cost every turn in flight.  A 4xx
#: is never retried: the request itself is wrong.
#:
#: Each wait is twice the one before it — 1s, 2s, 4s, 8s, … — because the
#: thing being waited for is a model being loaded, which takes seconds and
#: then takes them all at once.  The ceiling keeps a long schedule sane.
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CEILING_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 5
_attempts = DEFAULT_MAX_ATTEMPTS

#: Whisper and Ollama usually share a machine, and often a GPU, whose queue
#: holds one request at a time.  Asking for more than the device serves is
#: what produces 5xx answers while one model evicts another — including when
#: two meetings overlap and one summarises while the other transcribes — so
#: the daemon holds its own requests to the same depth.  See
#: [ai] max_concurrent_requests.
_AI_SLOTS = threading.Semaphore(1)


def set_ai_limits(max_concurrent_requests: int, max_attempts: int) -> None:
    """Apply the AI limits from the configuration (see [ai])."""
    global _AI_SLOTS, _attempts
    _AI_SLOTS = threading.Semaphore(max(1, max_concurrent_requests))
    _attempts = max(1, max_attempts)


#: A failure body is the service explaining itself, and it is the only place
#: the reason appears: "model not loaded", "all slots are busy", a CUDA error.
#: It goes into our log, so it has to be bounded and on one line.
_FAILURE_DETAIL_CHARS = 200


def describe_failure(response: requests.Response) -> str:
    """The service's own words about a failure, or an empty string."""
    try:
        body = " ".join((response.text or "").split())
    except Exception:  # noqa: BLE001 - a broken body must not mask the failure
        return ""
    if not body:
        return ""
    if len(body) > _FAILURE_DETAIL_CHARS:
        body = body[:_FAILURE_DETAIL_CHARS] + "…"
    return f": {body}"


def backoff_seconds(attempt: int) -> float:
    """How long to wait before *attempt*, counting from 1.  0 for the first."""
    if attempt < 2:
        return 0.0
    return min(_BACKOFF_CEILING_SECONDS, _BACKOFF_BASE_SECONDS * 2 ** (attempt - 2))


def _post_json(url: str, payload: dict[str, object], endpoint: EndpointConfig) -> dict | None:
    """POST *payload* and return the JSON object, retrying a 5xx answer.

    Returns ``None`` — having logged why — when every attempt failed, which is
    the same contract the callers had before, only with fewer ways to lose a
    meeting to a service that was busy for a second.
    """
    attempts = _attempts
    for attempt in range(1, attempts + 1):
        pause = backoff_seconds(attempt)
        if pause:
            time.sleep(pause)
        try:
            # Held for the whole request: the device is busy until the answer
            # is in, so waiting here is the point.
            with _AI_SLOTS:
                response = requests.post(
                    url, json=payload, verify=endpoint.verify_tls, timeout=endpoint.timeout
                )
        except requests.RequestException as exc:
            logger.warning(
                "request to %s failed (attempt %d/%d): %s", url, attempt, attempts, exc
            )
            continue

        status = response.status_code
        if status >= 500:
            # The service is there and cannot serve right now.
            logger.warning(
                "%s answered %d (attempt %d/%d)%s",
                url,
                status,
                attempt,
                attempts,
                describe_failure(response),
            )
            continue
        if status >= 400:
            # The request itself is wrong: asking it again changes nothing.
            logger.error(
                "%s answered %d; not retrying%s", url, status, describe_failure(response)
            )
            return None

        try:
            body = response.json()
        except ValueError as exc:  # a non-JSON body, on a status that looked fine
            logger.error("%s returned a malformed response: %s", url, exc)
            return None
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
    language: str | None = None,
) -> str:
    """Generate a meeting summary with Ollama, in the transcript's language.

    *language* may be supplied when the caller has already asked — the mail
    around the summary is written in it too.  Returns the summary, or ``""``
    if Ollama could not be reached.
    """
    if language is None:
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
