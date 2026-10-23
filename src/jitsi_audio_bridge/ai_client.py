# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
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
import re
import threading
import time
import unicodedata
from pathlib import Path

import requests

from .config import EndpointConfig, OllamaConfig

logger = logging.getLogger(__name__)

#: How much of a transcript is shown when asking Ollama for the language, per
#: slice.  The whole of a short meeting; for a long one, this much from the
#: start, the middle and the end.
#:
#: Only the start was read once, on the reasoning that the answer settles early
#: -- and the start of a meeting is where speech recognition invents lines over
#: silence, in languages nobody in the call was speaking.  A meeting that was
#: 90% English could be judged on the part of it that was not, and the answer
#: then told the correction pass which language to keep the transcript in.
_LANGUAGE_SAMPLE_CHARS = 1500
_LANGUAGE_SAMPLE_SLICES = 3

#: How much of the letters must be in one script before it is stated to the
#: model as a fact about the transcript.  Below it the text really is a mix,
#: and saying so would be the model's job, not this one's.
_SCRIPT_SHARE_OF_TRUTH = 0.75

#: ``[00:00:12] speaker: `` and ``[speaker]: `` -- the tags the transcript is
#: rendered with, which are not what the question is about.
_TRANSCRIPT_OFFSET = re.compile(r"^\s*\[\d[\d:.]*\]\s*")
_TRANSCRIPT_SPEAKER = re.compile(r"^(?:\[[^\]\n]{1,60}\]|[^:\n]{1,60}):\s*")

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


#: Scripts that are the same writing system for this purpose.  A summary is
#: allowed to use more kana than the meeting did; it is not allowed to arrive
#: in Chinese characters when the meeting was written in Latin letters.
_SCRIPT_FAMILY = {
    "CJK": "cjk",
    "HIRAGANA": "cjk",
    "KATAKANA": "cjk",
    "HANGUL": "cjk",
    "BOPOMOFO": "cjk",
}


def script_family(script: str | None) -> str | None:
    """The writing system a script belongs to, coarse enough to compare."""
    if script is None:
        return None
    return _SCRIPT_FAMILY.get(script, script.lower())


def came_back_in_another_script(output: str, source: str) -> bool:
    """Whether a model's answer is written in a different script than its input.

    The last thing that was not checked: a summary or a corrected chunk was
    taken as written, and a model that answered in Chinese to an English
    question produced a Chinese mail -- with every other link in the chain
    saying English.
    """
    produced = dominant_script(output)
    original = dominant_script(source)
    if produced is None or original is None:
        return False
    if produced[1] < _SCRIPT_SHARE_OF_TRUTH or original[1] < _SCRIPT_SHARE_OF_TRUTH:
        return False
    return script_family(produced[0]) != script_family(original[0])


def language_sample(transcript_text: str) -> str:
    """What to show the model when asking which language a meeting is in.

    The speaker tags are dropped, so the budget is spent on what was said, and
    a long meeting is sampled from three places rather than from its beginning
    alone.  Gaps are marked, because three slices read as one text otherwise.
    """
    spoken = "\n".join(
        _TRANSCRIPT_SPEAKER.sub("", _TRANSCRIPT_OFFSET.sub("", line))
        for line in transcript_text.splitlines()
    ).strip()

    budget = _LANGUAGE_SAMPLE_CHARS
    if len(spoken) <= budget * _LANGUAGE_SAMPLE_SLICES:
        return spoken

    starts = [
        0,
        (len(spoken) - budget) // 2,
        len(spoken) - budget,
    ]
    return "\n[...]\n".join(spoken[start:start + budget] for start in starts)


#: What the language name of a script looks like, for the question that says
#: which one the transcript is written in.  The prefix is unicodedata's, so a
#: script that is not listed here is still described, in lower case.
_SCRIPT_NAMES = {
    "LATIN": "the Latin alphabet",
    "CYRILLIC": "the Cyrillic alphabet",
    "GREEK": "the Greek alphabet",
    "ARABIC": "the Arabic script",
    "HEBREW": "the Hebrew script",
    "DEVANAGARI": "the Devanagari script",
    "HIRAGANA": "Japanese kana",
    "KATAKANA": "Japanese kana",
    "HANGUL": "the Korean alphabet",
    "CJK": "Chinese characters",
    "THAI": "the Thai script",
    "ARMENIAN": "the Armenian script",
    "GEORGIAN": "the Georgian script",
}


def script_of(char: str) -> str | None:
    """The script a letter belongs to, from its Unicode name.

    The name is the only reliable way to ask this of a single character without
    a table of thousands of ranges, and it is what the census below is built
    from.
    """
    if not char.isalpha():
        return None
    name = unicodedata.name(char, "")
    if not name:
        return None
    # "CJK UNIFIED IDEOGRAPH-4E2D" has no script word of its own.
    return name.split(" ", 1)[0]


def script_census(text: str) -> dict[str, int]:
    """How many letters of the text are written in each script."""
    counts: dict[str, int] = {}
    for char in text:
        script = script_of(char)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
    return counts


def dominant_script(text: str) -> tuple[str, float] | None:
    """The script most of the letters are in, and its share of them.

    ``None`` when there are no letters to count.  Speech recognition writes in
    the script of the language it *thinks* it heard, so a transcript with a
    run of Chinese in it is a transcript with Chinese characters in it -- and
    counting them is the one part of this that cannot be argued with.
    """
    counts = script_census(text)
    total = sum(counts.values())
    if not total:
        return None
    script, count = max(counts.items(), key=lambda item: item[1])
    return script, count / total


def detect_language(transcript_text: str, endpoint: OllamaConfig) -> str:
    """Ask Ollama which language a transcript is written in.

    Deliberately a separate pass with no other instructions in the prompt: a
    model asked to summarise *and* name the language tends to answer in the
    language it is summarising.

    Falls back to English, saying so in the log — unlike a silent default,
    which hides every outage behind a plausible-looking answer.
    """
    sample = language_sample(transcript_text)
    census = script_census(sample)
    dominant = dominant_script(sample)
    if dominant is not None:
        logger.info(
            "transcript scripts: %s",
            ", ".join(
                f"{script.lower()} {count * 100 / max(1, sum(census.values())):.0f}%"
                for script, count in sorted(census.items(), key=lambda item: -item[1])
            ),
        )

    prompt = build_language_prompt(sample, dominant)
    detected = _ask_language(endpoint, prompt)
    if _contradicts_the_script(detected, dominant):
        # The letters are a fact and the answer is not: a transcript that is
        # 99% Latin is not Chinese, however insistently one of its artefacts
        # is.  Asked once more with that said out loud, then given up on.
        logger.warning(
            "the model said %r, which the transcript's script does not allow", detected
        )
        detected = _ask_language(
            endpoint, build_language_prompt(sample, dominant, strict=True)
        )
        if _contradicts_the_script(detected, dominant):
            logger.warning(
                "the model said %r twice, which the transcript's script does not "
                "allow; assuming English",
                detected,
            )
            return "English"

    if not detected:
        logger.warning("language detection returned nothing, assuming English")
        return "English"
    # A language is a word or two.  Anything else is an answer to a question
    # the model decided to rewrite -- "the transcript is written in ..." is
    # still usable, a paragraph is not, and a language that is a sentence
    # would be handed to the summariser as the one to write the meeting in.
    if len(detected) > 40 or any(mark in detected for mark in ".\n!?"):
        logger.warning(
            "language detection answered %.60r, which is not a language; "
            "assuming English",
            detected,
        )
        return "English"
    logger.info("detected transcript language: %s", detected)
    return detected


def build_language_prompt(
    sample: str, dominant: tuple[str, float] | None, *, strict: bool = False
) -> str:
    """The question, with what was counted put to the model as a fact.

    ``strict`` is the second attempt, after an answer that the letters do not
    allow: it names the script and rules the answer out explicitly.
    """
    lines = [
        "Identify the primary language spoken in the following transcript text.",
        "Judge by the language most of the text is written in. A few lines in "
        "some other language, or single words in one, are transcription "
        "artefacts: they are what speech recognition invents over silence, and "
        "they are not the language of the conversation.",
    ]
    if dominant is not None and dominant[1] >= _SCRIPT_SHARE_OF_TRUTH:
        script = _SCRIPT_NAMES.get(dominant[0], dominant[0].lower())
        lines.append(
            f"Most of this text ({dominant[1] * 100:.1f}% of its letters) is written "
            f"in {script}."
        )
        if strict:
            lines.append(
                f"The transcript is therefore NOT written in a language that uses "
                f"another script. {script.capitalize()} is a fact about the text; "
                f"name the language it is in."
            )
    lines.extend(
        [
            "Return ONLY the English name of the language "
            "(for example: English, Italian, French, German, Spanish).",
            "Do NOT write explanations. Do NOT include quotes or punctuation.",
            "",
            "Transcript sample:",
            sample,
            "",
        ]
    )
    return "\n".join(lines)


#: Languages that cannot be written in the Latin alphabet, so that an answer
#: naming one of them against a Latin census is visibly wrong.  Only the ones
#: a transcription artefact is likely to produce are listed; the list decides
#: whether to ask again, not which language the meeting was in.
_NOT_LATIN = frozenset(
    name.lower()
    for name in (
        "Chinese", "Mandarin", "Cantonese", "Japanese", "Korean", "Russian",
        "Ukrainian", "Bulgarian", "Serbian", "Macedonian", "Belarusian",
        "Arabic", "Persian", "Farsi", "Urdu", "Hebrew", "Hindi", "Bengali",
        "Tamil", "Telugu", "Thai", "Greek", "Georgian", "Armenian",
    )
)


def _contradicts_the_script(
    detected: str, dominant: tuple[str, float] | None
) -> bool:
    """Whether an answer names a language the counted script cannot write."""
    if dominant is None or dominant[0] != "LATIN":
        return False
    if dominant[1] < _SCRIPT_SHARE_OF_TRUTH:
        return False
    return detected.strip().lower() in _NOT_LATIN


def _ask_language(endpoint: OllamaConfig, prompt: str) -> str:
    """One question, one trimmed answer -- or ``""`` if it could not be asked."""
    payload = {
        "model": endpoint.model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    body = _post_json(endpoint.url, payload, endpoint)
    if body is None:
        logger.warning("language detection failed, assuming English")
        return ""
    return str(body.get("response") or "").strip().strip(".").strip('"')


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
        "STRICT CONTENT RULE:\n"
        "- Report only what the transcript says. Do not invent examples, code, "
        "quotations or decisions that are not in it, and do not illustrate a "
        "point with something the speakers did not mention.\n"
        "\n"
        "STRICT LANGUAGE RULE:\n"
        f"- Target Language: {language.upper()}\n"
        "- The transcript may contain a few lines or single words in another "
        "language. Those are transcription artefacts, not something anybody "
        "said in the meeting: do not write the summary in their language and do "
        "not report them as discussion.\n"
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
        # Structure only, and phrased so it cannot be read as a language rule:
        # the mail lays these sections out, and a heading it cannot recognise
        # is a section it cannot number.  The heading's wording is still the
        # model's, in the meeting's language — only the marker is fixed.
        "FORMATTING (this is about structure, not language — keep answering in "
        f"{language}):\n"
        '- Begin each section with a markdown heading: "## ", then that heading.\n'
        '- Write every point as a markdown bullet: "- ", then the point.\n'
        "\n"
        "Meeting Transcript:\n"
        f"{transcript_text}\n"
        "\n"
        # Last, and after the transcript, on purpose: a meeting about software
        # is full of English words whatever language it is held in, and a
        # model answers in the language of what it has just read.  Told once
        # at the top, this instruction loses to the transcript.
        f"REMINDER — answer in {language.upper()}. Write every heading, bullet and "
        f"sentence of your answer in {language}: the transcript is in {language} too, "
        "and the English words inside it are technical terms to keep as they are, not "
        f"a language to answer in. Do not reply in English.\n"
    )


#: A correction pass is one Ollama call per this many characters.  A long
#: meeting does not fit a model's context, and a corrected transcript that
#: quietly lost its second half would be worse than an uncorrected one.
CORRECTION_CHUNK_CHARS = 6000


def build_correction_prompt(chunk: str, language: str) -> str:
    """Ask for the transcript back, repaired — not summarised.

    Speech recognition mishears words, drops endings and leaves punctuation
    out; the model is good at repairing that, and bad at resisting the urge to
    improve on what was said.  The instructions spend their words on the
    second risk: keep the tags, keep the order, keep everything, invent
    nothing.
    """
    return (
        "You are editing a meeting transcript for grammar and wording. "
        f"The conversation is mostly in {language}.\n"
        "\n"
        "RULES:\n"
        "- Keep every speaker tag exactly as it is, at the start of its line: "
        "[Name]: ...\n"
        "- Keep every line, in the order it appears. Do not merge, split, "
        "reorder, add or remove anything.\n"
        "- Repair grammar, punctuation and words that were clearly misheard, "
        "using the context of the conversation.\n"
        "- Where the intended word is not clear, leave what is written alone "
        "rather than inventing a replacement.\n"
        "- Do not summarise, do not comment, do not add headings.\n"
        f"- Repair each line in the language it is written in. Never translate a "
        f"line, into {language} or out of it; a line in another language is left "
        "as it is.\n"
        "\n"
        "Reply with the corrected transcript only.\n"
        "\n"
        f"{chunk}"
    )


def split_for_correction(text: str, limit: int = CORRECTION_CHUNK_CHARS) -> list[str]:
    """Split a transcript into chunks no longer than *limit* characters.

    Cut on line boundaries so a speaker's turn is never split in half — the
    model is told to keep every line intact, and it cannot do that with half
    of one.  A single line longer than the limit is cut anyway: it is better
    corrected in pieces than not at all.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        while len(line) > limit:
            # A pathological line (no newlines for thousands of characters).
            if current:
                chunks.append("\n".join(current))
                current, size = [], 0
            chunks.append(line[:limit])
            line = line[limit:]
        if current and size + len(line) + 1 > limit:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def correct_transcript(
    transcript_text: str, endpoint: OllamaConfig, language: str | None = None
) -> str:
    """Return the transcript with grammar and wording repaired.

    Returns ``""`` — having logged why — when any chunk could not be
    corrected, so the caller falls back to the transcript as it was rather
    than mailing half a meeting.
    """
    chunks = split_for_correction(transcript_text)
    if not chunks:
        return ""
    if language is None:
        language = detect_language(transcript_text, endpoint)

    corrected: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        payload = {
            "model": endpoint.model,
            "prompt": build_correction_prompt(chunk, language),
            "stream": False,
            "options": {"temperature": 0.1},
        }
        body = _post_json(endpoint.url, payload, endpoint)
        if body is None:
            logger.error("correction of chunk %d/%d failed", index, len(chunks))
            return ""
        piece = str(body.get("response") or "").strip()
        if not piece:
            logger.error("correction of chunk %d/%d produced nothing", index, len(chunks))
            return ""
        if came_back_in_another_script(piece, chunk):
            # Not a correction: the model answered in another language, and
            # what it wrote is not what was said.  The raw chunk is, so it is
            # the one that is kept -- the transcript is never made worse by
            # this pass.
            logger.warning(
                "correction of chunk %d/%d came back in another script; "
                "keeping that chunk as transcribed",
                index,
                len(chunks),
            )
            piece = chunk
        corrected.append(piece)

    logger.info("corrected %d chunk(s) of the transcript", len(corrected))
    return "\n\n".join(corrected)


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

    summary = (body.get("response") or "").strip()
    if came_back_in_another_script(summary, transcript_text):
        logger.warning(
            "the summary came back in another script than the transcript; asking again"
        )
        retry = _post_json(
            endpoint.url,
            {
                "model": endpoint.model,
                "prompt": prompt
                + "\n\nThe previous attempt was written in the wrong language. "
                f"Write this summary in {language}, and in no other language.",
                "stream": False,
                "options": {"temperature": 0.0},
            },
            endpoint,
        )
        if isinstance(retry, dict) and (retry.get("response") or "").strip():
            second = str(retry["response"]).strip()
            if not came_back_in_another_script(second, transcript_text):
                return second
            # A model that will not write in the language it was asked for may
            # still translate into it: that is a different task, and the one
            # these models are best at.  The mail is worth the extra call.
            translated = _translate(endpoint, second, language, expected=transcript_text)
            if translated:
                logger.warning(
                    "the summary was written in another language; the %s it carries "
                    "is a translation",
                    language,
                )
                return translated
            logger.error(
                "the summary is not in %s and %s will not write it so, even as a "
                "translation; it is being mailed as it came -- the model is the "
                "thing to change",
                language,
                endpoint.model,
            )
            return second
        logger.error("ollama produced no summary on the second attempt")
    return summary


def _translate(
    endpoint: OllamaConfig, text: str, language: str, *, expected: str
) -> str:
    """The text, in *language* -- or ``""`` if the model would not.

    *expected* is what the language looks like: the transcript, whose script
    the translation has to be written in.  Comparing the translation with the
    text being translated would call every translation a failure.
    """
    body = _post_json(
        endpoint.url,
        {
            "model": endpoint.model,
            "prompt": (
                f"Translate the following text into {language}. "
                "Keep its structure, its headings and its list items. "
                "Reply with the translation only, in "
                f"{language} and in no other language.\n\n{text}"
            ),
            "stream": False,
            "options": {"temperature": 0.0},
        },
        endpoint,
    )
    if not isinstance(body, dict):
        return ""
    translated = str(body.get("response") or "").strip()
    if not translated or came_back_in_another_script(translated, expected):
        return ""
    return translated
