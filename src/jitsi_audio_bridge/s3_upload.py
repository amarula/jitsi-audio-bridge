# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Upload a finished meeting's Jibri recording to an S3-compatible endpoint.

The video is not made here: Jibri makes it, and only when somebody pressed
Record.  It writes into its own tree — one directory per recording, named after
Jibri's session id — and inside it

    <callName>_<yyyy-MM-dd-HH-mm-ss>.<extension>    the recording itself
    metadata.json                                   the meeting, as Jibri saw it

so the room is recoverable twice over: from the filename Jibri builds out of
the call name, and from the URL in the metadata it leaves beside the
recording.  Matching a transcribed meeting to one of those directories is
therefore a room-name comparison plus a time window — of a room's recordings,
the one that stopped after the meeting started, most recently, is the
meeting's.

Two things this module deliberately will not do: upload a recording that may
still be being written (a file whose mtime is seconds old can be an ffmpeg
still flushing its last frames), and delete a local recording whose upload has
not been confirmed at the endpoint.  Neither mistake would be recoverable.

Nothing here raises.  A meeting's transcript and its mail matter more than its
video, so a failure is a log line and an absent ``video.json`` — which is also
what makes the upload safe to attempt again later.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .audio import TIMELINE_FILENAME, parse_metadata
from .config import Config, S3Config
from .timeline import SessionTimeline, utc_now

logger = logging.getLogger(__name__)

#: The claim a meeting writes once its video is at the endpoint: what was
#: uploaded, from where, and when.  Its presence is what makes the upload
#: idempotent, and what stops a later meeting in the same room from claiming
#: the same recording.
CLAIM_FILENAME = "video.json"

#: One transfer at a time, whatever is asking for one.  Two meetings in the
#: same room can match the same recording — the later meeting's window reaches
#: back far enough to include it — and the claim is written only after the
#: transfer, so without this both would upload it and, with
#: ``delete_after_upload``, the second would take the file out from under the
#: first.
_UPLOAD_LOCK = threading.Lock()

#: The metadata Jibri writes into a recording's directory.
JIBRI_METADATA_NAME = "metadata.json"

#: Where the room is in that file.  Jibri's own name for it is ``meeting_url``;
#: the others are what the same field has been called elsewhere, and reading
#: them costs one dict lookup each.
JIBRI_URL_FIELDS = ("meeting_url", "meetingUrl", "callUrl", "call_url")

#: Containers Jibri may have been told to write —
#: ``jibri.ffmpeg.recording-extension`` decides which, and mp4 is the stock one.
VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".webm", ".mov", ".flv", ".m4v", ".ts"})

#: Jibri builds the filename from the call name and stamps the moment the
#: recording stopped onto it: ``<callName>_2026-10-08-10-11-12.mp4``.
_RECORDING_STAMP = re.compile(r"_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")

#: How often to look again while waiting for a recording that has not been
#: finished yet.
POLL_SECONDS = 15.0

#: A recording cannot have stopped before the meeting began, less this much
#: slack for two clocks that are not the same clock.  It is what separates this
#: meeting's recording from the one the previous meeting in the same room left
#: behind.
RECORDING_START_SLACK_SECONDS = 120.0

#: Characters kept when a room name or filename becomes part of an object key.
#: A key is a path, so ``/`` is not among them: a room called ``a/b`` must not
#: become two levels of somebody's bucket.
_UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

_MAX_KEY_PART = 80


def normalise_name(value: str) -> str:
    """Reduce a room name or a filename to comparable letters.

    Jibri's call name and the bridge's room name reach their recordings by
    different routes — one through a URL, the other through the JVB's framing —
    and the two do not have to agree on case or punctuation to be the same
    room.  What is left is what they cannot disagree about.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _key_part(value: str, fallback: str = "meeting") -> str:
    """Reduce *value* to one safe component of an object key."""
    cleaned = _UNSAFE_KEY_CHARS.sub("_", str(value)).strip("._")
    return (cleaned or fallback)[:_MAX_KEY_PART]


def jibri_room(directory: Path) -> str | None:
    """The room a recording was made in, from Jibri's own metadata.

    Jibri writes the URL it joined with next to the recording, which names the
    room exactly — better than the filename, which it has already reduced to
    something a filesystem tolerates and cut to 125 characters.
    """
    try:
        raw = (directory / JIBRI_METADATA_NAME).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    for field in JIBRI_URL_FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or not value:
            continue
        # The room is the last path segment; a query or fragment is not part
        # of it, and a room name that was percent-encoded arrives here as the
        # name the bridge knows it by.
        node = unquote(urlparse(value).path).strip("/").rpartition("/")[2]
        if node:
            return node
    return None


def recording_room(recording: Path) -> str:
    """Which room a Jibri recording belongs to.

    Jibri's metadata decides when it is there; the filename is the fallback,
    with the timestamp Jibri stamped onto it removed.
    """
    named = jibri_room(recording.parent)
    if named:
        return named
    return _RECORDING_STAMP.sub("", recording.stem)


def iter_recordings(root: Path) -> Iterator[Path]:
    """Every video Jibri has under *root*, in either layout.

    A recording lives at ``<root>/<session>/<file>``, but a Jibri configured to
    write straight into its recordings directory would leave them one level
    up, and looking in both places costs one glob.
    """
    for pattern in ("*/*", "*"):
        try:
            entries = sorted(root.glob(pattern))
        except OSError:
            continue
        for entry in entries:
            if entry.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            try:
                if entry.is_file():
                    yield entry
            except OSError:  # a broken symlink, or a directory in the way
                continue


def find_recording(
    root: Path,
    room_name: str,
    *,
    not_before: float,
    settled_before: float,
    claimed: Iterable[Path] = (),
) -> Path | None:
    """The recording of *room_name* that belongs to the meeting just ended.

    *not_before* is when the meeting started, less a little slack;
    *settled_before* is now, less the time a recording is given to finish
    being written.  Only files in that window count, and of those the most
    recent one wins: a room reused for a second meeting has recordings from
    both, and the meeting that just ended has the later one.
    """
    wanted = normalise_name(room_name)
    if not wanted or not root.is_dir():
        return None
    taken = {Path(path) for path in claimed}

    best: Path | None = None
    best_mtime = float("-inf")
    for candidate in iter_recordings(root):
        if candidate in taken:
            continue
        if normalise_name(recording_room(candidate)) != wanted:
            continue
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if not not_before <= mtime <= settled_before:
            continue
        if mtime > best_mtime:
            best, best_mtime = candidate, mtime
    return best


def describe_recordings(root: Path, not_before: float, settled_before: float) -> str:
    """What was there instead, for the log line that reports a missing video.

    A name that does not match is the likeliest reason for finding nothing, and
    it is invisible unless the rooms that *were* found are written down next to
    the one that was looked for.
    """
    seen: list[str] = []
    for candidate in iter_recordings(root):
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        mark = "" if not_before <= mtime <= settled_before else " (outside the meeting)"
        seen.append(f"{recording_room(candidate)!r}{mark}")
    if not seen:
        return f"no recordings under {root}"
    return "found " + ", ".join(sorted(set(seen)))


def read_claim(path: Path) -> dict[str, Any]:
    """The contents of a ``video.json``, or an empty dict."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def claimed_recordings(recordings_dir: Path) -> frozenset[Path]:
    """Recordings that some earlier meeting has already uploaded.

    Every meeting writes what it uploaded into its own directory, so the
    siblings of the directory being processed are the whole answer.  Without
    this, two meetings in one room — back to back, or both ending within the
    grace period — could each match the same file and upload it twice.
    """
    claimed: set[Path] = set()
    try:
        claims = sorted(recordings_dir.glob(f"*/{CLAIM_FILENAME}"))
    except OSError:
        return frozenset()
    for claim in claims:
        source = read_claim(claim).get("source")
        if isinstance(source, str) and source:
            claimed.add(Path(source))
    return frozenset(claimed)


def object_key(prefix: str, room_name: str, source: Path) -> str:
    """Where a recording goes in the bucket.

    One directory per room, under the configured prefix, keeping Jibri's own
    filename: it is unique per recording already — that timestamp on the end is
    why — and it is the name somebody looking in the bucket will recognise.
    """
    parts = [_key_part(segment, "") for segment in str(prefix).strip("/").split("/")]
    parts.append(_key_part(room_name))
    parts.append(_key_part(source.name, "recording"))
    return "/".join(part for part in parts if part)


def object_url(s3: S3Config, bucket: str, key: str) -> str:
    """A URL for the uploaded object, as far as one can be built.

    Built on the address a *person* would use — ``link_endpoint`` when the
    daemon uploads to somewhere else — because this is the URL written into
    ``video.json`` and, unsigned, the one that goes into a mail.
    """
    endpoint = str(s3.link_base).rstrip("/")
    if s3.path_style:
        return f"{endpoint}/{bucket}/{key}"
    parsed = urlparse(endpoint)
    if not parsed.netloc:
        return f"{endpoint}/{bucket}/{key}"
    return f"{parsed.scheme}://{bucket}.{parsed.netloc}/{key}"


#: What SigV4 will sign: a presigned URL asking for longer than this is refused
#: by the server when somebody clicks it, which is a link that never works.
MAX_LINK_EXPIRY_SECONDS = 604800.0


def presigned_url(s3: S3Config, bucket: str, key: str) -> str | None:
    """A URL that lets whoever holds it download the object, for a while.

    The signature is computed here, from the daemon's own credentials — no
    request is made — and the host in it comes from ``link_endpoint`` when one
    is configured, because the address the daemon uploads to is often not one
    a recipient can reach.

    Returns ``None``, having logged why, when the link cannot be signed.  There
    is deliberately no fallback to the unsigned URL: against a private bucket
    that is a 403 the reader cannot tell from a broken link.
    """
    if s3.link_expiry_seconds <= 0:
        return object_url(s3, bucket, key)

    expires = min(s3.link_expiry_seconds, MAX_LINK_EXPIRY_SECONDS)
    if expires < s3.link_expiry_seconds:
        logger.warning(
            "[s3] link_expiry_seconds is %.0f; signing for %.0f, the most SigV4 allows",
            s3.link_expiry_seconds,
            MAX_LINK_EXPIRY_SECONDS,
        )

    try:
        client = _client(s3, endpoint=s3.link_base)
        return str(
            client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=int(expires),
            )
        )
    except Exception as exc:  # noqa: BLE001 - botocore's error tree is wide
        logger.error(
            "cannot sign a download link for %s/%s: %s; the mail will go without one",
            bucket,
            key,
            exc,
        )
        return None


def _client(s3: S3Config, endpoint: str | None = None) -> Any:
    """A boto3 client for the configured endpoint.

    Imported here rather than at the top of the module so that a deployment
    which does not upload videos does not need boto3 installed at all, and so
    that a missing boto3 is one log line instead of an import error that takes
    the daemon down before it records anything.

    *endpoint* overrides where the client points, which is how a download link
    is signed for the address a recipient uses rather than the one the daemon
    uploads to; signing touches no network either way.
    """
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=(endpoint or s3.endpoint) or None,
        aws_access_key_id=s3.access_key or None,
        aws_secret_access_key=s3.secret_key or None,
        region_name=s3.region or None,
        # False for a self-signed certificate; a CA bundle can also be named
        # through AWS_CA_BUNDLE, which botocore reads itself.
        verify=s3.verify_tls,
        config=BotoConfig(
            # Almost every self-hosted S3-compatible server serves
            # endpoint/bucket/key; AWS proper serves bucket.endpoint/key, and
            # both spellings can be had from most of them.
            s3={"addressing_style": "path" if s3.path_style else "virtual"},
            retries={"max_attempts": 3, "mode": "standard"},
            # Stated rather than left to botocore, which signs a *presigned*
            # URL with the v2 query signature against a custom endpoint
            # (?AWSAccessKeyId=…), and MinIO answers that with
            # SignatureDoesNotMatch.  Uploads would sign v4 by default anyway;
            # the links would not, and a link that looks signed and is refused
            # is the worst of both.
            signature_version="s3v4",
        ),
    )


def _object_size(client: Any, bucket: str, key: str) -> int | None:
    """The size the endpoint reports for an object, or ``None`` if it is not
    there or cannot be asked."""
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - botocore's error tree is wide
        logger.warning("cannot confirm %s/%s at the endpoint: %s", bucket, key, exc)
        return None
    try:
        return int(head["ContentLength"])
    except (KeyError, TypeError, ValueError):
        return None


def write_claim(meeting_dir: Path, claim: dict[str, Any]) -> bool:
    """Record what was uploaded, atomically, beside the transcript."""
    target = meeting_dir / CLAIM_FILENAME
    temporary: str | None = None
    try:
        handle_fd, temporary = tempfile.mkstemp(
            dir=meeting_dir, prefix=".video-", suffix=".tmp"
        )
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(claim, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, target)
    except OSError as exc:
        logger.error("cannot write %s: %s", target, exc)
        if temporary is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
        return False
    return True


def meeting_started_epoch(meeting_dir: Path, timeline: SessionTimeline | None) -> float:
    """When the meeting began, as a timestamp.

    The timeline knows; a directory without one falls back to its own mtime,
    which is later than the meeting started but still inside it — and the
    comparison it feeds carries slack for exactly that.
    """
    if timeline is not None and timeline.started_at:
        try:
            return datetime.fromisoformat(timeline.started_at).timestamp()
        except ValueError:
            pass
    try:
        return meeting_dir.stat().st_mtime
    except OSError:
        return time.time()


@dataclass(frozen=True)
class VideoLink:
    """A recording where somebody reading a mail can fetch it."""

    url: str
    #: When the link stops working, as an ISO timestamp; ``None`` when it does
    #: not expire, which is what a bucket anyone may read gives.
    expires_at: str | None = None


def upload_meeting_video(
    meeting_dir: Path, config: Config, *, wait: bool = True
) -> bool:
    """Upload this meeting's Jibri recording, if it has one.

    Blocking, and minutes long for a long recording.  *wait* controls whether a
    recording that has not been finished yet is waited for — the daemon turns
    it off while shutting down, so that a stop does not sit through it.

    Returns whether a video is at the endpoint, having been uploaded now or
    already.  Never raises.
    """
    s3 = config.s3
    if not s3.enabled:
        return False

    try:
        return _upload_meeting_video(Path(meeting_dir), config, wait=wait)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        logger.exception("could not upload the recording of %s: %s", meeting_dir, exc)
        return False


def prepare_video(
    meeting_dir: Path, config: Config, *, wait_seconds: float
) -> VideoLink | None:
    """Upload the recording before the mail, so the mail can link to it.

    The same work :func:`upload_meeting_video` does, run earlier and for a
    different reason: a link is only worth putting in a mail if it works when
    the mail is read.  Waiting is bounded by *wait_seconds*, and a meeting
    nobody recorded gives up quickly and returns ``None`` rather than holding
    the transcript back.

    Returns the link to put in the mail, or ``None`` — having logged why — when
    there is no recording, the upload failed, or no link could be signed.
    Never raises.
    """
    if not config.s3.enabled:
        return None

    try:
        return _prepare_video(Path(meeting_dir), config, wait_seconds=wait_seconds)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        logger.exception("could not prepare the recording of %s: %s", meeting_dir, exc)
        return None


def _prepare_video(
    meeting_dir: Path, config: Config, *, wait_seconds: float
) -> VideoLink | None:
    """The pre-mail upload, with the caller's guard around it."""
    s3 = config.s3
    claim = read_claim(meeting_dir / CLAIM_FILENAME)
    if claim:
        # A directory processed before, or a session whose upload already ran.
        return _link_from_claim(s3, claim)

    room_name = parse_metadata(meeting_dir)["room_name"]
    recording = _locate(
        meeting_dir, config, room_name, wait_seconds, only_if_pending=True
    )
    if recording is None:
        return None
    with _UPLOAD_LOCK:
        claim = read_claim(meeting_dir / CLAIM_FILENAME)
        if not claim:
            claim = _upload(s3, recording, meeting_dir, room_name) or {}
    return _link_from_claim(s3, claim) if claim else None


def _link_from_claim(s3: S3Config, claim: dict[str, Any]) -> VideoLink | None:
    """The link for an upload that is already there, signed if it needs to be."""
    bucket = claim.get("bucket")
    key = claim.get("key")
    if not isinstance(bucket, str) or not isinstance(key, str) or not bucket or not key:
        logger.warning("%s holds no bucket and key; the mail will have no link", CLAIM_FILENAME)
        return None

    url = presigned_url(s3, bucket, key)
    if url is None:
        return None
    if s3.link_expiry_seconds <= 0:
        return VideoLink(url)
    return VideoLink(url, expires_at=_expiry_moment(s3.link_expiry_seconds))


def _expiry_moment(seconds: float) -> str:
    """When a link signed now stops working, as an ISO timestamp."""
    return datetime.fromtimestamp(
        time.time() + min(seconds, MAX_LINK_EXPIRY_SECONDS), UTC
    ).isoformat(timespec="seconds")


def _locate(
    meeting_dir: Path,
    config: Config,
    room_name: str,
    wait_seconds: float,
    *,
    only_if_pending: bool = False,
) -> Path | None:
    """Find this meeting's recording, waiting up to *wait_seconds* for it."""
    s3 = config.s3
    timeline = SessionTimeline.load(meeting_dir / TIMELINE_FILENAME)
    not_before = (
        meeting_started_epoch(meeting_dir, timeline) - RECORDING_START_SLACK_SECONDS
    )
    return _wait_for_recording(
        s3,
        room_name,
        not_before=not_before,
        claimed=claimed_recordings(config.storage.recordings_dir),
        wait_seconds=wait_seconds,
        only_if_pending=only_if_pending,
    )


def _upload_meeting_video(meeting_dir: Path, config: Config, *, wait: bool) -> bool:
    """The upload, with the caller's guard around it."""
    s3 = config.s3
    if (meeting_dir / CLAIM_FILENAME).exists():
        logger.info("the recording of %s has already been uploaded", meeting_dir.name)
        return True

    room_name = parse_metadata(meeting_dir)["room_name"]
    recording = _locate(meeting_dir, config, room_name, s3.wait_seconds if wait else 0.0)
    if recording is None:
        return False
    with _UPLOAD_LOCK:
        # Two meetings in one room can look at the same recording, and the one
        # that gets there second should find it claimed rather than send it
        # again — or delete it under the first, with delete_after_upload.
        if (meeting_dir / CLAIM_FILENAME).exists():
            return True
        return _upload(s3, recording, meeting_dir, room_name) is not None


def _upload(
    s3: S3Config, recording: Path, meeting_dir: Path, room_name: str
) -> dict[str, Any] | None:
    """Put one recording in the bucket and claim it.  ``None`` if it did not go.

    This is the whole of what an upload is, whichever order it runs in: the
    key, the file's size checked first, the transfer, the size read back, and
    the claim that makes it idempotent — plus the local file's removal, when
    the endpoint has confirmed the copy.
    """
    key = object_key(s3.prefix, room_name, recording)
    try:
        size = recording.stat().st_size
    except OSError as exc:
        logger.error("cannot read %s: %s", recording, exc)
        return None
    if not size:
        logger.warning("%s is empty; not uploading it", recording)
        return None

    logger.info(
        "uploading %s (%.1f MiB) to %s/%s", recording, size / (1 << 20), s3.bucket, key
    )
    try:
        client = _client(s3)
        client.upload_file(str(recording), s3.bucket, key)
    except Exception as exc:  # noqa: BLE001 - botocore's error tree is wide
        logger.error("upload of %s to %s/%s failed: %s", recording, s3.bucket, key, exc)
        return None

    confirm = _object_size(client, s3.bucket, key)
    logger.info(
        "uploaded %s to %s/%s (%s bytes at the endpoint)",
        recording.name,
        s3.bucket,
        key,
        "unconfirmed" if confirm is None else confirm,
    )
    claim = {
        "source": str(recording),
        "bucket": s3.bucket,
        "key": key,
        "url": object_url(s3, s3.bucket, key),
        "size": size,
        "confirmed_size": confirm,
        "room_name": room_name,
        "uploaded_at": utc_now(),
    }
    write_claim(meeting_dir, claim)

    if s3.delete_after_upload:
        _delete_uploaded(recording, size, confirm)
    return claim


def has_recording_since(root: Path, room_name: str, *, not_before: float) -> bool:
    """Whether this room has any recording at all from the meeting onwards.

    Deliberately says nothing about a file being *finished*: it is the question
    "is something coming, or is there nothing to wait for?".

    Jibri's ffmpeg writes straight to the file it names after the room, with no
    ``+faststart`` in the stock command, so a recording in progress is visible
    here as a growing file.  A deployment that added faststart would move the
    file into place only at the end, and this would answer "no" throughout —
    costing the mail its link, not the recording its upload.
    """
    wanted = normalise_name(room_name)
    if not wanted or not root.is_dir():
        return False
    for candidate in iter_recordings(root):
        if normalise_name(recording_room(candidate)) != wanted:
            continue
        try:
            if candidate.stat().st_mtime >= not_before:
                return True
        except OSError:
            continue
    return False


def _wait_for_recording(
    s3: S3Config,
    room_name: str,
    *,
    not_before: float,
    claimed: frozenset[Path],
    wait_seconds: float,
    only_if_pending: bool = False,
) -> Path | None:
    """Look for the meeting's recording, and keep looking for a while.

    The recording does not end when the meeting does — somebody stops it, or
    Jibri stops it a little after — so the file can be missing when the
    transcript is already written.  Both the waiting and the settling window
    are about that: an ffmpeg that is still writing must not be uploaded, and
    an upload that gave up too early is one nobody notices is missing.

    *only_if_pending* ends the wait as soon as there is nothing in this room to
    wait for.  A mail that carries a link waits; a mail must not spend two
    minutes on a meeting nobody recorded, which is most of them.
    """
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while True:
        now = time.time()
        recording = find_recording(
            s3.jibri_dir,
            room_name,
            not_before=not_before,
            settled_before=now - max(0.0, s3.settle_seconds),
            claimed=claimed,
        )
        if recording is not None:
            return recording
        pending = not only_if_pending or has_recording_since(
            s3.jibri_dir, room_name, not_before=not_before
        )
        if not pending or time.monotonic() >= deadline:
            logger.info(
                "no recording of %r in %s; %s",
                room_name,
                s3.jibri_dir,
                describe_recordings(
                    s3.jibri_dir, not_before, now - max(0.0, s3.settle_seconds)
                ),
            )
            return None
        time.sleep(POLL_SECONDS)


def _delete_uploaded(recording: Path, size: int, confirmed: int | None) -> None:
    """Remove the local recording, once the endpoint has confirmed it.

    Only when the endpoint reports back the size that was uploaded, and the
    file has not changed since: deleting the only copy of a meeting because an
    upload half-finished is not a failure this daemon gets to make.
    """
    try:
        current = recording.stat().st_size
    except OSError as exc:
        logger.warning("cannot check %s before removing it: %s", recording, exc)
        return
    if confirmed is None or confirmed != size or current != size:
        logger.warning(
            "keeping %s: the endpoint reported %s bytes for the %d uploaded, and the "
            "file is now %d",
            recording,
            "nothing" if confirmed is None else confirmed,
            size,
            current,
        )
        return
    try:
        recording.unlink()
    except OSError as exc:
        logger.warning("could not remove %s: %s", recording, exc)
        return
    logger.info(
        "removed %s; delete_after_upload is enabled and the endpoint has it (%d bytes)",
        recording,
        confirmed,
    )
