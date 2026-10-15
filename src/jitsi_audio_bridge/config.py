# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
"""Configuration loading for the Jitsi audio bridge.

Values are resolved from three sources, in increasing order of precedence:

1. the built-in defaults in :data:`DEFAULTS` (so the daemon runs with no file);
2. a ``config.ini`` file;
3. the process environment, via ``JITSI_AUDIO_BRIDGE_<SECTION>_<KEY>``.

The environment layer exists so that secrets such as the SMTP password can be
supplied by a systemd ``EnvironmentFile`` and never written to disk in
``config.ini``.

This is the only module in the package that reads files or the environment.
"""

from __future__ import annotations

import configparser
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Prefix for environment overrides, e.g. ``JITSI_AUDIO_BRIDGE_SMTP_PASSWORD``.
ENV_PREFIX = "JITSI_AUDIO_BRIDGE_"

#: Environment variable pointing at a config file, equivalent to ``--config``.
ENV_CONFIG_PATH = ENV_PREFIX + "CONFIG"

#: Locations searched, in order, when no config file is given explicitly.
SEARCH_PATHS: tuple[Path, ...] = (
    Path("config.ini"),
    Path("/etc/jitsi-audio-bridge/config.ini"),
)

#: Fallback values, as strings so that every source is parsed identically.
DEFAULTS: dict[str, dict[str, str]] = {
    "server": {
        "host": "127.0.0.1",
        "port": "8080",
    },
    "storage": {
        "recordings_dir": "/srv/recordings",
        "cleanup_after_send": "false",
        "session_metadata_dir": "",
        "capture_timeline": "true",
        "session_grace_seconds": "60",
    },
    "transcript": {
        "interleave": "true",
        "merge_gap_seconds": "1.0",
    },
    "ai": {
        "max_concurrent_requests": "1",
        "max_attempts": "5",
    },
    "whisper": {
        "url": "https://whisper.omnia.amarulasolutions.com/transcribe-b64",
        "timeout": "600",
        "verify_tls": "true",
    },
    "ollama": {
        "url": "https://ollama.omnia.amarulasolutions.com/api/generate",
        "model": "qwen2.5:14b-instruct",
        "timeout": "600",
        "verify_tls": "true",
        "correct_transcript": "false",
    },
    "smtp": {
        "host": "127.0.0.1",
        "port": "25",
        "user": "",
        "password": "",
        "sender": "no-reply@amarulasolutions.com",
        "fallback_recipient": "admin@omnia.amarulasolutions.com",
        "use_starttls": "true",
        # Appended to the subject line, e.g. " - Amarula Solutions".
        "subject_suffix": "",
    },
    "s3": {
        # Empty endpoint or bucket turns the whole feature off.
        "endpoint": "",
        "bucket": "",
        "prefix": "",
        "region": "us-east-1",
        # Empty credentials fall back to botocore's own resolution: the
        # environment, a shared credentials file, or an instance profile.
        "access_key": "",
        "secret_key": "",
        "path_style": "true",
        # Set to false only for an endpoint with a self-signed certificate.
        "verify_tls": "true",
        # Where Jibri writes its recordings.
        "jibri_dir": "/srv/jibri-recordings",
        # How long to wait for the recording to appear after the meeting ends.
        "wait_seconds": "900",
        # How long a recording must have been untouched before it is uploaded.
        "settle_seconds": "30",
        "delete_after_upload": "false",
        # Put the recording's link in the summary mail, and how long the mail
        # waits for the recording before giving up and sending without one.
        "link_in_mail": "false",
        "link_wait_seconds": "120",
        # How long a signed link works.  Seven days is the most SigV4 allows;
        # 0 means "do not sign", for a bucket that is readable without it.
        "link_expiry_seconds": "604800",
        # The address recipients reach the bucket by, when that is not the
        # one this daemon uploads to.  Empty means the same address.
        "link_endpoint": "",
    },
    "mail": {
        # Send an HTML part alongside the plain-text body.  The text part is
        # sent either way: it is what a client that refuses HTML falls back to.
        "html": "true",
        # A stylesheet to use instead of the packaged one, which is how a
        # deployment re-brands the mail.  Empty uses the stock theme in
        # jitsi_audio_bridge/templates/summary_email.css.
        "stylesheet": "",
        # The closing line.  It is the one place the mail names whoever runs
        # the deployment, so it is configuration rather than a constant.
        "footer": (
            "Generated entirely on-premise on our private Debian infrastructure. "
            "No cloud dependency."
        ),
        # How many speaking turns the mail shows before deferring to the
        # attached transcript.
        "preview_turns": "10",
    },
}


class ConfigError(Exception):
    """Raised when a configuration value is missing or malformed."""


@dataclass(frozen=True)
class ServerConfig:
    """Where the WebSocket server listens."""

    host: str
    port: int


@dataclass(frozen=True)
class StorageConfig:
    """Where per-meeting recordings and transcripts are written."""

    recordings_dir: Path
    #: Delete the audio, transcript and summary once the email has gone out.
    #: Off by default: deleting a recording is irreversible, so a
    #: misconfiguration should not be able to destroy a meeting.
    cleanup_after_send: bool
    #: Where a companion service (a Prosody module) drops per-meeting metadata
    #: for stock-Jitsi sessions, keyed by meeting id.  ``None`` disables it:
    #: the JVB's framing carries no names, addresses or room name, so those
    #: sessions then run entirely on the defaults.
    session_metadata_dir: Path | None = None
    #: Write ``timeline.json`` for sessions fed by the JVB's media export: who
    #: spoke when, on the session's own clock.  It can only be captured while
    #: the meeting is running, so switching it off means those transcripts are
    #: never interleaved, however they are processed later.
    capture_timeline: bool = True
    #: How long a session may stay silent before the meeting counts as over.
    #: A connection ending is not the meeting ending — the JVB closes one
    #: export and opens another for the same conference — so post-processing
    #: waits this long, and a connection arriving sooner cancels it.  Zero
    #: transcribes and mails as soon as a connection closes, which is how a
    #: meeting interrupted mid-way gets mailed in parts.
    session_grace_seconds: float = 60.0


@dataclass(frozen=True)
class AiConfig:
    """How the two AI services are used together."""

    #: How many requests the daemon may have in flight against Whisper and
    #: Ollama together.  Whisper and Ollama usually share one machine, and
    #: often one GPU, which queues what it is given — one at a time, usually.
    #: Asking for more than the device serves is what produces 5xx answers
    #: while one model evicts another, so this defaults to the queue depth of
    #: a single GPU.  Raise it when the services run on separate machines.
    max_concurrent_requests: int
    #: How many times one request may be tried before it is given up on: the
    #: first attempt plus this many minus one retries, each waiting twice as
    #: long as the one before.  Five spans about fifteen seconds, which is the
    #: scale of a model being evicted from a shared GPU and loaded again.
    max_attempts: int


@dataclass(frozen=True)
class TranscriptConfig:
    """How the transcript is assembled from the recordings."""

    #: Merge the participants' speaking turns into one time-ordered document.
    #: Off means one block per participant, in filename order, which is what
    #: sessions without a timeline produce either way.
    interleave: bool
    #: Silence between two turns of one speaker that still counts as one turn
    #: when they are transcribed: a pause inside a sentence is shorter than
    #: this, a reply is usually longer.
    merge_gap_seconds: float


@dataclass(frozen=True)
class EndpointConfig:
    """A JSON-over-HTTP endpoint, such as the Whisper transcribe service."""

    url: str
    timeout: float
    verify_tls: bool


@dataclass(frozen=True)
class OllamaConfig(EndpointConfig):
    """The Ollama generate endpoint, which additionally needs a model name."""

    model: str
    #: Ask the model to repair grammar and wording before summarising.  It
    #: costs one more pass per meeting, and it rewrites what people said —
    #: which is why it is off by default and why the raw transcript is always
    #: kept beside the corrected one.
    correct_transcript: bool = False


@dataclass(frozen=True)
class SmtpConfig:
    """Outgoing mail settings."""

    host: str
    port: int
    user: str
    password: str
    sender: str
    fallback_recipient: str
    use_starttls: bool
    #: Optional suffix for the subject line, such as an organisation name.
    subject_suffix: str


@dataclass(frozen=True)
class MailConfig:
    """How the summary mail is presented, as opposed to how it is delivered."""

    #: Send an HTML part alongside the plain-text body.
    html: bool
    #: A stylesheet to use in place of the packaged one, or ``None`` for the
    #: stock theme.  This is the whole of re-branding: see html_mail.
    stylesheet: Path | None
    #: The closing line of the mail.
    footer: str
    #: Speaking turns shown before the mail defers to the attached transcript.
    preview_turns: int


@dataclass(frozen=True)
class S3Config:
    """Where a finished meeting's video is uploaded, and how.

    An S3-compatible endpoint holds the video, which is the one product of a
    meeting this daemon cannot make itself: Jibri records it, and only when
    somebody asked it to.  Everything here is empty by default, so a
    deployment that does not archive its recordings uploads nothing.
    """

    endpoint: str
    bucket: str
    #: Key prefix, e.g. ``meetings``.  Empty puts the videos at the root.
    prefix: str
    region: str
    access_key: str
    secret_key: str
    #: Address the bucket as ``endpoint/bucket`` rather than
    #: ``bucket.endpoint``.  Nearly every self-hosted S3-compatible server
    #: wants the former, and AWS accepts both.
    path_style: bool
    verify_tls: bool
    #: Where Jibri writes its recordings; the video for a meeting is found
    #: here.  It must not be the bridge's own recordings directory.
    jibri_dir: Path
    #: How long to keep looking for a recording that has not appeared yet.
    #: The recording ends when somebody stops it, which can be after the
    #: meeting and after its transcript has been written.
    wait_seconds: float = 900.0
    #: How long a recording must have been untouched before it is uploaded,
    #: so that an ffmpeg still writing its last frames is left alone.
    settle_seconds: float = 30.0
    #: Remove the local recording once the endpoint has confirmed it.
    delete_after_upload: bool = False
    #: Put the recording's link in the summary mail.  Off by default because
    #: it changes *when* the mail is sent: the daemon then looks for the
    #: recording and uploads it before mailing, so the link works as soon as
    #: somebody reads it.
    link_in_mail: bool = False
    #: How long the mail waits for a recording that has not appeared yet
    #: before giving up and going out without a link.  A meeting nobody
    #: recorded must not hold the transcript back.
    link_wait_seconds: float = 120.0
    #: How long a signed link works, in seconds; zero means the URL is not
    #: signed at all, which is only right for a bucket anyone may read.
    #: SigV4 refuses more than seven days, so a larger value is clamped.
    link_expiry_seconds: float = 604800.0
    #: Where recipients reach the bucket.  A daemon behind a tunnel usually
    #: uploads to an address nobody outside can use, and the link in a mail is
    #: read outside; empty keeps the two the same.
    link_endpoint: str = ""

    @property
    def enabled(self) -> bool:
        """Whether an endpoint and a bucket were configured."""
        return bool(self.endpoint and self.bucket)

    @property
    def link_base(self) -> str:
        """The address a link for a person should be built on."""
        return self.link_endpoint or self.endpoint


@dataclass(frozen=True)
class Config:
    """Fully resolved configuration."""

    server: ServerConfig
    storage: StorageConfig
    transcript: TranscriptConfig
    ai: AiConfig
    whisper: EndpointConfig
    ollama: OllamaConfig
    smtp: SmtpConfig
    s3: S3Config
    mail: MailConfig
    #: The file the values came from, or ``None`` if only defaults applied.
    source: Path | None = None


def _env_override(section: str, key: str) -> str | None:
    """Return the environment override for *section*/*key*, if one is set."""
    return os.environ.get(f"{ENV_PREFIX}{section.upper()}_{key.upper()}")


class _Resolver:
    """Looks values up across the environment, a config file, and defaults."""

    def __init__(self, parser: configparser.ConfigParser, source: Path | None) -> None:
        self._parser = parser
        self._source = source

    def raw(self, section: str, key: str) -> str:
        override = _env_override(section, key)
        if override is not None:
            logger.debug("%s.%s overridden by environment", section, key)
            return override
        if self._parser.has_option(section, key):
            return self._parser.get(section, key)
        return DEFAULTS[section][key]

    def where(self, section: str, key: str) -> str:
        """Describe where a value came from, for error messages."""
        if _env_override(section, key) is not None:
            return f"{ENV_PREFIX}{section.upper()}_{key.upper()}"
        if self._source is not None and self._parser.has_option(section, key):
            return f"{self._source} [{section}] {key}"
        return "built-in default"

    def _fail(self, section: str, key: str, value: str, expected: str) -> ConfigError:
        return ConfigError(
            f"invalid value for [{section}] {key}: {value!r} is not {expected} "
            f"(from {self.where(section, key)})"
        )

    def integer(self, section: str, key: str, *, minimum: int, maximum: int) -> int:
        raw = self.raw(section, key).strip()
        try:
            value = int(raw)
        except ValueError:
            raise self._fail(section, key, raw, "an integer") from None
        if not minimum <= value <= maximum:
            raise ConfigError(
                f"invalid value for [{section}] {key}: {value} is outside the "
                f"permitted range {minimum}-{maximum} (from {self.where(section, key)})"
            )
        return value

    def number(self, section: str, key: str, *, minimum: float) -> float:
        raw = self.raw(section, key).strip()
        try:
            value = float(raw)
        except ValueError:
            raise self._fail(section, key, raw, "a number") from None
        if value < minimum:
            raise ConfigError(
                f"invalid value for [{section}] {key}: {value} must be at least "
                f"{minimum} (from {self.where(section, key)})"
            )
        return value

    def boolean(self, section: str, key: str) -> bool:
        raw = self.raw(section, key).strip()
        try:
            return self._parser.BOOLEAN_STATES[raw.lower()]
        except KeyError:
            raise self._fail(section, key, raw, "a boolean (yes/no, true/false, on/off)") from None

    def text(self, section: str, key: str) -> str:
        return self.raw(section, key).strip()

    def required(self, section: str, key: str) -> str:
        value = self.text(section, key)
        if not value:
            raise ConfigError(
                f"[{section}] {key} must not be empty (from {self.where(section, key)})"
            )
        return value

    def path(self, section: str, key: str) -> Path:
        value = self.required(section, key)
        return Path(value).expanduser()

    def optional_path(self, section: str, key: str) -> Path | None:
        """A path setting that is absent or empty when the feature is off."""
        value = self.text(section, key)
        return Path(value).expanduser() if value else None


def _find_config_file(explicit: str | os.PathLike[str] | None) -> Path | None:
    """Resolve which config file to read, if any."""
    requested = explicit or os.environ.get(ENV_CONFIG_PATH)
    if requested:
        candidate = Path(requested).expanduser()
        if not candidate.is_file():
            raise ConfigError(f"config file not found: {candidate}")
        return candidate

    for candidate in SEARCH_PATHS:
        if candidate.is_file():
            return candidate
    return None


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load configuration from *path*, the environment, and built-in defaults.

    Raises:
        ConfigError: if a file is named but missing, is unreadable, or holds a
            value that cannot be coerced to the expected type.
    """
    source = _find_config_file(path)

    # interpolation is disabled: SMTP passwords routinely contain '%', which
    # the default BasicInterpolation would try to expand.
    parser = configparser.ConfigParser(interpolation=None)
    if source is not None:
        try:
            with source.open("r", encoding="utf-8") as handle:
                parser.read_file(handle)
        except (OSError, configparser.Error) as exc:
            raise ConfigError(f"cannot read {source}: {exc}") from exc
        logger.info("loaded configuration from %s", source)
    else:
        logger.info("no config file found; using built-in defaults")

    resolver = _Resolver(parser, source)

    return Config(
        server=ServerConfig(
            host=resolver.required("server", "host"),
            port=resolver.integer("server", "port", minimum=1, maximum=65535),
        ),
        storage=StorageConfig(
            recordings_dir=resolver.path("storage", "recordings_dir"),
            cleanup_after_send=resolver.boolean("storage", "cleanup_after_send"),
            session_metadata_dir=resolver.optional_path("storage", "session_metadata_dir"),
            capture_timeline=resolver.boolean("storage", "capture_timeline"),
            session_grace_seconds=resolver.number(
                "storage", "session_grace_seconds", minimum=0.0
            ),
        ),
        ai=AiConfig(
            max_concurrent_requests=resolver.integer(
                "ai", "max_concurrent_requests", minimum=1, maximum=64
            ),
            max_attempts=resolver.integer("ai", "max_attempts", minimum=1, maximum=10),
        ),
        transcript=TranscriptConfig(
            interleave=resolver.boolean("transcript", "interleave"),
            merge_gap_seconds=resolver.number("transcript", "merge_gap_seconds", minimum=0.0),
        ),
        whisper=EndpointConfig(
            url=resolver.required("whisper", "url"),
            timeout=resolver.number("whisper", "timeout", minimum=0.1),
            verify_tls=resolver.boolean("whisper", "verify_tls"),
        ),
        ollama=OllamaConfig(
            correct_transcript=resolver.boolean("ollama", "correct_transcript"),
            url=resolver.required("ollama", "url"),
            timeout=resolver.number("ollama", "timeout", minimum=0.1),
            verify_tls=resolver.boolean("ollama", "verify_tls"),
            model=resolver.required("ollama", "model"),
        ),
        smtp=SmtpConfig(
            host=resolver.required("smtp", "host"),
            port=resolver.integer("smtp", "port", minimum=1, maximum=65535),
            # user/password are optional: an internal relay may accept mail
            # without authentication.
            user=resolver.text("smtp", "user"),
            password=resolver.text("smtp", "password"),
            sender=resolver.required("smtp", "sender"),
            fallback_recipient=resolver.text("smtp", "fallback_recipient"),
            use_starttls=resolver.boolean("smtp", "use_starttls"),
            subject_suffix=resolver.text("smtp", "subject_suffix"),
        ),
        s3=S3Config(
            # endpoint and bucket are optional: empty ones are what turns the
            # upload off, so a missing endpoint is not a configuration error.
            endpoint=resolver.text("s3", "endpoint"),
            bucket=resolver.text("s3", "bucket"),
            prefix=resolver.text("s3", "prefix"),
            region=resolver.text("s3", "region"),
            access_key=resolver.text("s3", "access_key"),
            secret_key=resolver.text("s3", "secret_key"),
            path_style=resolver.boolean("s3", "path_style"),
            verify_tls=resolver.boolean("s3", "verify_tls"),
            jibri_dir=resolver.path("s3", "jibri_dir"),
            wait_seconds=resolver.number("s3", "wait_seconds", minimum=0.0),
            settle_seconds=resolver.number("s3", "settle_seconds", minimum=0.0),
            delete_after_upload=resolver.boolean("s3", "delete_after_upload"),
            link_in_mail=resolver.boolean("s3", "link_in_mail"),
            link_wait_seconds=resolver.number("s3", "link_wait_seconds", minimum=0.0),
            link_expiry_seconds=resolver.number("s3", "link_expiry_seconds", minimum=0.0),
            link_endpoint=resolver.text("s3", "link_endpoint"),
        ),
        mail=MailConfig(
            html=resolver.boolean("mail", "html"),
            # Empty means "use the packaged stylesheet", so an absent path is
            # not an error the way a missing recordings directory would be.
            stylesheet=resolver.optional_path("mail", "stylesheet"),
            footer=resolver.text("mail", "footer"),
            preview_turns=resolver.integer("mail", "preview_turns", minimum=0, maximum=100),
        ),
        source=source,
    )
