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
        "serialize_requests": "true",
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

    #: Whisper and Ollama usually run on one machine, often on one GPU, where
    #: a model loaded by one starves the other — the starved service answers
    #: 5xx until its own model is back.  Sending one request at a time keeps
    #: the daemon from doing that to itself when two meetings overlap.  Off is
    #: right when the two services are on separate machines.
    serialize_requests: bool


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
class Config:
    """Fully resolved configuration."""

    server: ServerConfig
    storage: StorageConfig
    transcript: TranscriptConfig
    ai: AiConfig
    whisper: EndpointConfig
    ollama: OllamaConfig
    smtp: SmtpConfig
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
            serialize_requests=resolver.boolean("ai", "serialize_requests"),
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
        source=source,
    )
