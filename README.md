# jitsi-audio-bridge

Captures per-participant audio from a Jitsi conference, then transcribes it with
a local Whisper instance, summarises the transcript with a local Ollama model,
and emails the result.

```
        JVB / sender
             │  ws://host:port/transcribe?sessionId=<id>
             │  ── text frame:  JSON control frame (room, participants)
             │  ── binary frame: [16-byte participant id][Opus packet]
             ▼
    ┌──────────────────┐
    │   daemon.py      │  WebSocket server, one connection per meeting
    │  (asyncio loop)  │
    └────────┬─────────┘
             │  per participant, in-process
             ▼
    ┌──────────────────┐
    │    audio.py      │  libopus (via ctypes) ──▶ 16 kHz mono WAV
    └────────┬─────────┘
             │  on disconnect, handed to a worker thread
             ▼
    ┌──────────────────┐   ┌───────────────┐   ┌──────────────┐
    │   ai_client.py   │──▶│    Whisper    │   │    Ollama    │
    │                  │◀──│  (HTTP/JSON)  │   │ (HTTP/JSON)  │
    └────────┬─────────┘   └───────────────┘   └──────────────┘
             │                                        ▲
             │  transcript.txt                        │ summary
             ▼                                        │
    ┌──────────────────┐                              │
    │    mailer.py     │◀─────────────────────────────┘
    └──────────────────┘   summary + transcript by SMTP
```

## Contents

- [Requirements](#requirements)
- [Network access](#network-access)
- [Install](#install)
- [Debian package](#debian-package)
- [Configuration](#configuration)
- [Running](#running)
- [Wire protocol](#wire-protocol)
- [Jitsi integration](docs/jitsi-integration.md)
- [Batch mode](#batch-mode)
- [Running as a service](#running-as-a-service)
- [Test environment](#test-environment)
- [Output layout](#output-layout)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Development](#development)
- [Licence](#licence)

## Requirements

| | |
|---|---|
| Python | 3.11 or newer (developed against 3.14) |
| libopus | `libopus.so.0` — `apt install libopus0`. The `-dev` package is **not** needed. |
| Whisper | Any HTTP endpoint accepting `{"audio_base64": ..., "filename": ...}` and returning `{"text": ...}` |
| Ollama | Any HTTP endpoint accepting the `/api/generate` request shape |
| SMTP | Any relay |
| S3 | Optional: any S3-compatible endpoint, and Jibri's recordings directory readable — see [`[s3]`](#s3) |

ffmpeg is **not** required. Audio is decoded in-process through libopus, which
is both faster and less fragile than piping packets to a subprocess.

## Network access

Everything the daemon connects to is **outbound**, from the host it runs on, to
an endpoint named in `config.ini`. Nothing needs to be opened towards it: the
WebSocket server listens on `[server] host`/`port` (loopback by default) and the
JVB connects to that.

| Destination | Port | When | Configured by |
|---|---|---|---|
| Whisper endpoint | `443` (as written in its URL) | Every meeting, once per turn or recording | `[whisper] url` |
| Ollama endpoint | `443` | Every meeting: language probe, summary, and the correction pass if enabled | `[ollama] url` |
| SMTP relay | `25` by default, `587` for a relay that wants STARTTLS submission | When a meeting's summary is ready | `[smtp] host`, `[smtp] port` |
| S3-compatible endpoint | `443` for TLS, `9000` for a MinIO reached directly | Once the transcript is written, when `[s3] endpoint` and `bucket` are set — before the mail with `link_in_mail` | `[s3] endpoint` |
| Jibri's recordings | *(no network)* | Reading the meeting's video | `[s3] jibri_dir`, on this host |
| `[s3] link_endpoint` | as written | *(inbound, for the recipients)* — the mail's link has to reach them | `[s3] link_endpoint` |

Port `465` is not on that list and will not work: it expects TLS from the first
byte, while this daemon opens a plain connection and upgrades it with STARTTLS
(`[smtp] use_starttls`), which is what `25` and `587` are for.

Every row but the last is something the daemon itself connects to, and every
URL carries its own port: what has to be reachable is whatever `[whisper] url`,
`[ollama] url` and `[s3] endpoint` actually name — there is nothing else to
read, and nothing is contacted that `config.ini` does not mention.
`link_endpoint` is the other direction: it is where *recipients* fetch the
recording from, so it has to be reachable by them, and a mail that links
somewhere only the daemon can reach is a link that fails for everybody who
reads it. The shipped defaults point at two external hostnames,
`whisper.omnia.amarulasolutions.com` and `ollama.omnia.amarulasolutions.com`,
so outbound `443` has to be open to those, and the same for whatever the S3
endpoint turns out to be. Where the services sit behind a reverse proxy or a
tunnel — Pangolin fronts both of the above — that is one hop more to whitelist:
the daemon reaches them no other way, and a blocked connection is a per-request
retry and then a failed meeting, not a startup error.

Three things that catch people out:

- **`path_style = false` moves the request to another hostname.** The bucket
  becomes a subdomain of the endpoint — `https://bucket.minio.example.com/...`
  — so a whitelist entry for the endpoint's own name is not enough, and the
  certificate has to cover that name too. With `link_in_mail` that applies to
  the *recipients* as well: the link points at
  `<bucket>.<link_endpoint>`, which means a wildcard DNS entry and a wildcard
  certificate on the name people can reach. The default (`true`) keeps
  everything on the one host.
- **An empty `[s3] access_key`/`secret_key` makes botocore search for
  credentials**, and on an EC2 instance that search ends at the instance
  metadata service on `169.254.169.254:80`. Set the two keys explicitly (or in
  `/etc/jitsi-audio-bridge/env`) to keep that lookup from happening at all.
- **DNS has to work too**, including for whatever the tunnel carries; the
  systemd unit already allows the address families name resolution needs
  (`AF_INET`, `AF_INET6`, `AF_UNIX`, `AF_NETLINK`).

### What the S3 upload asks for, request by request

A proxy or firewall rule has to match something, so this is the whole of what
leaves the daemon. With

```ini
[s3]
endpoint = https://minio.omnia.amarulasolutions.com
bucket = meetings
prefix = videos
path_style = true
```

every request goes to **one host** — `minio.omnia.amarulasolutions.com`, on the
port written in that URL — and to a path of this shape, where only the room and
the filename vary:

```
/meetings/videos/<room>/<room>_<yyyy-MM-dd-HH-mm-ss>.<ext>
└─ bucket ─┘└ prefix ┘
```

| Method | URL | When |
|---|---|---|
| `POST` | `…?uploads` | opens a multipart upload |
| `PUT` | `…?uploadId=…&partNumber=N` | one 8 MiB piece; several go at once |
| `POST` | `…?uploadId=…` | completes the upload |
| `PUT` | `…` — no query | the whole object, when it is under 8 MiB |
| `HEAD` | `…` — no query | reads the size back before the claim is written |

So **one resource for the hostname, with the path matched as `^/meetings/`**
covers every request for every meeting; `/meetings/videos/**` is the narrower
version of the same rule. The bucket is the first path segment and the prefix
the second — a rule written for `/videos/` matches nothing at all.

Three details worth knowing while writing that rule:

- **Almost every request is a multipart one.** Boto3 switches at 8 MiB, and a
  recording is far past that, so the single `PUT` is the rare case in a proxy
  log, not the normal one.
- **The query string is not part of the path**, and a rule anchored on the
  filename (`…\.mp4$`) will not match the multipart requests, which carry on
  past it with `?uploads` or `?partNumber=`.
- **A path in `endpoint` is kept**: `https://gateway.example.com/minio` puts
  everything under `/minio/meetings/…`, which is then the path to match.

With `path_style = false` the same requests move to another hostname —
`https://meetings.minio.omnia.amarulasolutions.com/videos/…` — which in a proxy
means a wildcard resource and, for TLS, a certificate covering
`*.minio.omnia.amarulasolutions.com`. The default (`true`) keeps everything on
the one name.

To see what a given installation actually talks to, read it off its own
configuration rather than this table:

```sh
grep -E '^\[|url|endpoint|host|port' /etc/jitsi-audio-bridge/config.ini
```

A wrong Whisper or Ollama endpoint is visible at the first meeting; a wrong S3
endpoint only ever produces a log line, because a failed upload is deliberately
not allowed to cost a meeting its transcript — see
[Archiving the video](#archiving-the-video).

## Install

The daemon reads its configuration from `config.ini`. Copy the example and edit
it — `config.ini` itself is gitignored, because it is the one file that may end
up holding an SMTP password.

```sh
git clone <this repo> /opt/jitsi-audio-bridge
cd /opt/jitsi-audio-bridge

python3 -m venv .venv
.venv/bin/pip install '.[s3]'          # drop [s3] to leave boto3 out
cp config.ini.example config.ini
$EDITOR config.ini
```

The `s3` extra is what brings in boto3, which only
[`[s3]`](#s3) uses; without it the daemon runs normally and says so if an
upload is ever attempted.

Install into `/opt`, **not** `/home`: the shipped systemd unit sets
`ProtectHome=yes`, which makes `/home` inaccessible to the service.

For development, install editable with the test and lint extras:

```sh
.venv/bin/pip install -e '.[dev]'
```

## Debian package

`make deb` builds `dist/jitsi-audio-bridge_<version>_<arch>.deb`:

```sh
sudo apt install dpkg-dev python3-venv   # build prerequisites
make deb
sudo dpkg -i dist/jitsi-audio-bridge_*_amd64.deb
```

**Build on the machine you will install it on** (or in a container matching
it). The package carries its dependencies in a private virtualenv at
`/usr/lib/jitsi-audio-bridge/venv`, because the distribution's
`python3-websockets` is older than the `websockets>=13` this daemon requires
(Debian 12 ships 10.4, Ubuntu 24.04 ships 12.0). A virtualenv belongs to one
interpreter version and architecture, so the postinst warns loudly if the
target's `python3` minor differs from the one it was built for.

What the package installs: `/usr/bin/jitsi-audio-bridge` (a launcher for the
venv), the systemd unit, and `/etc/jitsi-audio-bridge/config.ini` **as a
conffile**, so upgrades never clobber your edits. It also installs the two
deployment tools, so a Jitsi host needs nothing but this package to run them:
`jitsi-audio-bridge-verify` ([the deployment checker](#jitsi-integration),
meant to be run on the Jitsi host) and `jitsi-audio-bridge-send` (the sender
simulator from [Test environment](#test-environment), for exercising the bridge
without Jitsi). The postinst creates the
`jitsi-bridge` system user and `/srv/recordings`, and enables the unit but
deliberately does not start it — review `config.ini`, put the SMTP password in
`/etc/jitsi-audio-bridge/env` (mode 0600, created empty), then
`systemctl start jitsi-audio-bridge`. `Depends: python3 (>= 3.11), libopus0`;
`Recommends: ffmpeg` (only batch mode's master-track extraction uses it).
Removing the package stops and disables the unit; purging leaves
`/srv/recordings` and the service user in place, because the recordings are the
only copy of a meeting.

## Configuration

Values are resolved from three sources, each overriding the one below it:

1. built-in defaults (so the daemon starts with no config file at all);
2. `config.ini`;
3. the process environment, as `JITSI_AUDIO_BRIDGE_<SECTION>_<KEY>`.

The environment layer exists so secrets can be supplied by a systemd
`EnvironmentFile` and never written to disk. The file is found in this order:
`--config PATH`, `$JITSI_AUDIO_BRIDGE_CONFIG`, `./config.ini`,
`/etc/jitsi-audio-bridge/config.ini`.

### `[server]`

| Option | Type | Default | Environment |
|---|---|---|---|
| `host` | address | `127.0.0.1` | `JITSI_AUDIO_BRIDGE_SERVER_HOST` |
| `port` | 1–65535 | `8080` | `JITSI_AUDIO_BRIDGE_SERVER_PORT` |

### `[storage]`

| Option | Type | Default | Environment |
|---|---|---|---|
| `recordings_dir` | path | `/srv/recordings` | `JITSI_AUDIO_BRIDGE_STORAGE_RECORDINGS_DIR` |
| `cleanup_after_send` | boolean | `false` | `JITSI_AUDIO_BRIDGE_STORAGE_CLEANUP_AFTER_SEND` |
| `session_metadata_dir` | path | *(empty)* | `JITSI_AUDIO_BRIDGE_STORAGE_SESSION_METADATA_DIR` |
| `capture_timeline` | boolean | `true` | `JITSI_AUDIO_BRIDGE_STORAGE_CAPTURE_TIMELINE` |
| `session_grace_seconds` | seconds | `60` | `JITSI_AUDIO_BRIDGE_STORAGE_SESSION_GRACE_SECONDS` |

The subject and the heading inside the mail carry the meeting's local date and
time — `Meeting Summary & Transcript: Weekly-Planning (2026-10-06 18:50)` —
because a room keeps its name and every meeting held in it would otherwise be
labelled identically. The time comes from the timeline; a session without one
(see `capture_timeline`) is labelled with the session directory's own
timestamp, or with nothing if even that cannot be read. `subject_suffix` is
appended after it, so the two are easy to tell apart in a mailbox.

The mail is written in the language the meeting was held in: the summary, its
headings (the model's) and the subject line, the heading and the introduction
(the daemon's) all follow the detected language, so an Italian meeting is not
mailed under an English title. English, Italian, Spanish, French and German
have their own words; anything else is mailed in English, which is also the
fallback when detection fails. The rule is also stated *after* the transcript,
not only before it: a meeting about software is full of English words whatever
language it is held in, and a model answers in the language of what it read
last — a transcript of Italian speech salted with "file name" and "widget
preview" produced an English summary under an Italian heading until the
instruction was repeated last. The subject and heading also carry the
meeting's local date and time, because a room keeps its name and two meetings
in it would otherwise be labelled identically.

`cleanup_after_send` deletes the audio, transcript and summary once the email
has been sent. It is off by default deliberately: those files are the only copy
of the meeting, so a mistake here destroys one.

`session_metadata_dir` is where a companion service — the Prosody module in
[docs/jitsi-integration.md](docs/jitsi-integration.md) §5 — drops per-meeting
metadata, keyed by meeting id. A session that has no `metadata.json` of its own
adopts the file written for it, which is how a stock-Jitsi session gets a room
name, speaker names and, where the deployment authenticates users, recipients.
Empty disables it.

`capture_timeline` writes `timeline.json` for sessions fed by the JVB's
media export: who spoke when, on the session's own clock. It can only be
captured while the meeting is running — a meeting recorded with this off can
never be interleaved afterwards — and it costs one small file per session.

`session_grace_seconds` is how long a session may stay silent before the
meeting is treated as over. A connection ending is not the meeting ending: the
JVB closes one export and opens another for the same conference (a transcriber
restarted, a bridge reconnected), so a meeting must be transcribed and mailed
once, not once per connection. Every connection arriving within this window
resumes the same session — the clock and the turns carry on, and each
connection's audio goes to its own part file, `participant-<id>-2.wav`,
because `wave.open` truncates and reusing the file would destroy what came
before. Zero restores the old behaviour of processing each connection's audio
on its own.

### `[transcript]`

| Option | Type | Default | Environment |
|---|---|---|---|
| `interleave` | boolean | `true` | `JITSI_AUDIO_BRIDGE_TRANSCRIPT_INTERLEAVE` |
| `merge_gap_seconds` | seconds | `1.0` | `JITSI_AUDIO_BRIDGE_TRANSCRIPT_MERGE_GAP_SECONDS` |

`interleave` merges the participants' speaking turns into one time-ordered
document — `[00:03:12] Alice: …` — instead of one block per participant in
filename order. It needs a timeline: a session without one (the legacy
framing, or a meeting processed from a directory someone else recorded) keeps
the per-participant shape whatever this is set to. `merge_gap_seconds` is how
much silence between two runs of one speaker still counts as the same turn.

### `[ai]`

| Option | Type | Default | Environment |
|---|---|---|---|
| `max_concurrent_requests` | 1–64 | `1` | `JITSI_AUDIO_BRIDGE_AI_MAX_CONCURRENT_REQUESTS` |
| `max_attempts` | 1–10 | `5` | `JITSI_AUDIO_BRIDGE_AI_MAX_ATTEMPTS` |

Whisper and Ollama usually run on one machine, often on one GPU, and a GPU's
queue takes one request at a time. Asking for more than the device serves is
what makes one model evict another, and the starved service answers 5xx until
its own model is back — that is what a `503 Service Unavailable` from Whisper
means while a large Ollama model is resident. `1` matches a single GPU: the
daemon waits for each request to complete before sending the next, so two
meetings finishing together cannot do that to each other — one meeting's
summary waits for the other's transcription. Raise it when the two services
are on separate machines and the device underneath can take more.

`max_attempts` is how many times one request may be tried before it is given
up on, waiting twice as long before each retry — 1 s, 2 s, 4 s, 8 s — because
what is being waited for is a model being loaded, which takes seconds and then
takes them all at once. Five attempts span about fifteen seconds; the wait is
capped at 30 s whatever the limit. Only a 5xx or a failed connection is
retried: a 4xx means the request itself is wrong. Either way the service's
own explanation is quoted in the log — "model not loaded", "all slots are
busy", a CUDA error — because a failure body is the only place that reason
appears, and the daemon's log is where you will be looking.

### `[whisper]`

| Option | Type | Default | Environment |
|---|---|---|---|
| `url` | URL | `https://whisper.omnia.amarulasolutions.com/transcribe-b64` | `JITSI_AUDIO_BRIDGE_WHISPER_URL` |
| `timeout` | seconds | `600` | `JITSI_AUDIO_BRIDGE_WHISPER_TIMEOUT` |
| `verify_tls` | boolean | `true` | `JITSI_AUDIO_BRIDGE_WHISPER_VERIFY_TLS` |

### `[ollama]`

| Option | Type | Default | Environment |
|---|---|---|---|
| `url` | URL | `https://ollama.omnia.amarulasolutions.com/api/generate` | `JITSI_AUDIO_BRIDGE_OLLAMA_URL` |
| `model` | string | `qwen2.5:14b-instruct` | `JITSI_AUDIO_BRIDGE_OLLAMA_MODEL` |
| `timeout` | seconds | `600` | `JITSI_AUDIO_BRIDGE_OLLAMA_TIMEOUT` |
| `verify_tls` | boolean | `true` | `JITSI_AUDIO_BRIDGE_OLLAMA_VERIFY_TLS` |
| `correct_transcript` | boolean | `false` | `JITSI_AUDIO_BRIDGE_OLLAMA_CORRECT_TRANSCRIPT` |

`correct_transcript` asks the model to repair grammar, punctuation and
clearly misheard words before the summary is written. Speech recognition
leaves all three behind, and the model that reads the whole meeting can often
tell "teh" from "the" or a mangled name from a correct one in context. It
costs one more pass, in chunks of a few thousand characters so a long meeting
still fits the model's context, and it **rewrites what people said** — which
is why it is off by default and why the raw transcript is kept: the corrected
text is written beside it as `transcript.corrected.txt`, is what gets
summarised and mailed, and is discarded (with a warning) if the pass fails, so
a meeting is never mailed half-repaired.

### `[smtp]`

| Option | Type | Default | Environment |
|---|---|---|---|
| `host` | hostname | `127.0.0.1` | `JITSI_AUDIO_BRIDGE_SMTP_HOST` |
| `port` | 1–65535 | `25` | `JITSI_AUDIO_BRIDGE_SMTP_PORT` |
| `user` | string | *(empty)* | `JITSI_AUDIO_BRIDGE_SMTP_USER` |
| `password` | string | *(empty)* | `JITSI_AUDIO_BRIDGE_SMTP_PASSWORD` |
| `sender` | address | `no-reply@amarulasolutions.com` | `JITSI_AUDIO_BRIDGE_SMTP_SENDER` |
| `fallback_recipient` | address | `admin@omnia.amarulasolutions.com` | `JITSI_AUDIO_BRIDGE_SMTP_FALLBACK_RECIPIENT` |
| `use_starttls` | boolean | `true` | `JITSI_AUDIO_BRIDGE_SMTP_USE_STARTTLS` |
| `subject_suffix` | string | *(empty)* | `JITSI_AUDIO_BRIDGE_SMTP_SUBJECT_SUFFIX` |

### `[s3]`

Uploads the meeting's recording — the video Jibri made, if somebody pressed
Record — to an S3-compatible endpoint once the meeting has been transcribed.
Off unless both `endpoint` and `bucket` are set. The section does nothing on
its own: Jibri has to be running and recording for there to be a video, and its
recordings directory has to be readable from this host.

By default the upload follows the mail, which never mentions it. With
`link_in_mail` the order is reversed on purpose — the mail waits for the
upload and links to the recording — see
[The link in the mail](#the-link-in-the-mail).

| Option | Type | Default | Environment |
|---|---|---|---|
| `endpoint` | URL | *(empty)* | `JITSI_AUDIO_BRIDGE_S3_ENDPOINT` |
| `bucket` | string | *(empty)* | `JITSI_AUDIO_BRIDGE_S3_BUCKET` |
| `prefix` | string | *(empty)* | `JITSI_AUDIO_BRIDGE_S3_PREFIX` |
| `region` | string | `us-east-1` | `JITSI_AUDIO_BRIDGE_S3_REGION` |
| `access_key` | string | *(empty)* | `JITSI_AUDIO_BRIDGE_S3_ACCESS_KEY` |
| `secret_key` | string | *(empty)* | `JITSI_AUDIO_BRIDGE_S3_SECRET_KEY` |
| `path_style` | boolean | `true` | `JITSI_AUDIO_BRIDGE_S3_PATH_STYLE` |
| `verify_tls` | boolean | `true` | `JITSI_AUDIO_BRIDGE_S3_VERIFY_TLS` |
| `jibri_dir` | path | `/srv/jibri-recordings` | `JITSI_AUDIO_BRIDGE_S3_JIBRI_DIR` |
| `wait_seconds` | seconds | `900` | `JITSI_AUDIO_BRIDGE_S3_WAIT_SECONDS` |
| `settle_seconds` | seconds | `30` | `JITSI_AUDIO_BRIDGE_S3_SETTLE_SECONDS` |
| `delete_after_upload` | boolean | `false` | `JITSI_AUDIO_BRIDGE_S3_DELETE_AFTER_UPLOAD` |
| `link_in_mail` | boolean | `false` | `JITSI_AUDIO_BRIDGE_S3_LINK_IN_MAIL` |
| `link_wait_seconds` | seconds | `120` | `JITSI_AUDIO_BRIDGE_S3_LINK_WAIT_SECONDS` |
| `link_expiry_seconds` | seconds | `604800` | `JITSI_AUDIO_BRIDGE_S3_LINK_EXPIRY_SECONDS` |
| `link_endpoint` | URL | *(empty)* | `JITSI_AUDIO_BRIDGE_S3_LINK_ENDPOINT` |

Objects are stored as `<prefix>/<room>/<Jibri's own filename>`, so a bucket
reads as one directory per room. Empty `access_key` and `secret_key` fall back
to botocore's own credential chain — the environment, a shared credentials
file, an instance profile — and `AWS_CA_BUNDLE` names a CA for an endpoint with
a certificate of its own.

`endpoint` is the one host that has to be reachable from this daemon for the
upload to work, outbound, on whatever port it names — see
[Network access](#network-access) for what to whitelist and what changes when
`path_style` is off.

`jibri_dir` must be Jibri's `recordings_directory` and must not be this
daemon's `recordings_dir`: Jibri creates a session directory per recording, and
a bridge-owned directory it cannot write to is what makes Jibri report itself
unhealthy. See [Jitsi integration](docs/jitsi-integration.md) for how the two
trees end up apart.

The daemon finds the recording by room name and by time: Jibri names each file
after the call and stamps the moment the recording stopped onto it, and it
leaves a `metadata.json` holding the call URL beside it. A recording that
stopped too long before the meeting began, or that another meeting has already
uploaded, is not a candidate. See [Output layout](#output-layout) for what the
meeting directory records about the upload.

`link_expiry_seconds` is how long the link in the mail works, and 604800 —
seven days — is the most SigV4 will sign for; a longer value is clamped to it,
with a warning, because the server refuses such a link only when somebody
clicks it. `0` signs nothing and mails the plain `endpoint/bucket/key` URL,
which is right only for a bucket anyone may read. `link_endpoint` is the
address links are built on when recipients reach the bucket by another name
than the daemon uploads to.

Booleans accept `true/false`, `yes/no`, `on/off` and `1/0`. Values are validated
at startup and a bad one names the exact setting and where it came from:

```
ERROR  invalid value for [server] port: 'bogus' is not an integer (from /etc/jitsi-audio-bridge/config.ini [server] port)
```

### Secrets

Keep `config.ini` world-readable and pass the SMTP password through the
environment instead:

```sh
install -m 0600 -o root -g root /dev/null /etc/jitsi-audio-bridge/env
printf 'JITSI_AUDIO_BRIDGE_SMTP_PASSWORD=...\n' >> /etc/jitsi-audio-bridge/env
```

`verify_tls = false` disables certificate validation on the Whisper and Ollama
connections. Only use it for an endpoint with a self-signed certificate on a
trusted network; it makes the transcript path interceptable.

To keep verification on against a self-signed endpoint, point
`REQUESTS_CA_BUNDLE` at a PEM holding its certificate — via
`/etc/jitsi-audio-bridge/env`, which the unit reads:

```sh
sudo openssl s_client -connect whisper.example.com:443 -servername whisper.example.com \
    </dev/null 2>/dev/null | sudo tee /etc/jitsi-audio-bridge/ca.pem >/dev/null
printf 'REQUESTS_CA_BUNDLE=/etc/jitsi-audio-bridge/ca.pem\n' \
    | sudo tee -a /etc/jitsi-audio-bridge/env
```

One file, so concatenate the certificates into it if Whisper and Ollama have
different ones. The system trust store is **not** consulted: the daemon runs
from its bundled virtualenv, whose `requests` uses its own `certifi` bundle, so
`update-ca-certificates` on its own changes nothing. The certificate also has
to carry a `subjectAltName` for the host — a CN-only one fails verification
however it was installed.

## Running

```sh
.venv/bin/jitsi-audio-bridge --config config.ini
```

| Flag | Meaning |
|---|---|
| `--config PATH` | Configuration file to read |
| `--process-dir PATH` | Process an existing meeting directory and exit, instead of serving — see [Batch mode](#batch-mode) |
| `--upload-video PATH` | Upload the recording that belongs to an existing meeting directory, and nothing else — see [Archiving the video](#archiving-the-video) |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default `INFO`) |
| `--version` | Print the version and exit |

Exit status is `0` on a clean shutdown, `1` if the listener cannot bind, and `2`
for a configuration problem. `SIGINT` and `SIGTERM` shut down gracefully,
letting in-flight post-processing finish.

The recordings directory is created and checked for writability at startup, so a
`ProtectSystem`/`ReadWritePaths` mismatch is reported immediately rather than at
the end of the first meeting.

## Wire protocol

> **This contract is not verified.** It is taken from the behaviour the code was
> written against, and is documented here because the sender lives outside this
> repository. See [Limitations](#limitations), and
> [docs/jitsi-integration.md](docs/jitsi-integration.md) for how stock Jitsi's
> sender differs from it.

The server accepts a single route, `/transcribe`. A connection to any other path
is closed with code `1008`.

```
ws://<host>:<port>/transcribe?sessionId=<id>
```

`sessionId` names the directory the meeting is recorded into. It is sanitised
before use — anything outside `[A-Za-z0-9._-]` becomes `_`, leading and trailing
dots are stripped, and it is capped at 64 characters — so a traversal attempt
such as `sessionId=../../tmp/pwned` records into `<recordings_dir>/tmp_pwned`.
A connection with no `sessionId` uses `session_default`.

### Text frames — control

Any number of JSON control frames may be sent. Each one is validated and written
to `metadata.json` atomically, replacing the previous one, so **the last control
frame is the one that counts**.

```json
{
  "meeting_url": "https://meet.example.com/Weekly-Planning",
  "participants": [
    {"user": {"id": "a1b2c3", "name": "Alice", "email": "alice@example.com"}},
    {"user": {"id": "d4e5f6", "name": "Bob", "email": "bob@example.com"}}
  ]
}
```

There is **no `room_name` field**. The room is the last path segment of
`meeting_url` — here `Weekly-Planning` — which is how Jitsi identifies a
meeting. A literal `room_name` is still honoured if a sender provides one.

Participants are read from either the nested `user` object or the flat level,
and both spellings of each field are accepted: `email` or `mail`, `name` or
`display_name`. An address that looks like an address is used as a recipient,
and the same participant is mapped for attribution by *both* their `id` and
their address, because a recording may be named after either.

If nothing structured yields an address, the raw document is swept for
anything shaped like one before falling back to `fallback_recipient` — an
address buried in an unexpected field is still better than mailing the admin.

### Binary frames — audio

```
┌────────────────────┬─────────────────────────────┐
│ 16 bytes           │ remainder                   │
│ participant id     │ exactly one Opus packet     │
│ ASCII, NUL-padded  │ (no container, no RTP hdr)  │
└────────────────────┴─────────────────────────────┘
```

One complete Opus packet per WebSocket message. The identifier is decoded as
UTF-8 with trailing NULs removed, then sanitised exactly like `sessionId`, and
must match a participant `id` from the control frame for the speaker's name to
appear in the transcript.

The identifier is matched against participant ids and addresses to attribute
each recording to a speaker; if nothing matches, the filename is used.

The sender must forward the **raw Opus payload** with RTP headers already
stripped. This bridge does not de-RED: if redundancy encapsulation is enabled,
payloads will not decode as Opus and the affected participant's recording will
be near-silent. Decode failures are counted per participant and logged at
session end, which is the signal to look for.

Each packet is decoded straight to 16 kHz mono. A packet that libopus rejects is
counted and skipped — it never aborts the recording.

### Text frames — stock Jitsi's media-json

The same route also accepts the framing stock Jitsi's JVB uses for bridge-based
transcription (see [Jitsi integration](#jitsi-integration)): JSON text frames
with an `event` key. A text frame carrying that key is always treated as a
media-json event and never as a control frame, so the key is reserved.

| `event` | Handling |
|---|---|
| `info` | Logged (application, version, region); ignored |
| `start` | Logged (tag, endpoint id, format); no file is created from it alone |
| `media` | Base64-decoded to one Opus packet, appended to `participant-<tag>.wav` |
| `ping` | Answered with `{"event":"pong","id":<same>}`; the JVB requires it |
| `session-end` | Finalises the recording without waiting for the close |
| anything else | Logged and ignored; a malformed event is counted, never fatal |

Frames are dispatched by shape, so both framings can be served on one route
without negotiating a mode.

The protocol carries no participant names, addresses or room name. A meeting
recorded this way is therefore summarised as "General Meeting", its speakers
are attributed by source tag, and the mail goes to `[smtp]
fallback_recipient`. [docs/jitsi-integration.md](docs/jitsi-integration.md)
documents the framing in full.

## Jitsi integration

Stock Jitsi can feed this daemon directly. With *bridge-based transcription*,
Jicofo tells the JVB to open one WebSocket per conference and forward every
participant's Opus audio over it, tagged per participant — the media-json
framing described above, which the daemon receives.

[docs/jitsi-integration.md](docs/jitsi-integration.md) has the Jitsi-side
configuration (Jicofo's `transcription.url-template`, the Prosody gating, the
`config.js` flag) and the exact protocol the JVB sends. Two things to plan for:
the framing carries no participant names or addresses (mail falls back to
`fallback_recipient`), and a reconnecting JVB reuses its `sessionId`.

`python3 -m tools.verify_jitsi` checks a deployment against that document and
is read-only by default: `--only config` parses the Jicofo, Prosody and client
files (finding, for example, a `transcription` block that is still commented
out, or a Record button whose Jicofo brewery is missing), `--only probe`
connects to the configured URL as the JVB would and
requires the pong, and `--only logs` scans recent
`jitsi-videobridge2`/`jicofo` journal entries for the connect lifecycle. Every
failure prints the fix; exit status is 1 if any check failed.

`--fix --bridge-url bridge.example.com` goes further and writes the remedy as
`<file>.new` beside the file it would change — Jicofo's transcription block,
the Prosody module, its enablement, the room-metadata component and the
`features_identity` entry that publishes it to clients, the client
configuration — leaving the originals untouched, with the diff/move/restart
commands printed for each. `--output-dir` stages them elsewhere when `/etc` is
not writable.

## Batch mode

The same pipeline also runs over a meeting directory that already exists on
disk, for recordings produced by something other than this bridge:

```sh
jitsi-audio-bridge --config config.ini --process-dir /srv/recordings/Weekly-Planning
```

It transcribes, summarises and emails, then exits: `0` if a summary was sent,
`1` if it was not, `2` for a bad directory or configuration. Nothing is served
and the recordings directory is not required.

Two directory shapes are handled, because both occur in practice:

| Shape | Files | Handling |
|---|---|---|
| One file per participant | `participant-<id>.wav`, `<address>_audio.wav` | Each transcribed separately and attributed to its speaker |
| A single master recording | any `.wav`, `.mp4`, `.m4a`, `.mkv` | Extracted to mono 16 kHz with ffmpeg, then transcribed as one speaker |

Participant files win if both are present, and `extracted_audio.wav` is never
treated as a source — it is this program's own output.

This is the batch counterpart to the WebSocket server, not a replacement: the
capture path handles live meetings, and this handles recordings that were
already written to disk. Both share the same transcribe, summarise and email
code, so a fix to one applies to the other.

## Running as a service

```sh
sudo useradd --system --no-create-home --shell /usr/sbin/nologin jitsi-bridge
sudo install -d -o jitsi-bridge -g jitsi-bridge /srv/recordings
sudo install -d -m 0755 /etc/jitsi-audio-bridge
sudo cp config.ini /etc/jitsi-audio-bridge/config.ini
sudo install -m 0600 -o root -g root /dev/null /etc/jitsi-audio-bridge/env
sudo cp systemd/jitsi-audio-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jitsi-audio-bridge
```

`recordings_dir` must match `ReadWritePaths=` in the unit. Under
`ProtectSystem=strict` the rest of the filesystem is read-only, and a missing
directory makes the unit fail to start.

The unit is hardened fairly aggressively. These are the directives most likely
to need relaxing, and why:

| Directive | If it causes trouble |
|---|---|
| `ProtectHome=yes` | Breaks if the virtualenv, config, or recordings live under `/home`. Move them to `/opt`, `/etc`, `/srv`. |
| `ReadWritePaths=/srv/recordings` | Must match `recordings_dir`. Add any other path the daemon writes to. |
| `RestrictAddressFamilies` | `AF_UNIX` and `AF_NETLINK` are needed for name resolution, not just for local sockets. |
| `MemoryDenyWriteExecute=yes` | Verified working with the ctypes-loaded libopus. First to drop if an unexplained crash appears. |
| `SystemCallFilter=@system-service` | Add `SystemCallLog=@system-service` temporarily to find a blocked call. |
| `TimeoutStopSec=900` | Post-processing can run for minutes; the 90 s default would kill it mid-transcription. |

## Test environment

Nothing in this repository speaks the sender half of the protocol, so `tools/`
provides one, together with stub Whisper, Ollama and SMTP services. That is
enough to run the whole pipeline on a laptop, with no GPU, no model download
and no mail relay.

```sh
# Run a three-person, ten-second meeting and report what came out.
python3 -m tools.testenv --auto --participants 3 --duration 10

# Or bring the environment up and drive it by hand from another shell.
python3 -m tools.testenv
```

`--auto` starts everything, runs a scripted meeting, prints the resulting
recordings, transcript, prompts and email, then tears it all down. Without it,
the environment stays up and prints the URL to point the sender at.

| Component | What it does |
|---|---|
| `tools.testenv` | Starts the stubs and the real daemon together, wired with a generated `config.ini` |
| `tools.send_meeting` | The sender simulator: speaks the wire protocol from the [Wire protocol](#wire-protocol) section |
| `tools.stubs` | Stub Whisper, Ollama, SMTP and S3, each runnable on its own |
| `tools.sample_audio` | Synthesises speech WAVs with ffmpeg, for the replay path |

### Driving it by hand

```sh
# Tones: proves framing and decoding, but Whisper correctly returns nothing.
python3 -m tools.send_meeting --participants 3 --duration 30

# Real speech: exercises transcription and speaker attribution end to end.
python3 -m tools.sample_audio --outdir /tmp/samples
python3 -m tools.send_meeting --audio wav \
    --wav /tmp/samples/alice.wav --wav /tmp/samples/bob.wav --wav /tmp/samples/carol.wav \
    --participant alice:Alice --participant bob:Bob --participant carol:Carol

# Stock Jitsi's framing: JSON events, base64 Opus, ping/pong, no metadata.
python3 -m tools.send_meeting --protocol media-json --participants 2 --duration 10

# Edge cases worth trying.
python3 -m tools.send_meeting --audio none        # no audio: nothing should be emailed
python3 -m tools.send_meeting --no-metadata       # no control frame: no room, no recipients
python3 -m tools.send_meeting --session-id '../../tmp/escape'   # traversal attempt
```

The stub SMTP server writes every accepted message to `mail/message-NNN.eml`
under the work directory, so you can read the summary that was sent rather than
taking a log line's word for it. The stub Whisper reports the format of the
audio it was handed, which catches a silent or malformed recording that a
frame count alone would miss.

Because `tools/send_meeting.py` is the only implementation of the sender side,
it doubles as the way to settle the open question in
[Limitations](#limitations): run it against the bridge, then compare what the
bridge records with what the real sender produces.

## Output layout

```
<srv/recordings>/<sessionId>/
├── metadata.json            # the last control frame received — see below
├── timeline.json            # who spoke when, captured live — see below
├── participant-<id>.wav     # 16 kHz, mono, 16-bit PCM, one per participant
├── extracted_audio.wav      # only when a master recording had to be extracted
├── transcript.txt           # written once post-processing succeeds
├── transcript.corrected.txt # only with [ollama] correct_transcript
├── summary.md               # the LLM summary, likewise
└── video.json               # only with [s3]: where the recording went
```

`timeline.json` is written for sessions fed by the JVB's media export (the
binary framing carries no timing): the session's start, how much audio each
participant produced, and every speaking turn with two clocks — when it began
on the session, and where it begins in that participant's WAV. It is what
makes the transcript interleaved; see [Limitations](#limitations) for what it
cannot do. The `.turns/` directory it is transcribed through is removed
afterwards.

A session recorded over more than one connection — a transcriber restarted, a
bridge reconnected — holds one WAV per connection: the first as
`participant-<id>.wav`, the next as `participant-<id>-2.wav`, and so on. They
are transcribed as what they are, the same speaker carrying on, and the
timeline keeps their turns apart because a turn's offset only means something
next to the file it came from.

`metadata.json` is written by the **control frame** path, so it exists only when
the sender sent one. The stock-Jitsi path has no control frame at all — the
JVB's framing carries no meeting metadata — so a Jitsi-driven session has no
such file, and post-processing runs on the defaults: the mail goes to
`[smtp] fallback_recipient`, the summary is titled "General Meeting", and
speakers are attributed by their source tag rather than a name. See
[Limitations](#limitations).

Where a control frame did arrive, `transcript.txt` is attributed by name:

```
[Alice]: Let's start with the release schedule.

[Bob]: I'll have the migration ready by Thursday.
```

Note that blocks are ordered by participant, not by time — see
[Limitations](#limitations).

### Archiving the video

If Jibri recorded the meeting and `[s3]` is configured, the recording is copied
to the bucket — never *instead* of the mail, because a broken endpoint must not
cost a meeting its summary, and by default not before it either, because a long
upload should not delay a transcript.

With `link_in_mail` on this is turned around on purpose: the mail *waits* for
the recording, so that the link in it works the moment somebody reads it — see
[The link in the mail](#the-link-in-the-mail).

What was uploaded is recorded in the meeting directory, which is what makes the
upload idempotent and what stops a second meeting in the same room from
claiming the same file:

```json
{
  "source": "/srv/jibri-recordings/<jibri-session>/Room_2026-10-08-10-11-12.mp4",
  "bucket": "meetings",
  "key": "videos/Room/Room_2026-10-08-10-11-12.mp4",
  "url": "https://minio.example.com/meetings/videos/Room/Room_2026-10-08-10-11-12.mp4",
  "size": 104857600,
  "confirmed_size": 104857600,
  "room_name": "Room",
  "uploaded_at": "2026-10-08T10:15:00+00:00"
}
```

`confirmed_size` is what the endpoint itself reports for the object, and
`delete_after_upload` only removes the local file when it matches both the size
that was uploaded and the size on disk: an upload that half-finished must not
take the only copy with it.

The recording is looked for by room name and by time, and it is given up to
`wait_seconds` to appear, because the file is still being written when the
meeting ends — somebody has to stop the recording, which can be well after the
transcript is written. If nothing matches, the log says which rooms *were*
found, which is usually enough to see why:

```
no recording of 'Standup' in /srv/jibri-recordings; found 'Retrospective', 'Daily' (outside the meeting)
```

To archive the videos of meetings that were processed before an endpoint was
configured, point the daemon at the directory and let it upload without
transcribing anything again:

```sh
jitsi-audio-bridge --upload-video /srv/recordings/<sessionId>
```

### The link in the mail

With `[s3] link_in_mail = true` the summary mail also says where the recording
is, and the daemon does the upload *before* sending it:

```
Recording (available until 2026-10-15):
https://minio.omnia.amarulasolutions.com/meetings/videos/Team-Sync/Team-Sync_2026-10-08-10-11-12.mp4?X-Amz-Signature=…
```

Four things are worth knowing about that link:

- **It is a bearer token.** Anyone the mail reaches can download the recording
  until it expires, without logging in anywhere. Mail gets forwarded and
  archived; set `link_expiry_seconds` to the shortest window that still lets
  people fetch it, and `0` only when the bucket itself is meant to be readable
  by anyone.
- **The mail waits for the recording**, up to `link_wait_seconds`. It gives up
  at once when there is nothing recorded for that room, and either way the mail
  goes out — a meeting nobody recorded is mailed with no recording line, not
  held back. While it waits, the meeting holds one of the
  [processing slots](#ai) that `MAX_CONCURRENT_JOBS` bounds, so a very slow
  upload on a busy daemon delays other meetings' transcripts.
- **Recipients have to be able to reach the address in it.** The daemon often
  uploads to one it cannot: `link_endpoint` is the name to set when the bucket
  is reached from outside by another one, and it is the name the link's host
  and the endpoint is validated against.
- **Whatever sits in front of the bucket has to pass the request through
  unchanged** — the query string included, and the `Host` header most of all,
  since the signature covers both. A proxy that rewrites either, or that puts
  its own login in front of the bucket, turns every link into a
  `SignatureDoesNotMatch` or a login page.

The link is signed with the same credentials as the upload and is not stored:
`video.json` keeps the plain, permanent address of the object, so a meeting
archived before this was switched on can be linked to later.

## Troubleshooting

**The unit cannot bind — `address already in use`.**
Check what already holds the port: `ss -ltnp | grep 8080`. On the machine this
was developed on, `127.0.0.1:8080` is taken by a SeaweedFS container listening
on `0.0.0.0:8080`, so the default configuration cannot bind. Either stop the
other service or change `[server] port`.

**Every connection closes with `1011`.**
That was the original bug: the handler took `(websocket, path)`, but websockets
13 and later pass a single argument. `websockets>=13` is now a hard dependency.

**No WAV files are produced.**
Check the log for decode failures. Confirm libopus is loadable:
`python3 -c "import ctypes.util; print(ctypes.util.find_library('opus'))"` should
print `libopus.so.0`. If it prints `None`, install `libopus0`.

**`Could not find the libopus shared library`.**
The daemon exits with this message at startup. `apt install libopus0`, or set
`LD_LIBRARY_PATH` so `find_library` can see it.

**Recordings land in one directory regardless of meeting.**
The sender is not including `?sessionId=`. The log warns when it falls back.

**No email arrives.**
The summary is skipped, with a logged reason, when no participant produced any
transcript text or when there were no usable recipients. Check the `[smtp]`
settings and the log; a send failure never discards `transcript.txt`.

**The transcript is empty or garbled.**
Look for the per-participant dropped-packet counts at session end. A high ratio
means the payloads are not plain Opus — most often RED is enabled on the sender.

**Jitsi's Record button says "All recorders are currently busy".**
That is Jibri's business, not this daemon's: the bridge is started by Jicofo
only for *transcription*, so recording can be broken while transcription works
perfectly. Jicofo answers `busy` whenever its recorder pool has no available
instance, and an empty pool is indistinguishable from a busy one from the
outside — so this message usually means no Jibri ever registered, not that one
is occupied. Three things have to line up, and
`jitsi-audio-bridge-verify --only config` reports all three: Jicofo's
`jicofo.jibri.brewery-jid`, a Prosody MUC at that JID's domain (the stock
`internal.auth.<domain>` component), and a Jibri that actually logs into it —
`journalctl -u jicofo | grep 'brewery instance'` shows the last one registering.

The **room name** has to match as well, and nothing checks that for you: Jicofo
watches a room nobody enters and reports it as a busy pool, exactly as if no
recorder existed. `jibribrewery` is the convention, but what counts is the
`control-muc` in Jibri's own `jibri.conf` — the checker reads it and compares
the two sides directly (`--jibri-conf` when Jibri runs on another host).

And on a host that runs both Jibri and this bridge, **check
`recording.recordings-directory`**. Jibri's package and this one both default to
`/srv/recordings`, which only one user can own: Jibri runs as `jibri`, the
bridge as `jitsi-bridge`, and the bridge's package claims the directory. Jibri's
attempt then fails with `ErrorCreatingRecordingsDirectory … SYSTEM` and a
`AccessDeniedException` on its session directory, and because a system error
marks Jibri unhealthy — a state it does not re-advertise until something changes
— Jicofo reads it as "all recorders are currently busy" from then on, however
healthy the rest of the deployment is. Give Jibri a tree of its own:

```sh
sudo grep -rn "recordings-directory\|recording_directory" /etc/jitsi/jibri/
sudo install -d -o jibri -g jibri -m 0750 /srv/jibri-recordings
# set it where the grep found it — recordings-directory in jibri.conf, or the
# legacy config.json's recording_directory — then:
sudo systemctl restart jibri
```

The two spellings matter: Jibri's own default is `/tmp/recordings`, so a value
of `/srv/recordings` is always written down somewhere, and on a host upgraded
from an older Jibri that place is the legacy `config.json`.

The checker reports the collision as `recording.directory`, comparing Jibri's
setting with this bridge's own `[storage] recordings_dir`.

**A `SIGKILL` left an unreadable WAV.**
The `wave` module writes the real length into the header only on `close()`. The
daemon closes every file in a `finally`, so this needs a hard kill; see
[REVIEW.md](REVIEW.md) for the recovery options.

## Limitations

- **The binary frame format is unverified.** The metadata shape is known from a
  working reference implementation, but the `[16-byte id][Opus]` framing is not:
  the sending side is not in this repository. Stock Jitsi does export
  per-participant Opus over a WebSocket — the JVB's media export, driven by
  Jicofo's `transcription.url-template` — but it frames the audio as JSON media
  events with base64 payloads, not as binary frames, so it is not the sender
  this framing came from — the daemon receives that path separately, keyed by
  source tag (see [docs/jitsi-integration.md](docs/jitsi-integration.md)).
  `tools/send_meeting.py` encodes the binary assumption rather than validating
  it.
- **A media-json meeting loses its participant names.** Stock Jitsi's framing
  carries no names, addresses or room name, so by default speakers are
  attributed by their source tag, the summary is titled "General Meeting", and
  the mail goes to `[smtp] fallback_recipient`. Recovering them means
  correlating `sessionId` with the conference elsewhere — which
  [docs/jitsi-integration.md](docs/jitsi-integration.md) §5 does with a Prosody
  module and `[storage] session_metadata_dir`.
- **Ordering is per speaker, not per sentence.** A speaking turn is the unit:
  two people talking over each other come out as two turns that overlap in
  time, and one long turn split at 30 seconds keeps only that resolution. The
  timestamps are offsets from the session start (the wall clock is in
  `timeline.json`), and a session recorded without `capture_timeline` has no
  timing at all — that cannot be recovered afterwards.
- **Sequential transcription.** Participants are transcribed one after another.
  A long meeting is slow, and one Whisper timeout costs that participant's text.
- **Retries are bounded and narrow.** A 5xx answer or a failed connection is
  retried up to `[ai] max_attempts` times, waiting twice as long each time
  (1 s, 2 s, 4 s, 8 s); turns that still produced nothing are tried again once
  the rest of the meeting is done, ten seconds later, because an intermittent
  service is usually back by then.
  A participant whose every turn failed is handed over as one whole recording
  instead — and if that fails too, their turns join the retry pass. Anything
  else — a 4xx, an unreadable file, SMTP — is logged and skipped, as before.
- **No retention policy.** Recordings accumulate indefinitely, here and in the
  bucket. `[s3] delete_after_upload` frees the local copy once the endpoint has
  confirmed it; nothing prunes either side after that.
- **The video is matched by room and time, not by identity.** Jibri does not
  tell the daemon which meeting it recorded, so a recording is the one whose
  room name and ending time fit this meeting's. Two meetings in one room at
  once — or a recording started by hand well before the meeting — can be
  matched wrongly, and a recording already uploaded by another meeting is
  deliberately not a candidate. `video.json` records what was chosen, so a
  wrong match is visible after the fact.
- **Last control frame wins.** Participant metadata is replaced, not merged, so
  someone who left before the final frame may lose their name.

See [REVIEW.md](REVIEW.md) for the full list of known issues and deferred work.

## Development

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

.venv/bin/pytest tests/              # unit tests, no network needed
python3 tests/smoke_test.py          # end-to-end, starts the stub services
python3 -m tools.testenv --auto      # the same environment, to poke at
.venv/bin/ruff check src tests tools # lint
```

`tests/test_units.py` covers configuration precedence, identifier sanitisation,
frame splitting and metadata parsing. `tests/test_tools.py` covers the Opus
encoder and the tooling. Neither needs audio fixtures: the tests encode their
own tone.

`tests/smoke_test.py` is the end-to-end check. It stands up the
[test environment](#test-environment), drives it with the sender simulator, and
asserts on the WAV files, the transcript, the prompts sent to Ollama, the
delivered message, and the negative cases — path traversal, a wrong request
path, an audio-less session, and a malformed control frame. It is the test that
would have caught all three original blockers.

Modules are deliberately isolated:

| Module | Responsibility | Knows about |
|---|---|---|
| `config` | Resolve configuration | files, the environment |
| `audio` | Decode Opus, parse metadata | libopus |
| `ai_client` | Talk to Whisper and Ollama | HTTP |
| `mailer` | Send the summary | SMTP |
| `s3_upload` | Archive the meeting's recording | S3, Jibri's recordings |
| `daemon` | Serve WebSockets, run the pipeline | asyncio |

`config` is the only module that reads the environment or parses a config file;
a test enforces this.

`tools/` sits outside the package and is not installed: it is the test
environment described [above](#test-environment), and nothing in `src/` depends
on it.

## Licence

Copyright 2026 Amarula Solutions, under the
[GNU Affero General Public License, version 3](LICENSE) — `AGPL-3.0-only`, not
"or later", and with no warranty of any kind.

In practice, for whoever runs this:

- **Using it, changing it and passing it on are all fine**, as long as what
  comes out stays under the same licence and carries this notice.
- **Offering it to users over a network — a modified copy, on your own host —
  means those users have to be able to get the source**, modifications
  included. That is the whole point of the AGPL, and it applies to a copy
  reached over a WebSocket or a mail link, not to running it for yourself:
  a meeting recorded and transcribed on your own machines, with no outside
  users, triggers nothing.
- **Selling it, or offering it as a hosted service without publishing your
  changes, needs a different licence from Amarula Solutions** — the AGPL is
  not the only terms available for it, it is the ones it comes with.

The dependencies keep their own licences and are all permissive: Apache-2.0
(boto3, botocore, requests, s3transfer), MIT (jmespath, six, urllib3,
charset_normalizer), BSD-3-Clause (idna, websockets), MPL-2.0 (certifi),
plus libopus and Python themselves. Nothing in the chain constrains the
licence above. The `.deb` carries them in its virtualenv and accounts for each
one in its `copyright` file, which the build refuses to produce without.

The recordings this daemon makes are a different matter entirely: they are the
meeting participants' personal data, and no licence here grants anybody the
right to record them. That is between the people in the meeting, their
employer, and whichever law applies to them.
