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
- [Install](#install)
- [Configuration](#configuration)
- [Running](#running)
- [Wire protocol](#wire-protocol)
- [Running as a service](#running-as-a-service)
- [Test environment](#test-environment)
- [Output layout](#output-layout)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Development](#development)

## Requirements

| | |
|---|---|
| Python | 3.11 or newer (developed against 3.14) |
| libopus | `libopus.so.0` — `apt install libopus0`. The `-dev` package is **not** needed. |
| Whisper | Any HTTP endpoint accepting `{"audio_base64": ..., "filename": ...}` and returning `{"text": ...}` |
| Ollama | Any HTTP endpoint accepting the `/api/generate` request shape |
| SMTP | Any relay |

ffmpeg is **not** required. Audio is decoded in-process through libopus, which
is both faster and less fragile than piping packets to a subprocess.

## Install

The daemon reads its configuration from `config.ini`. Copy the example and edit
it — `config.ini` itself is gitignored, because it is the one file that may end
up holding an SMTP password.

```sh
git clone <this repo> /opt/jitsi-audio-bridge
cd /opt/jitsi-audio-bridge

python3 -m venv .venv
.venv/bin/pip install .
cp config.ini.example config.ini
$EDITOR config.ini
```

Install into `/opt`, **not** `/home`: the shipped systemd unit sets
`ProtectHome=yes`, which makes `/home` inaccessible to the service.

For development, install editable with the test and lint extras:

```sh
.venv/bin/pip install -e '.[dev]'
```

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

### `[whisper]`

| Option | Type | Default | Environment |
|---|---|---|---|
| `url` | URL | `https://whisper.internal.domain.com/transcribe-b64` | `JITSI_AUDIO_BRIDGE_WHISPER_URL` |
| `timeout` | seconds | `600` | `JITSI_AUDIO_BRIDGE_WHISPER_TIMEOUT` |
| `verify_tls` | boolean | `true` | `JITSI_AUDIO_BRIDGE_WHISPER_VERIFY_TLS` |

### `[ollama]`

| Option | Type | Default | Environment |
|---|---|---|---|
| `url` | URL | `https://ollama.internal.domain.com/api/generate` | `JITSI_AUDIO_BRIDGE_OLLAMA_URL` |
| `model` | string | `qwen2.5:14b-instruct` | `JITSI_AUDIO_BRIDGE_OLLAMA_MODEL` |
| `timeout` | seconds | `600` | `JITSI_AUDIO_BRIDGE_OLLAMA_TIMEOUT` |
| `verify_tls` | boolean | `true` | `JITSI_AUDIO_BRIDGE_OLLAMA_VERIFY_TLS` |

### `[smtp]`

| Option | Type | Default | Environment |
|---|---|---|---|
| `host` | hostname | `127.0.0.1` | `JITSI_AUDIO_BRIDGE_SMTP_HOST` |
| `port` | 1–65535 | `25` | `JITSI_AUDIO_BRIDGE_SMTP_PORT` |
| `user` | string | *(empty)* | `JITSI_AUDIO_BRIDGE_SMTP_USER` |
| `password` | string | *(empty)* | `JITSI_AUDIO_BRIDGE_SMTP_PASSWORD` |
| `sender` | address | `no-reply@internal.domain.com` | `JITSI_AUDIO_BRIDGE_SMTP_SENDER` |
| `fallback_recipient` | address | `admin@internal.domain.com` | `JITSI_AUDIO_BRIDGE_SMTP_FALLBACK_RECIPIENT` |
| `use_starttls` | boolean | `true` | `JITSI_AUDIO_BRIDGE_SMTP_USE_STARTTLS` |

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

## Running

```sh
.venv/bin/jitsi-audio-bridge --config config.ini
```

| Flag | Meaning |
|---|---|
| `--config PATH` | Configuration file to read |
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
> repository. See [Limitations](#limitations).

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
  "room_name": "Weekly Planning",
  "participants": [
    {"id": "a1b2c3", "name": "Alice", "email": "alice@example.com"},
    {"user": {"id": "d4e5f6", "name": "Bob", "email": "bob@example.com"}}
  ]
}
```

Both the flat and the nested `user` shape are accepted. `id` is what links a
participant to their audio; unnamed or id-less entries are ignored. If no
addresses are found, the summary goes to `fallback_recipient`.

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

The sender must forward the **raw Opus payload** with RTP headers already
stripped. This bridge does not de-RED: if redundancy encapsulation is enabled,
payloads will not decode as Opus and the affected participant's recording will
be near-silent. Decode failures are counted per participant and logged at
session end, which is the signal to look for.

Each packet is decoded straight to 16 kHz mono. A packet that libopus rejects is
counted and skipped — it never aborts the recording.

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
| `tools.stubs` | Stub Whisper, Ollama and SMTP, each runnable on its own |
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
├── metadata.json            # last control frame received
├── participant-<id>.wav     # 16 kHz, mono, 16-bit PCM, one per participant
└── transcript.txt           # written once post-processing succeeds
```

`transcript.txt` is the transcript, one block per participant, attributed by
name:

```
[Alice]: Let's start with the release schedule.

[Bob]: I'll have the migration ready by Thursday.
```

Note that blocks are ordered by participant, not by time — see
[Limitations](#limitations).

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

**A `SIGKILL` left an unreadable WAV.**
The `wave` module writes the real length into the header only on `close()`. The
daemon closes every file in a `finally`, so this needs a hard kill; see
[REVIEW.md](REVIEW.md) for the recovery options.

## Limitations

- **The wire format is unverified.** The sending side is not in this repository
  and could not be inspected. Stock Jitsi Videobridge Colibri WebSockets carry
  JSON control messages for Jicofo, not per-participant Opus media, so this
  format is specific to a custom sender. Confirm it against that sender before
  relying on any of it.
- **No conversational ordering.** No timestamps are recorded, so the transcript
  is one block per participant in filesystem order, not interleaved by time.
  This is the single largest quality limitation; recovering it means recording
  per-packet arrival times and interleaving before transcription.
- **Sequential transcription.** Participants are transcribed one after another.
  A long meeting is slow, and one Whisper timeout costs that participant's text.
- **No retries.** A failed Whisper, Ollama, or SMTP call is logged and skipped.
- **No retention policy.** Recordings accumulate indefinitely.
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
| `daemon` | Serve WebSockets, run the pipeline | asyncio |

`config` is the only module that reads the environment or parses a config file;
a test enforces this.

`tools/` sits outside the package and is not installed: it is the test
environment described [above](#test-environment), and nothing in `src/` depends
on it.
