# Code review — findings and dispositions

A review of the original seven-file version of this project, and what happened
to each finding. The code had never been run when it was written, and no part of
it had been executed.

Findings marked **Fixed** are in this change. The rest were consciously
deferred; each carries the reason and the shape of the eventual fix, so this
does not read as a to-do list to be actioned out of context.

## Fixed in this change

### 1. Every WebSocket connection failed immediately — **Fatal** — Fixed

`daemon.py` declared `async def handle_jvb_stream(websocket, path)`. websockets
13 changed the connection handler to take a single argument; on the installed
16.1 this raised `TypeError: handler() missing 1 required positional argument:
'path'` for every connection, which the server turns into a `1011` close. The
daemon could not have completed a single handshake.

Reproduced against websockets 16.1 before the fix. The handler now takes one
argument and reads the path from `websocket.request.path`.

### 2. No audio was ever decoded — **Fatal** — Fixed

`audio.spawn_ffmpeg_writer` ran:

```
ffmpeg -y -f opus -i pipe:0 -ar 16000 -ac 1 out.wav
```

There is no `opus` **demuxer** in ffmpeg. `opus` exists only as an Ogg Opus
**muxer**. Verified on ffmpeg 8.0.1:

```
[in#0] Unknown input format: 'opus'
Error opening input file /dev/null.
```

`stderr` was `DEVNULL`, so the failure was invisible. Every participant's
recording would have been silently empty.

Replaced with an in-process libopus decoder. Two rejected alternatives are worth
recording: the obvious `opuslib` wrapper was last released in 2018 with only
Python 2 classifiers, and its maintained fork declares support up to 3.13 while
this targets 3.14; wrapping each packet in an Ogg container just to reach
ffmpeg would have kept a subprocess and added a container parser. libopus's
decode API is four functions, so binding it directly costs about eighty lines
and no third-party dependency.

### 3. The Whisper/Ollama/SMTP pipeline ran on the event loop — **High** — Fixed

`process_completed_session` — transcription, summarisation and email delivery,
minutes of blocking network I/O — was called directly inside the async handler's
`finally`, alongside `proc.stdin.write()` and `proc.wait()`. The server was
frozen for the entire post-processing window, and every concurrent meeting's
frames were dropped meanwhile.

The subprocess is gone with finding 2; the pipeline now runs through
`asyncio.to_thread`, bounded by a semaphore. Per-packet work stays inline
because it is a decode plus a small buffered write.

### 4. Path traversal through `sessionId` — **High** — Fixed

```python
session_id = path.split("sessionId=")[-1].split("&")[0] if "sessionId=" in path else "session_default"
meeting_dir = os.path.join(RECORDINGS_DIR, session_id)
```

`sessionId=../../etc` escaped the recordings directory. The same value was also
used unsanitised as a directory name, and every connection without the parameter
collided on `session_default`.

Now parsed with `urlparse`/`parse_qs` and reduced to a single safe path
component: anything outside `[A-Za-z0-9._-]` is replaced, leading and trailing
dots are stripped, and the result is capped at 64 characters. Covered by tests
and by the end-to-end smoke test.

### 5. Participant identifiers were equally untrusted — **High** — Fixed

Found while fixing 4, and missed in the original review: the participant ID
prefixing every binary frame became part of a filename just as directly.

```python
p_id = message[:16].decode("utf-8", errors="ignore").strip()
wav_path = os.path.join(meeting_dir, f"participant-{p_id}.wav")
```

Two problems. `str.strip()` does not remove NUL — `'\x00'.isspace()` is `False`
— so the ordinary NUL-padded identifier produced a filename containing a null
byte, and `open()` raises `ValueError: embedded null byte`. And the field
permitted traversal, so `participant-../../x.wav` normalises out of the
recordings directory. One hostile or merely differently-padded sender could take
down a session.

Trailing NULs are now stripped from the raw bytes, and the decoded identifier
goes through the same sanitiser as `sessionId`.

### 6. TLS verification was disabled — **Medium** — Fixed

All three HTTP calls passed `verify=False`, making the transcript path
interceptable and silencing the `InsecureRequestWarning` that would have said
so. Now driven by `verify_tls` in `[whisper]` and `[ollama]`, defaulting to
enabled.

Note this is a **behaviour change**: against an endpoint with a self-signed
certificate, the original code connected and this will not. Set
`verify_tls = false` for those, having weighed it.

### 7. `room_name` reached an attachment filename and a header — **Medium** — Fixed

```python
msg.add_attachment(..., filename=f"{room_name}_transcript.txt")
msg['Subject'] = f"Meeting Summary: {room_name}"
```

`room_name` comes from the sender's control frame. A path separator in it
redirects the attachment name; a newline raises `ValueError` inside
`EmailMessage`. The filename is now sanitised to `[A-Za-z0-9._-]` and bounded,
the subject has its whitespace flattened, and both are covered by tests.

### 8. A failed connection still emailed an empty meeting — **Medium** — Fixed

`process_completed_session` ran in `finally` unconditionally, while
`parse_metadata` defaults `recipients` to `[]`, which falls through to
`fallback_recipient`. Every aborted connection — a port scan, a restarted
sender, a client that disconnected immediately — would have emailed the admin an
empty summary.

Post-processing now runs only when at least one frame arrived *and* at least one
participant produced audio.

### 9. `start_daemon` could not start on Python 3.14 — **Fatal** — Fixed

```python
asyncio.get_event_loop().run_until_complete(server)
```

Implicit event-loop creation was deprecated in 3.12 and removed in 3.14, where
this raises `RuntimeError: There is no current event loop in thread
'MainThread'`. Even had findings 1 and 2 not existed, the daemon could not have
started at all. Replaced with `asyncio.run()` over a `serve()` coroutine.

### 10. No `if __name__ == "__main__"`, no logging, no signal handling — **Low** — Fixed

All output was `print()` to stdout, which under systemd lands in the journal
with no levels and no timestamps. There is now a real entry point, stdlib
logging with a `--log-level` flag, and `SIGINT`/`SIGTERM` handling that lets
in-flight work finish.

### 11. Miscellaneous — Fixed

- `json.loads` on every text frame was unguarded, so one malformed control frame
  aborted the session. Now validated, and written atomically so no reader sees a
  half-written file.
- `getattr(f, "name", "audio.wav")` was evaluated after the `with` block that
  bound `f` had closed — working by accident. Now uses the path directly.
- `detect_language` swallowed every exception and returned `"English"`, hiding
  outages behind a plausible answer. It now logs the fallback.
- `smtplib.SMTP(...)` had no timeout and could hang indefinitely. Now bounded.
- `starttls()` ran only when a username *and* password were set, so an
  unauthenticated relay got no encryption even when it offered it. Now
  controlled by `use_starttls`.
- `int(os.getenv(...))` raised a bare `ValueError` with no indication of which
  setting was wrong. Configuration errors now name the option and its source.

## Security note — the API key in `claude.sh`

`claude.sh` exported a live `ANTHROPIC_API_KEY` in plaintext, alongside an
`ANTHROPIC_BASE_URL` pointing at a third-party API-compatible endpoint.

The file is now gitignored, and the repository had no commits at the time, so
**nothing leaked into git history** — verified by searching the whole history
for the credential. The key is nevertheless a real secret that has been sitting
unencrypted in a working directory, so it should be rotated. Delete or move the
file once that is done.

The value is deliberately not reproduced anywhere in this repository, including
in this document.

## Corrected against a working reference

A working implementation of the post-processing half was supplied after the
first pass, and contradicted several things this code had assumed. These are
corrections, not preferences: the reference is what actually runs.

### The metadata format was wrong — **High** — Fixed

`parse_metadata` looked for `meta["room_name"]`. **There is no such field.**
The room is the last path segment of `meeting_url`, which is how Jitsi
identifies a meeting. Every real meeting would therefore have been summarised
and emailed as "General Meeting", with a subject line naming the wrong room —
a silent, plausible-looking failure rather than an error.

Also unhandled: participants arrive nested under `user`, and the field names
have varied in the wild — `mail` as well as `email`, `display_name` as well as
`name`. Attribution was mapped only by participant id, but recordings are also
named after addresses, so those participants would have been attributed to a
filename. And when nothing structured yielded an address, the code fell
straight through to the admin fallback rather than looking any further.

All four are now handled, and attribution resolves by id *or* address with the
filename as a last resort.

### The summary prompt was markedly weaker — **Medium** — Fixed

The original prompt asked for three sections and nothing more. The reference
carries two rules that do real work:

- a speaker-attribution rule, without which the model flattens the `[Name]:`
  tags into unattributed prose — the difference between minutes and a wall of
  text;
- a language rule repeated against every section heading, because models
  otherwise translate the body and leave the headings in English.

Adopted verbatim, and a test asserts the generated prompt is byte-identical to
the reference, so the wording cannot drift by accident. The prompt is now built
from explicit `\n` rather than a triple-quoted block purely so no source line
has to be unreasonably long; the text sent is unchanged.

### Two calls to one endpoint were indistinguishable in tests — **Low** — Fixed

The stub Ollama service told the language probe from the summary request by
matching a phrase in the prompt. Changing the prompt to the reference's wording
silently broke that: the stub began answering the language question with the
summary text, which then became the language the summary was requested in. The
discriminator is now a phrase unique to the language prompt, with a comment
saying why it has to be one.

### Behaviour the reference had and this did not — **Medium** — Fixed

- **A single master recording.** The reference falls back to any `.wav`,
  `.mp4`, `.m4a` or `.mkv` in the directory and extracts a 16 kHz mono track
  with ffmpeg. Without it, a meeting recorded as one file was unprocessable.
  ffmpeg is genuinely the right tool here — unlike raw Opus packets, a real
  container is something it reads natively.
- **`summary.md` was not written or attached.** Only the transcript was. Both
  are attached now; the summary is also in the message body so it is legible
  without opening anything.
- **Batch mode.** The reference is invoked as `process_meeting.py <dir>` over a
  directory of already-recorded audio. That path now exists here too, as
  `--process-dir`, sharing the identical pipeline rather than duplicating it.

### Adopted differently

**Cleanup.** The reference deletes the audio, transcript and summary once the
email is sent. That is implemented, but behind `cleanup_after_send` and **off
by default**: those files are the only copy of the meeting, and a
misconfiguration that destroys one is not recoverable. It also runs only after
a *confirmed* send.

**TLS.** The reference passes `verify=False` to every request. That is now
`verify_tls` per endpoint, defaulting to enabled. If the internal endpoints use
a self-signed certificate, this will fail where the reference succeeded — set
it to `false` deliberately, or install the CA.

## Deferred

### Peak memory during transcription — **Medium**

`transcribe_audio` reads the whole WAV, base64-encodes it, and embeds that in a
JSON body: roughly 2.3× the file size resident at peak, per request. A one-hour
16 kHz mono recording is about 115 MB, so ~265 MB transiently. Tolerable for
short meetings and for one participant at a time, which is what the semaphore
allows.

*Fix:* stream the upload as `multipart/form-data` and change the Whisper
contract, or transcode to a compressed format before upload. Both are protocol
changes and belonged outside a review-and-package pass.

### No conversational ordering — **Medium, design**

No arrival timestamps are recorded. `process_completed_session` globs
`participant-*.wav` and concatenates blocks, so the transcript reads as a
sequence of monologues rather than a conversation, and the order is by
participant identifier.

*Fix:* record the arrival time of each packet (or each speech run) and
interleave before transcription. This is the largest single quality improvement
available, and it is the one that makes a meeting transcript actually readable.

### `metadata.json` is last-write-wins — **Medium**

Every control frame replaces the file wholesale, so only the final participant
list survives. A participant who joined early and left before the last frame
loses their name, and their audio is attributed to a bare identifier.

*Fix:* merge participants across frames, and record the merge. Whether the
control frame is a snapshot or an increment is a contract question for the
sender, so this should be settled with it rather than guessed at here.

### No idempotency — **Medium**

A sender that reconnects with the same `sessionId` re-transcribes, overwrites
`transcript.txt`, and sends a second email. Two concurrent connections with the
same identifier interleave writes into the same participant files and corrupt
both.

*Fix:* a session registry holding `capturing`/`processing`/`done`, rejecting a
second live connection with the same identifier and refusing to re-process a
finished one. A `session.json` recording the state would also make an
interrupted run detectable.

### `wave` header only finalises on close — **Low**

`Wave_write.close()` is what writes the true lengths into the RIFF header. A
`SIGKILL` between the first packet and `close()` leaves a file whose header
claims zero frames. Every recorder is closed in a `finally`, so this needs a
hard kill to trigger.

*Fix, if it matters:* the stdlib writer emits a canonical 44-byte header for
plain PCM, so a truncated file can be repaired deterministically —
`data_size = filesize - 44`, then patch the two length fields. Around a dozen
lines, and it can run at startup for any `participant-*.wav` with a zero frame
count.

### Whisper and Ollama failures are not isolated per participant — **Low**

A `requests` timeout aborts the loop in `process_completed_session`, so one slow
participant can cost the remaining ones their text. The transcription call
already returns `""` on failure rather than raising, but a timeout raised while
the connection is being established is not caught in every path.

*Fix:* wrap each participant's transcription in its own `try`, and consider a
retry with backoff. Likewise, no retry exists anywhere in the pipeline; a
transient Ollama failure loses the summary while keeping the transcript.

### The summary prompt is unbounded — **Low**

The full transcript is interpolated into the prompt. A long meeting will exceed
the model's context window, and Ollama's behaviour then is to truncate silently
or to fail.

*Fix:* chunk the transcript and summarise hierarchically.

### Operability gaps — **Low**

No metrics endpoint, no health check, and no readiness notification. `Type=simple`
means systemd reports the unit as started as soon as the process is spawned,
not when the port is actually bound.

*Fix:* `Type=notify` with `sd_notify` once the listener is up; a
`/healthz`-style endpoint or a periodic log line for liveness.

### Supply chain — **Low**

No lockfile and no hash pinning. Worth adding if this is deployed beyond a
single host: prefer `pip install --require-hashes` against a compiled lock.

### Not handled: redundancy (RED) and FEC — **Low**

If the sender enables RED encapsulation, every payload carries a redundancy
header and will not decode as Opus. Nothing detects this; the symptom is a
near-silent recording plus a high dropped-packet count at session end, which the
log now surfaces.

*Heuristic de-RED detection was deliberately not added.* It would be guesswork
that can misfire on legal Opus TOC bytes, and RED is off in Jitsi's default
configuration. The wire protocol section of the README states the requirement
instead.

### Unknown config keys are ignored — **Low**

A typo such as `[smtp] sever = ` is silently discarded, which is the classic
"why is my configuration not being applied" bug. The schema here is small
enough that rejecting unknown sections and options outright would be cheap.

*Fix:* a closed schema, with the offending keys listed in the error.

## Verification performed

- 140 unit tests, including an Opus encode/decode round trip against the real
  libopus binding, requiring no audio fixtures.
- A 41-check end-to-end smoke test driving the real daemon with stub Whisper,
  Ollama and SMTP services, asserting on the WAV files, the transcript, the
  prompts sent to Ollama, the delivered message, and the negative cases. It
  covers both the live capture path and batch mode, including a master
  recording that only ffmpeg can read.
- The generated summary prompt is asserted byte-identical to the working
  reference's, so its wording cannot drift.
- A round trip with real synthesised speech verified by audio level: a 48 kHz
  source at −15.4 dB mean came back as a 16 kHz recording at −15.0 dB, with
  each participant's distinct speech still separated. Checking levels rather
  than frame counts is what rules out a decoder emitting correctly-sized
  silence.
- `ruff check`, clean.
- `systemd-analyze verify`, clean.
- `MemoryDenyWriteExecute=yes` tested against the ctypes-loaded libopus under a
  transient systemd unit: it loads and decodes. Kept in the unit on that
  evidence.
- Error paths exercised: missing config, malformed value, an occupied port, and
  an unreachable bridge each exit with the documented status and a message
  naming the cause.

The test environment in `tools/` exists because none of this could be checked
without a sender, and nothing in the repository spoke the protocol. It also
means the checks above are repeatable rather than one-off.

**Not verified:** the binary frame format. The metadata schema is now known
from the working reference, but the `[16-byte participant id][Opus packet]`
framing is not — the sender is not in this repository and no Jitsi deployment
exists on the development host. `tools/send_meeting.py` encodes that assumption
rather than confirming it; pointing it at a real sender and comparing the two
is the way to settle it, and that has not been done.

Worth noting that the reference does not receive Opus at all: it reads WAV and
container files that something else has already written. So the frame format is
not something the reference can vouch for, and the live-capture half remains
the unproven part of this project.
