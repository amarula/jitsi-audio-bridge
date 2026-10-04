# Configuring Jitsi to feed the audio bridge

This is the Jitsi-side counterpart to the [wire protocol](../README.md#wire-protocol)
in the README: how to configure a Jitsi deployment so that the Jitsi
Videobridge (JVB) opens a WebSocket to a receiving service and forwards every
participant's Opus audio over it.

Jitsi calls this **bridge-based transcription**, and it is the path that
replaces the older Jigasi transcriber. The reference receiver is
[jitsi/opus-transcriber-proxy](https://github.com/jitsi/opus-transcriber-proxy),
which streams the audio to a speech-to-text backend and sends results back into
the meeting. The audio bridge in this repository is a simplified receiver: it
records the audio and transcribes it *after* the meeting, so it never sends
anything back except the keepalive replies the JVB requires.

> **Status.** Everything below was checked against the upstream sources listed
> at the [end of this document](#verified-against) on 2026-10-03, and the daemon
> implements this framing: point Jicofo's `url-template` at it and the JVB's
> audio is recorded. Two properties of the protocol shape what comes out — no
> participant names, addresses or room name arrive on it (mail goes to
> `[smtp] fallback_recipient`, the summary is titled "General Meeting", speakers
> are attributed by source tag), and a JVB reconnect reuses the same
> `sessionId`. See [Receiver behaviour](#receiver-behaviour).

## How the connection is established

```
  jitsi-meet          Prosody                  Jicofo                   JVB
      │                  │                        │                      │
      │  user enables    │                        │                      │
      │  transcription ─►│                        │                      │
      │                  │  room metadata:        │                      │
      │                  │   asyncTranscription=true                     │
      │                  │   recording.isTranscribingEnabled=true        │
      │                  │                        │                      │
      │                  │──── room metadata ────►│                      │
      │                  │                        │                      │
      │                  │                        │ builds the URL from  │
      │                  │                        │ the url-template     │
      │                  │                        │                      │
      │                  │                        │─ Colibri2 connect ──►│
      │                  │                        │  type=TRANSCRIBER    │
      │                  │                        │  url=ws://…          │
      │                  │                        │  ping=10s/3s         │
      │                  │                        │                      │
      │                  │                        │                      │── ws://<bridge>/transcribe
      │                  │                        │                      │   ?sessionId=<meeting-id>
      │                  │                        │                      │   ── info, start, media, ping ──►
```

**One WebSocket per conference, not one per participant.** The JVB opens a
single connection per meeting and multiplexes every participant's audio over
it; each audio frame carries a `tag` identifying the source (see
[What arrives on the socket](#what-arrives-on-the-socket)). With cascaded
bridges the connect is hosted on exactly one chosen Colibri session, so there is
still one socket per meeting.

Three consequences worth internalising before the step-by-step:

- `sessionId` in the URL is the conference's *meeting id*, substituted by Jicofo
  from its URL template. With the standard `muc_meeting_id` Prosody module
  (enabled by default in docker-jitsi-meet) that is a random UUID, **not** the
  human room name. The room name is not carried on this protocol at all.
- The URL is fixed once the Colibri session exists. Jicofo refuses a change
  mid-conference with `Changing to a different transcriber URL is not supported`,
  so a new template only takes effect for conferences created after the restart.
- If the JVB reconnects (restart, network loss, ping timeout), it reuses the
  same `sessionId`. A receiver must tolerate that, including the possibility of
  a second live connection for a meeting it is already recording.

## What has to be configured

| Component | File | Change |
|---|---|---|
| jitsi-meet (web) | `config.js` | Enable the transcription UI |
| Prosody | `prosody.cfg.lua` + a small module | Force `asyncTranscription` on the room |
| Jicofo | `jicofo.conf` | Point the transcriber connect at the audio bridge |
| JVB | — | Nothing: it is driven over Colibri2. Optional reconnect tuning only |

### 1. jitsi-meet — `config.js`

```javascript
transcription: {
    enabled: true,
},
```

This enables the feature in the client. Jicofo will not start the transcriber
until **both** of these room-metadata flags are set:

| Flag | Set by |
|---|---|
| `asyncTranscription = true` (top level of the room metadata) | Prosody, server-side (clients are blocked from setting it) |
| `recording.isTranscribingEnabled = true` | The client, when transcription/subtitles are turned on |

If nothing in the room ever sets the second flag, the transcriber never starts.
To transcribe unconditionally, force it server-side as well — see the note at
the end of the Prosody section.

### 2. Prosody — force `asyncTranscription`

Transcription is gated by per-room metadata stored by
`mod_room_metadata_component` under `room.jitsiMetadata`. The relevant key,
`asyncTranscription`, is server-controlled: the handbook says clients are
forbidden from setting it (a `blocked_metadata_keys` list in the component),
though the module versions inspected while writing this do not implement such a
list. Either way, set it server-side, as below — never rely on a client doing
it.

Create `mod_force_async_transcription.lua` on the Prosody plugin path (for
example `/usr/share/jitsi-meet/prosody-plugins/`):

```lua
-- mod_force_async_transcription.lua
-- Forces asyncTranscription=true on every room's metadata.
-- Enable on the main MUC component (e.g. conference.<domain>).

local util = module:require 'util';
local is_healthcheck_room = util.is_healthcheck_room;

module:hook('muc-room-created', function(event)
    local room = event.room;

    if is_healthcheck_room(room.jid) then
        return;
    end

    -- mod_room_metadata_component initializes this table at priority -1,
    -- so run after it.
    if not room.jitsiMetadata then
        room.jitsiMetadata = {};
    end

    room.jitsiMetadata.asyncTranscription = true;

    module:log('info', 'Forced asyncTranscription=true for room %s', room.jid);
end, -2); -- priority -2: after room_metadata_component (-1)
```

The `-2` priority makes the hook run *after* `mod_room_metadata_component`
(which runs at `-1` and creates `room.jitsiMetadata`).

Enable it on the MUC component:

```lua
Component "conference.example.com" "muc"
    modules_enabled = {
        -- ... existing modules ...
        "muc_meeting_id";
        "force_async_transcription";
    }
```

Then `systemctl restart prosody`.

> **Note.** This only forces `asyncTranscription`. For transcription to start,
> the room must also have `recording.isTranscribingEnabled` set — normally by a
> user turning transcription on. To transcribe every room unconditionally,
> extend the module to set the `recording` metadata too, or pre-seed it from
> whatever creates the room.

### 3. Jicofo — `jicofo.conf`

Jicofo builds the transcriber URL from a template in HOCON. Add this to
`jicofo.conf` (`/etc/jitsi/jicofo/jicofo.conf` on a package install):

```hocon
jicofo {
  transcription {
    url-template = "ws://bridge.example.com:8080/transcribe?sessionId={{MEETING_ID}}"

    ping {
      enabled = true
      interval = 10 seconds
      timeout = 3 seconds
    }
  }
}
```

| Key | Type | Default | Purpose |
|---|---|---|---|
| `jicofo.transcription.url-template` | string | — (transcription silently disabled) | WebSocket URL. Must contain `{{MEETING_ID}}`; `{{REGION}}` is optional and expands to the chosen bridge's region (`""` when it has none) |
| `jicofo.transcription.http-headers` | map | `{}` | Headers sent on the connect (auth tokens, Cloudflare Access service tokens, …) |
| `jicofo.transcription.ping.enabled` | boolean | `true` | Send keepalive pings over the connect |
| `jicofo.transcription.ping.interval` | duration | `10 seconds` | Ping period |
| `jicofo.transcription.ping.timeout` | duration | `3 seconds` | How long the JVB waits for a pong before it drops and reconnects |

Restart Jicofo afterwards (`systemctl restart jicofo`).

Notes:

- The path and query parameter are conventions of the receiving service, not of
  Jitsi — Jicofo substitutes the template verbatim. `/transcribe` and
  `sessionId` match the reference proxy and this repository's server.
- `{{MEETING_ID}}` is substituted without URL encoding. It is a UUID under
  `muc_meeting_id`; a receiver that uses it as a directory name should sanitise
  it, which this daemon already does.
- `sendBack=true` in the reference proxy's examples is that service's own query
  parameter (it asks for transcriptions to be returned). A record-only receiver
  does not need it.
- Per-room overrides can be supplied through the `room_metadata` component
  (`transcription.httpHeaders` and `transcription.urlParams`); those merge over
  the base config with per-room values taking precedence.
- **Docker (docker-jitsi-meet):** the shipped `jicofo.conf` ends with
  `include "custom-jicofo.conf"`. Put the block above in
  `~/.jitsi-meet-cfg/jicofo/custom-jicofo.conf`, which is mounted at
  `/config/custom-jicofo.conf`, and restart the `jicofo` container.

### 4. JVB — nothing to configure

No JVB setting opens the socket. Jicofo drives it over Colibri2 with a
`<connect type="TRANSCRIBER">` element carrying the URL, the headers and the
ping settings, and the JVB opens the WebSocket itself. Two things do belong in
the deployment's planning:

- **Network:** every JVB host must be able to reach the audio bridge at the
  configured address. In Docker, use the container/service name and a shared
  network, not `localhost` — `localhost` from inside the JVB container is the
  JVB container. Terminate TLS with a reverse proxy and use `wss://` if the
  socket crosses an untrusted network.
- **Reconnect tuning** (optional, `jvb.conf`): connections are retried on
  failure with exponential backoff.

| Key | Default | Purpose |
|---|---|---|
| `videobridge.exporter.max-reconnect-attempts` | unset (unlimited) | Give up after this many consecutive failures |
| `videobridge.exporter.base-delay` | `1 second` | First retry is immediate, then this, doubling per attempt |
| `videobridge.exporter.max-delay` | `30 seconds` | Backoff ceiling |
| `videobridge.exporter.stable-connection-threshold` | `30 seconds` | A connection must live this long before it resets the attempt counter, to prevent tight reconnect loops |

## What arrives on the socket

The JVB→service protocol is JSON text frames in a format derived from
VoxImplant's WebSocket media protocol (the same one the reference proxy
speaks). **The JVB never sends binary frames on this socket.**

| `event` | Direction | Meaning |
|---|---|---|
| `info` | JVB → service | Once, when the connection opens: `application`, `version`, and `region` when set |
| `start` | JVB → service | Announces one audio stream before its first `media`: `tag`, `mediaFormat`, `customParameters.endpointId` |
| `media` | JVB → service | One Opus packet: `tag`, `chunk`, `timestamp`, `payload` (base64), optional `audioLevel` / `vad` |
| `ping` | JVB → service | Keepalive; the service must answer `{"event":"pong","id":<same id>}` |
| `session-end` | JVB → service | The JVB is closing the connection |
| `sources` | JVB → service | Exported/requested source names; only for connects that declare them (translation), not plain transcription |
| `stop` | service → JVB | Part of the format, but only ever sent *to* the bridge (a translation peer bracketing a talk): the JVB does not emit it. Do not wait for one |
| `transcription-result` | service → JVB | Optional; results to inject into the meeting. A record-only receiver never sends this |

Example sequence for one participant:

```json
{"event":"info","application":"jitsi-videobridge","version":"2.3-123-gabcdef"}
{"event":"start","sequenceNumber":"1","start":{"tag":"<source-name>","mediaFormat":{"encoding":"opus","sampleRate":48000,"channels":2},"customParameters":{"endpointId":"<endpoint-id>"}}}
{"event":"media","sequenceNumber":"2","media":{"tag":"<source-name>","chunk":"42","timestamp":"1234567","payload":"<base64 Opus packet>"}}
{"event":"ping","id":1}
{"event":"pong","id":1}
{"event":"session-end"}
```

Framing rules that matter to a parser:

- `event` is the discriminator; unknown events must be ignored, not fatal.
- The numeric fields inherited from VoxImplant are **encoded as JSON strings**:
  `sequenceNumber`, `media.chunk` and `media.timestamp` are quoted. The
  additions to that format are natural numbers: `ping`/`pong` `id`,
  `media.audioLevel`, `media.vad` and `start.timestamp`. Null fields are
  omitted.
- `media.payload` is standard base64 of **one Opus packet** — the RTP payload
  with the RTP header already stripped and no container. It decodes directly
  with libopus, exactly like the binary framing this repository already handles.
- `tag` is the bridge-assigned source name for a participant's audio stream and
  is stable for the life of that stream; `start.customParameters.endpointId`
  carries the participant's endpoint id, which is the closest thing to a
  stable participant identity on this protocol.
- **Pings are mandatory if enabled.** With the defaults above (enabled, 10 s
  interval, 3 s timeout) a receiver that does not answer a ping is dropped and
  reconnected every ~13 seconds. Either implement the pong or set
  `jicofo.transcription.ping.enabled = false`.

## Receiver behaviour

Frames are dispatched per shape rather than per connection, so this framing and
the custom binary one are served on the same route without negotiating a mode:

- A text frame carrying an `event` key is a media-json event; any other text
  frame is still a legacy control frame, written to `metadata.json`. The
  `event` key is therefore reserved.
- `media` events are base64-decoded to one Opus packet each and appended to the
  recorder for the sanitised `tag`, producing `participant-<tag>.wav`. The
  recording is keyed by tag even if it arrives before the source's `start`
  event.
- `ping` is answered with a matching `pong`, which the JVB requires.
- `session-end` stops reading and finalises the recording without waiting for
  the TCP close.
- `info` and `start` are logged (application, version, tag, endpoint id,
  format); `sources`, `stop`, `transcription-result` and unknown events are
  ignored. A malformed event is counted and skipped — one bad frame never
  aborts a meeting.
- The binary `[16-byte id][Opus]` framing keeps working unchanged, including
  alongside media-json on the same socket.

Consequences of this path:

- **No participant metadata.** There are no names, addresses or meeting URL on
  this protocol. The daemon's metadata-driven attribution and recipient
  selection cannot work from it alone: speakers are attributed by tag, the
  summary is titled "General Meeting", and every meeting goes to
  `fallback_recipient`. Recovering recipients means correlating `sessionId`
  with the conference elsewhere (Prosody/Jicofo), which is outside this
  protocol.
- **The meeting id is opaque.** `sessionId` is a UUID, so the recording
  directory and anything derived from it are named by UUID, not by room.

Known hazards, already recorded in [REVIEW.md](../REVIEW.md): a reconnecting
JVB reuses its `sessionId` and the daemon has no idempotency (a reconnect
re-transcribes, re-emails, and two live connections interleave writes into the
same directory); and if RED is enabled for the endpoint, the payload is not
plain Opus and libopus will reject it (the per-participant dropped-packet count
at session end is the signal).

## Verifying

### Run the checker first

`tools/verify_jitsi.py` automates most of this and prints a fix per failure.
Run it on the Jitsi host; it is read-only. From a checkout it is
`python3 -m tools.verify_jitsi`; if the bridge was installed from the Debian
package (`make deb`), the same tool is on the PATH as
`jitsi-audio-bridge-verify`.

```sh
python3 -m tools.verify_jitsi                       # the config files (default)
python3 -m tools.verify_jitsi --only probe          # reach the bridge as the JVB would
python3 -m tools.verify_jitsi --only logs --since "10 min ago"
```

`config` reads `/etc/jitsi/jicofo/jicofo.conf` (with its includes, so a
`custom-jicofo.conf` the config does not include is reported), the Prosody site
config for the discovered domain (or `--domain`/`--prosody-config`/`--meet-config`),
the jitsi-meet `config.js`, and `jvb.conf`. It distinguishes the traps a manual
review misses: a `url-template` that is only present commented out, a
`transcription` block still commented in `config.js`, and no enabled Prosody
module that sets `asyncTranscription`. `probe` connects to the resolved
template (or `--url`), sends a `ping` and requires the `pong` when pings are
enabled, then sends `session-end`; it transmits no audio, so nothing is emailed.
`logs` classifies the JVB exporter lifecycle and Jicofo's transcription errors
from journald (or `--jvb-log`/`--jicofo-log` files). Exit status is 1 if
anything failed.

From the bridge's side, in order of what proves what:

1. **Watch the first bytes.** The first message on a healthy connection is
   `info` with `"application":"jitsi-videobridge"`. Seeing `start` then a
   steady stream of `media` means audio is flowing. A bare `wscat -c
   "ws://bridge.example.com:8080/transcribe?sessionId=test"` shows this, but
   note that it will not answer pings: type the `{"event":"pong",...}` reply by
   hand (or disable pings) if the connection drops after ~13 seconds.
2. **Jicofo's log.** Misconfiguration is explicit:
   `Transcription enabled, but no URL is configured.` means transcription is on
   but `url-template` is unset or Jicofo was not restarted.
3. **The JVB's log** (INFO) shows the lifecycle:
   `Websocket connected: true`, `Sending info to transcriber: {...}`,
   `Starting SSRC <ssrc> for endpoint <id>`, and on failure
   `Websocket closed with status ...` / `Ping timeout, reconnecting websocket`.
4. **Metrics.** The JVB registers per-exporter counters:
   `exporter_starts`, `exporter_packets_sent`, `exporter_websocket_failures`,
   `exporter_websocket_internal_errors`, `exporter_parse_failures`,
   `exporter_info_received`. A non-zero `exporter_packets_sent` with zero
   `exporter_parse_failures` means the JVB is sending audio and considers the
   peer's messages well-formed.
5. **Connectivity.** `ss -tnp | grep <bridge port>` on the JVB host shows the
   socket; if the connect never appears, the JVB cannot reach the bridge or the
   room metadata never satisfied the two-flag condition.

The simulator speaks both framings: `python3 -m tools.send_meeting --protocol
media-json` emits `info`, `start` and `media` events, pings as it streams,
reports how many pongs came back, then sends `session-end`. The smoke test
(`tests/smoke_test.py`) drives it through the whole pipeline and asserts on the
tag-keyed recordings, the transcript and the mail to the fallback recipient.

## Verified against

Upstream sources inspected on 2026-10-03 (current master at the time):

- [Jitsi handbook — Transcriptions (bridge-based)](https://jitsi.github.io/handbook/docs/devops-guide/transcription/) — the Prosody module, Jicofo keys and `config.js` shape.
- `jitsi/jicofo` — `TranscriptionConfig.kt` (keys and template rules), `ColibriV2SessionManager.kt` (one connect session per conference, `Connect.Types.TRANSCRIBER`, per-room overrides), `JitsiMeetConferenceImpl.java` (the two-flag gating and the "no URL" log), `jicofo-selector/src/main/resources/reference.conf` (ping defaults).
- `jitsi/jitsi-videobridge` — `Exporter.kt` (`info`/`sources`/ping lifecycle, reconnect backoff), `MediaJsonSerializer.kt` (start/media shapes, payload is the RTP payload), `jvb/src/main/resources/reference.conf` (`videobridge.exporter.*` defaults).
- `jitsi/jicoco` — `jicoco-mediajson/…/MediaJson.kt` — the wire format and its string-encoded fields.
- `jitsi/jitsi-meet` — `mod_muc_meeting_id.lua` (the meeting id is a generated UUID), transcription room-metadata writes.
