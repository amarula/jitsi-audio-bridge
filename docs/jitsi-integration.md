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
      │  someone joins ─►│                        │                      │
      │                  │  asyncTranscription=true (server-side)        │
      │◄── the client now knows a backend exists ─│                      │
      │                  │                        │                      │
      │  user enables    │                        │                      │
      │  transcription ─►│  recording.isTranscribingEnabled=true         │
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

Jitsi's **Record** button is not part of this: recording is
[Jibri](https://jitsi.github.io/handbook/docs/devops-guide/devops-guide-quickstart)'s
job, and the bridge never sees it. A deployment can therefore transcribe
perfectly while the UI answers every recording request with "all recorders are
currently busy" — see the README's troubleshooting entry, and `verify_jitsi`,
which checks Jicofo's recorder pool as well.

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

### 2. Prosody — advertise the transcriber

Transcription is gated by two per-room metadata keys stored by
`mod_room_metadata_component` under `room.jitsiMetadata`. The first,
`asyncTranscription`, is server-controlled — the component's
`blocked_metadata_keys` list rejects client writes to it — and answers the
client's question "is there a backend transcriber here?":

- **false**: turning transcription on dials the `jitsi_meet_transcribe`
  extension, which needs Jigasi in the room; with none, the user gets
  "Transcribing failed".
- **true**: the client, already knowing a backend exists, writes the second key
  (`recording.isTranscribingEnabled`) itself and Jicofo starts the bridge.

So the module below does exactly that one thing: it announces the backend in
every room. Transcription then starts when a user turns it on — which is the
default the checker expects.

Create `mod_force_async_transcription.lua` on the Prosody plugin path (for
example `/usr/share/jitsi-meet/prosody-plugins/`):

```lua
-- mod_force_async_transcription.lua
-- Makes every room advertise that a backend transcriber exists, so turning
-- transcription on in the UI starts the audio bridge rather than dialling the
-- legacy Jigasi number.
-- Enable on the main MUC component (e.g. conference.<domain>).
--
-- Only asyncTranscription is set here. Jicofo also waits for
-- recording.isTranscribingEnabled, which the client of whoever asks for
-- transcription writes; rooms are transcribed on request, not on creation.
-- Set that key here too to transcribe every room from the first join,
-- whether or not anyone asks for it.
--
-- The metadata component broadcasts only when 'room-metadata-changed' fires,
-- so writing room.jitsiMetadata alone never reaches Jicofo or the clients.

local jid = require 'util.jid';

local util = module:require 'util';
local is_healthcheck_room = util.is_healthcheck_room;

local function announce_transcription(room)
    -- mod_room_metadata_component initializes this table at priority -1,
    -- so run after it.
    if not room.jitsiMetadata then
        room.jitsiMetadata = {};
    end

    room.jitsiMetadata.asyncTranscription = true;
end

module:hook('muc-room-created', function(event)
    local room = event.room;

    if is_healthcheck_room(room.jid) then
        return;
    end

    announce_transcription(room);

    module:log('info', 'Announced transcription for room %s', room.jid);
end, -2);

-- The metadata component publishes only on this event, and at room creation
-- there is nobody to publish to, so re-publish as occupants arrive: Jicofo
-- first, then the clients. A client needs the flag before its user can turn
-- transcription on.
module:hook('muc-occupant-joined', function(event)
    local room = event.room;

    if is_healthcheck_room(room.jid) then
        return;
    end

    announce_transcription(room);

    module:context(jid.host(room.jid)):fire_event('room-metadata-changed', { room = room; });
end, -2);
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

#### Clients also need the component advertised

Forcing the flag is not enough for the browser. lib-jitsi-meet accepts messages
from a component only if it discovered that component's address in the main
host's `disco#info` when it connected; a client that is never told about
`metadata.<domain>` drops every metadata message it receives as coming from an
unknown sender. The visible symptom is

```js
APP.store.getState()['features/base/conference'].conference
    .getMetadataHandler().getMetadata()   // {} — forever
```

and, in the UI, transcription falling back to dialling Jigasi and failing
("Transcribing failed"). Jicofo is unaffected: it is a MUC occupant, and the
component sends it the metadata directly.

`mod_room_metadata_component` announces itself with `jitsi-add-identity`, which
`mod_features_identity` turns into that disco entry. Enable it on the **main**
VirtualHost — the one with `bosh`/`websocket`, not the MUC component:

```lua
VirtualHost "example.com"
    modules_enabled = {
        -- ... existing modules ...
        "features_identity";
    }
```

Stock configurations generated since June 2025 already list it. An older, hand
edited or never-regenerated `prosody.cfg.lua` may not: a server upgrade adds the
module file without touching your site config (dpkg keeps the local version), so
the two drift apart silently. The checker reports this as
`prosody.features_identity`.

> **Note.** Rooms are transcribed **on request**: Jicofo waits for both keys,
> and `recording.isTranscribingEnabled` is written by the client of whoever
> turns transcription on. For the user doing that to be a moderator, since the
> component rejects metadata writes from anyone else — in practice the person
> who opened the room first. To transcribe every room unconditionally, extend
> the module to set the `recording` metadata too, or pre-seed it from
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

### 5. Optional — give sessions their names back

Everything above produces transcripts whose speakers are `participant-<tag>`,
whose summary is titled "General Meeting", and whose mail goes to
`[smtp] fallback_recipient` — because the JVB's framing carries no names,
addresses or room name. Prosody knows all three, and it already knows which
room a session belongs to: the session directory is named after the meeting ID
that `muc_meeting_id` generates and Jicofo substitutes into `{{MEETING_ID}}`.

So one more module writes what it knows under that ID, and the bridge adopts
it when a session has no metadata of its own. Create
`mod_audio_bridge_metadata.lua` beside the other module:

```lua
-- mod_audio_bridge_metadata.lua
-- Gives the audio bridge what stock Jitsi's media export does not carry: the
-- meeting's name, and the participants with the display names the JVB cannot
-- send.  The session directory the bridge creates is named after the meeting
-- id, which is exactly what this module has in room._data.meetingId.
-- Enable on the main MUC component (e.g. conference.<domain>), and point the
-- bridge at the same directory with [storage] session_metadata_dir.

local jid = require 'util.jid';
local json = require 'cjson.safe';
local lfs = require 'lfs';

local util = module:require 'util';
local is_admin = util.is_admin;
local is_jibri = util.is_jibri;
local is_transcriber = util.is_transcriber;
local is_healthcheck_room = util.is_healthcheck_room;

-- Must be the bridge's [storage] session_metadata_dir.
local output_dir = module:get_option_string(
    'audio_bridge_metadata_dir', '/srv/recordings/.session-metadata');

-- Display names travel in the occupant's presence, under XEP-0172.
local NICK_NS = 'http://jabber.org/protocol/nick';

-- Everyone the room has seen, by room and then by participant id.  The bridge
-- reads this file long after the meeting -- it waits for the session to go
-- quiet first -- so writing only who is *present* would hand it an empty
-- room.  Emails arrive in the session's token context rather than in the
-- presence, and only where the deployment authenticates users.
local known = {};

-- Jicofo, the JVB, Jibri and the transcriber are in the room but are not in
-- the audio: the bridge would have nobody to attribute them to.
local function is_participant(occupant)
    return not is_admin(occupant.bare_jid)
        and not is_jibri(occupant)
        and not is_transcriber(occupant.jid);
end

local function display_name(occupant)
    local presence = occupant:get_presence();
    local name = presence and presence:get_child_text('nick', NICK_NS);
    if name and #name > 0 then
        return name;
    end
    return nil;
end

local function remember(room, occupant, session)
    local id = jid.resource(occupant.nick);
    if not id then
        return;
    end

    local user = session and session.jitsi_meet_context_user;
    local store = known[room.jid];
    if not store then
        store = {};
        known[room.jid] = store;
    end

    local entry = store[id] or {};
    entry.name = display_name(occupant) or entry.name;
    entry.email = (user and user.email) or entry.email;
    store[id] = entry;
end

local function participants_of(room)
    local store = known[room.jid] or {};
    local ids = {};
    for id in pairs(store) do
        table.insert(ids, id);
    end
    table.sort(ids);

    local participants = {};
    for _, id in ipairs(ids) do
        local entry = store[id];
        table.insert(participants, {
            id = id;
            name = entry.name;
            email = entry.email;
        });
    end
    return participants;
end

local function write_metadata(room)
    local meeting_id = room._data and room._data.meetingId;
    if not meeting_id then
        -- Without muc_meeting_id Jicofo invents an id of its own, and this
        -- file could not be matched to the session it describes.
        module:log('warn', 'no meeting id for %s; is muc_meeting_id enabled?', room.jid);
        return;
    end

    local participants = participants_of(room);
    local encoded = json.encode({
        room_name = jid.node(room.jid);
        meeting_id = meeting_id;
        source = 'audio_bridge_metadata';
        participants = participants;
    });
    if not encoded then
        module:log('error', 'cannot encode the metadata of %s', room.jid);
        return;
    end

    if not lfs.attributes(output_dir, 'mode') then
        lfs.mkdir(output_dir);
    end

    -- Written whole and renamed into place: the bridge reads this file while
    -- the meeting is still running.
    local path = output_dir .. '/' .. meeting_id .. '.json';
    local temporary = path .. '.tmp';
    local handle, err = io.open(temporary, 'w');
    if not handle then
        module:log('error', 'cannot write %s: %s', temporary, err or 'unknown error');
        return;
    end
    handle:write(encoded);
    handle:close();
    local moved, move_err = os.rename(temporary, path);
    if not moved then
        module:log('error', 'cannot move %s into place: %s', temporary,
            move_err or 'unknown error');
        return;
    end

    module:log('info', 'Wrote metadata for %s: %d participant(s), meeting id %s',
        room.jid, #participants, meeting_id);
end

-- A room reused for a later meeting starts with nobody in its record.
module:hook('muc-room-created', function(event)
    known[event.room.jid] = nil;
end, -2);

module:hook('muc-occupant-joined', function(event)
    local room = event.room;

    if is_healthcheck_room(room.jid) then
        return;
    end

    if is_participant(event.occupant) then
        remember(room, event.occupant, event.origin);
    end

    write_metadata(room);
end, -2);

-- Rewritten when someone leaves too, so the file is complete before the
-- meeting ends -- but nobody is dropped from it: a participant who left early
-- is still someone who spoke.
module:hook('muc-occupant-left', function(event)
    local room = event.room;

    if is_healthcheck_room(room.jid) then
        return;
    end

    write_metadata(room);
end, -2);

-- The room's own record goes when the room does; the file stays, because the
-- bridge consumes it when it processes the meeting.
module:hook('muc-room-destroyed', function(event)
    known[event.room.jid] = nil;
end, -2);
```

Enable it on the MUC component, exactly like the other one:

```lua
Component "conference.example.com" "muc"
    modules_enabled = {
        -- ... existing modules ...
        "audio_bridge_metadata";
    }
```

Then the two sides have to share one directory. Prosody runs as `prosody` and
the bridge as `jitsi-bridge`, so the directory belongs to the bridge's group
and the setgid bit keeps it there:

```sh
sudo install -d -o jitsi-bridge -g jitsi-bridge -m 2770 /srv/recordings/.session-metadata
sudo usermod -aG jitsi-bridge prosody
```

and in the bridge's `config.ini`:

```ini
[storage]
session_metadata_dir = /srv/recordings/.session-metadata
```

```sh
sudo systemctl restart prosody jitsi-audio-bridge
```

The module keeps a record of everyone the room has *seen*, not of who is
still in it: the bridge adopts this file when it processes the meeting, which
is by then some time after the last person left, and a file that only listed
who was present would be empty exactly when it is read. It is also rewritten
as people join and leave, so a meeting cut short still names the people who
spoke before it was.

Now a session that has no `metadata.json` of its own adopts the dropped file:
`transcript.txt` is attributed by display name, the summary carries the room's
name, and — where the deployment authenticates users, so that the token has an
email — the transcript goes to the participants instead of
`fallback_recipient`. Each accepted file is removed as it is consumed, so the
directory only holds the meetings still running. Sessions that did send a
control frame are untouched: their own `metadata.json` always wins.

### 6. Optional — archive the recordings

The video is the one product of a meeting the bridge cannot make itself: Jibri
makes it, when a user presses Record, and the bridge only copies it out — to an
S3-compatible bucket, if `[s3]` says where, either after the mail or before it,
when the mail is to link to the recording. That leaves the bridge to work out
*which* recording belongs to the meeting it has just transcribed, and Jitsi
tells it only two things:

* **the room**, which Jibri puts in the filename it builds
  (`<callName>_<yyyy-MM-dd-HH-mm-ss>.<ext>`) and again in the `metadata.json`
  it writes beside the recording, as the call URL it joined with. With the
  module in §5 in place, this is the same room name the bridge has;
* **the time the recording stopped**, which is the timestamp on the filename
  and the file's own mtime. The recording ends when somebody stops it, which
  is after the meeting — so the bridge keeps looking for `[s3] wait_seconds`
  after the transcript is written.

Two rules follow for the deployment:

```ini
[s3]
; Jibri's recording.recordings-directory, and *not* [storage] recordings_dir:
; Jibri creates a session directory per recording, and a tree it cannot write
; to is what makes it report itself unhealthy — the "all recorders are
; currently busy" failure.
jibri_dir = /srv/jibri-recordings
```

and the bridge's user has to be able to read that tree — it runs as
`jitsi-bridge` while Jibri writes as `jibri`:

```sh
sudo chmod 0755 /srv/jibri-recordings
# or, when the tree stays group-only (0750):
sudo usermod -aG jibri jitsi-bridge
```

`jitsi-audio-bridge-verify` checks both, and says so by name when the bridge is
pointed at a directory this host's Jibri does not write to — which is otherwise
a silent failure, with one line in the log at the end of every meeting. If
`delete_after_upload` is on, the bridge also writes there: add the directory to
`ReadWritePaths` in the unit.

With `link_in_mail` the deployment needs one more thing of the network, and it
is the opposite direction: the address in `[s3] link_endpoint` has to be
reachable by the people who read the mail, and the proxy in front of the bucket
has to pass the request through with its query string and `Host` header intact,
because the link is signed over both. A resource that requires a login turns
every link into a login page, and one that rewrites the path or the host turns
it into `SignatureDoesNotMatch`.

The endpoint itself is one more outbound connection to whitelist, like the
Whisper and Ollama ones — same host, same tunnel — and it is the failure that
shows up last, because an upload that cannot connect costs a meeting nothing
but its video. All of it goes to the one hostname the endpoint names, to
`/<bucket>/<prefix>/<room>/<file>` on whatever port that URL carries; the
README's [Network access](../README.md#network-access) has the request-by-request
table to write a proxy rule from, including the hostname change that
`path_style = false` brings.

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
  event. Each packet also feeds the timeline: its arrival on the session's
  clock, its position in the recording, and whether it sounded like speech —
  which is what the interleaved transcript is built from.
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
- **Ordering comes from the bridge, not from Jitsi.** The daemon captures each
  packet as it arrives and writes the speaking turns to `timeline.json`, which
  is what makes the transcript interleaved rather than one block per
  participant. The protocol's own `media.timestamp` is *not* used for that:
  the timestamps of two participants' streams have no common origin, so
  arrival on one real-time socket is the only shared clock there is. The
  exporter's `vad` flag is used when it is set; otherwise the decoded audio's
  own level decides.

Known hazards, already recorded in [REVIEW.md](../REVIEW.md): a reconnecting
JVB reuses its `sessionId` and the daemon has no idempotency (a reconnect
re-transcribes, re-emails, and two live connections interleave writes into the
same directory); and if RED is enabled for the endpoint, the payload is not
plain Opus and libopus will reject it (the per-participant dropped-packet count
at session end is the signal).

## Verifying

### Run the checker first

`tools/verify_jitsi.py` automates most of this and prints a fix per failure.
Run it on the Jitsi host; it is read-only by default. From a checkout it is
`python3 -m tools.verify_jitsi` (from the repository root) or
`python3 /path/to/tools/verify_jitsi.py` (from anywhere). If the bridge was
installed from the Debian package (`make deb`), the same tool is on the PATH as
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

### Proposing the fixes

For the failures with a mechanical remedy the same tool can write the change
the way Debian handles conffiles: never touch the original, write `<file>.new`
beside it, and print how to review and install it.

```sh
sudo jitsi-audio-bridge-verify --fix --bridge-url ws://bridge.example.com:8080
```

It proposes, each only when its check failed:

- **Jicofo** — the `transcription` block, appended to `custom-jicofo.conf.new`
  (or, when `jicofo.conf` does not include it yet, that file plus a one-line
  `include` in `jicofo.conf.new`). The bridge address cannot be inferred, so
  `--bridge-url` is required for this one; a host, a `host:port`, or a
  `ws://`/`wss://` URL is expanded to `…/transcribe?sessionId={{MEETING_ID}}`
  (a bare host takes port 8080). The address must be one the **JVB** can reach,
  so it is usually the bridge host's name, not `127.0.0.1` — nothing in the
  Jitsi configuration records where the bridge is, so it cannot be inferred.
- **Prosody** — `mod_force_async_transcription.lua.new` (or the name of a
  module already present that does the job), the site config with that name
  and `muc_meeting_id` added to the MUC's `modules_enabled`, and — when the
  room-metadata component is missing — the stock
  `Component "metadata.<domain>" "room_metadata_component"` block with its
  `muc_component` line (the `room_metadata` module it used to pair with was
  removed upstream in June 2026). If `mod_room_metadata_component.lua` is not
  installed, that part is refused with the upgrade command instead of
  proposed, because a component whose module is missing stops Prosody from
  starting. The same treatment applies to `mod_features_identity.lua`, which
  puts the component into the client's disco#info — that name goes on the main
  VirtualHost's `modules_enabled`, not the MUC's.
- **jitsi-meet** — the client config with `transcription: { enabled: true }`
  inserted; a commented-out sample block is left as it is and a live one added.

Each entry prints the commands that follow:

```sh
diff -u /etc/jitsi/jicofo/custom-jicofo.conf /etc/jitsi/jicofo/custom-jicofo.conf.new
sudo mv /etc/jitsi/jicofo/custom-jicofo.conf.new /etc/jitsi/jicofo/custom-jicofo.conf
sudo systemctl restart jicofo
```

Nothing is applied, and the exit status is unchanged. Without write access to
`/etc`, pass `--output-dir /tmp/fix` to stage the proposals elsewhere; they are
written beside the file being *edited* (the `conf.avail` file, not the `conf.d`
symlink that points at it), keep the original's mode, and are never overwritten
unless `--force-fix` is given.

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
