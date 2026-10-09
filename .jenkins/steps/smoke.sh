# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
#
# The end-to-end check, inside a python container with the tree at /build: the
# daemon is started for real, driven over a WebSocket by the sender simulator,
# and the WAV files, transcript, summary, mail and archived video are asserted
# on the far side.  ffmpeg is there for the master-track part of it.
#
# Run by .jenkins/container.sh.  The run leaves its whole work directory in
# /tmp inside this container, and that is the first thing anybody wants when a
# check fails — so it is copied out whichever way the run ends.  `if !` rather
# than a bare call because the container script runs under `set -e`.
set -e

apt-get update -qq
apt-get install -y -qq libopus0 ffmpeg

pip install --quiet --root-user-action=ignore '.[s3]'

status=0
python tests/smoke_test.py || status=$?

cp -a /tmp/jitsi-bridge-smoke-* /out/ 2>/dev/null || true
exit "$status"
