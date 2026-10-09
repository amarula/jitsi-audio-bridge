#!/bin/sh
# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
#
# The tag has to be the version the source says: the package version, the
# changelog and the launcher's build stamp all come from that one string, and
# a tag that disagrees with it is a broken release rather than a typo to paper
# over.  The same check runs in the GitHub workflow — this one is the earlier
# warning, since Jenkins usually hears about a push first.
#
# It is a warning, not a gate: the GitHub release fires on the same push of the
# same tag, so Jenkins cannot stop it.  It can only say so, here, before anybody
# reads a release page and wonders.
#
# Run on the agent, not in a container: it reads one line of the checkout.
set -eu

version="$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' src/jitsi_audio_bridge/__init__.py)"
if [ -z "$version" ]; then
    echo "cannot read __version__ from src/jitsi_audio_bridge/__init__.py" >&2
    exit 1
fi

if [ "v$version" != "${TAG_NAME:-}" ]; then
    echo "tag ${TAG_NAME:-<none>} does not match __version__ $version (expected v$version)" >&2
    exit 1
fi

echo "tag $TAG_NAME matches jitsi-audio-bridge $version"
