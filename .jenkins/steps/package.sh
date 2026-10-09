# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
#
# Build the Debian package, inside the container for the distribution it is
# meant to be installed on: the bundled virtualenv belongs to that
# distribution's python3, and the launcher refuses one built for another.
#
# Run by .jenkins/container.sh, with DEB_MAINTAINER (the package's Maintainer
# field) and SLUG (the distribution, e.g. debian13) in the environment.  The
# .deb is renamed here, so three of them can sit side by side in one build
# without colliding.
#
# lintian is part of the check rather than an afterthought: a package a Debian
# reviewer would reject is not a package that passed.
set -e

# Lintian treats a malformed Maintainer as an error, and `--fail-on error`
# then turns the package's fallback — "Unknown <root@localhost>" — into a red
# build with a message about the wrong thing entirely.
: "${DEB_MAINTAINER:?the package Maintainer would fall back to Unknown <root@localhost>}"

apt-get update -qq
apt-get install -y -qq python3 python3-venv libopus0 lintian

packaging/build-deb.sh /out
lintian --fail-on error /out/*.deb

# The package's own interpreter, run from where it would be installed — the
# only check that the payload survived packaging rather than just the build.
unpack="$(mktemp -d)"
dpkg-deb -x /out/*.deb "$unpack"
"$unpack/usr/lib/jitsi-audio-bridge/venv/bin/python" -m jitsi_audio_bridge --version

# Named for the distribution on the way out — inside the container, so that
# what leaves it is already the finished artifact, with no host-side rename
# over a glob that could quietly match nothing.
: "${SLUG:?the distribution slug, e.g. debian13}"
for deb in /out/*.deb; do
    mv "$deb" "${deb%.deb}.$SLUG.deb"
done
ls -l /out
