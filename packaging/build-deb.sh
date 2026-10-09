#!/usr/bin/env bash
#
# Build a Debian package for the machine this runs on.
#
# The payload is a private virtualenv, so the package depends only on the
# system python3 and libopus0 — never on the distribution's python3-websockets,
# which is older than this daemon's websockets>=13 floor on Debian 12 (10.4)
# and Ubuntu 24.04 (12.0). A virtualenv belongs to one interpreter version and
# architecture, so build the package on the machine (or in a container
# matching it) where it will be installed; the postinst warns when the target's
# python3 differs from the one it was built for.
#
# Usage: packaging/build-deb.sh [OUTPUT_DIR]     (default: dist/)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/dist}"
PKG=jitsi-audio-bridge
CONF_DIR=/etc/jitsi-audio-bridge

command -v dpkg-deb >/dev/null || {
    echo "dpkg-deb is required (apt install dpkg)" >&2
    exit 1
}
python3 -c 'import ensurepip' 2>/dev/null || {
    echo "python3-venv is required to build the bundled virtualenv" >&2
    echo "(apt install python3-venv)" >&2
    exit 1
}

VERSION="$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "$ROOT/src/jitsi_audio_bridge/__init__.py")"
if [ -z "$VERSION" ]; then
    echo "cannot read __version__ from src/jitsi_audio_bridge/__init__.py" >&2
    exit 1
fi
DEB_VERSION="$VERSION-1"
ARCH="$(dpkg --print-architecture)"
PYTHON_MINOR="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
MAINTAINER="${DEB_MAINTAINER:-$(git -C "$ROOT" config user.name 2>/dev/null || echo Unknown) <$(git -C "$ROOT" config user.email 2>/dev/null || echo root@localhost)>}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
PKGROOT="$STAGE/$PKG"
VENV="$PKGROOT/usr/lib/$PKG/venv"

install -d "$PKGROOT/DEBIAN" "$PKGROOT/usr/bin" "$PKGROOT/usr/lib/$PKG" \
    "$PKGROOT/usr/share/doc/$PKG" "$PKGROOT/lib/systemd/system" "$PKGROOT$CONF_DIR"

echo "==> creating the bundled virtualenv (python3 $PYTHON_MINOR, $ARCH)"
python3 -m venv "$VENV"
# The [s3] extra brings boto3 in: a package installed on a host that never
# configures an endpoint would otherwise fail at the first upload, which is
# the one moment nobody wants to discover a missing dependency.
"$VENV/bin/pip" install --quiet --no-cache-dir "$ROOT[s3]"
# Stamped so the launcher can explain itself when the package is installed on
# a host whose python3 differs from the one it was built for.
printf '%s\n' "$PYTHON_MINOR" > "$VENV/BUILT-FOR"

# The venv is a runtime, not a development environment: drop the caches and
# the installer, then prove the payload still imports.
find "$PKGROOT" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$VENV"/lib/python*/site-packages/pip \
       "$VENV"/lib/python*/site-packages/pip-* \
       "$VENV"/lib/python*/site-packages/setuptools \
       "$VENV"/lib/python*/site-packages/setuptools-* \
       "$VENV"/lib/python*/site-packages/wheel \
       "$VENV"/lib/python*/site-packages/wheel-* \
       "$VENV"/lib/python*/site-packages/_distutils_hack \
       "$VENV"/lib/python*/site-packages/distutils-precedence.pth \
       "$VENV"/bin/pip "$VENV"/bin/pip3 "$VENV"/bin/pip3.* \
       "$VENV"/.gitignore
"$VENV/bin/python" -c 'import jitsi_audio_bridge, websockets, requests, boto3'

# Every distribution in the bundled virtualenv is distributed with this
# package, so every one of them has to appear in its copyright file.  Nothing
# else would notice a dependency arriving unaccounted for.  Names are compared
# with their punctuation flattened: pip says charset-normalizer where a
# directory says charset_normalizer, and both are the same package.
accounted="$(tr -- '_-' '--' < "$ROOT/packaging/deb/copyright.in" | tr 'A-Z' 'a-z')"
unlisted=""
for meta in "$VENV"/lib/python*/site-packages/*.dist-info/METADATA; do
    name="$(sed -n 's/^Name: //p' "$meta" | head -1)"
    [ "$name" = "$PKG" ] && continue
    flat="$(printf '%s' "$name" | tr -- '_-' '--' | tr 'A-Z' 'a-z')"
    case "$accounted" in
        *"$flat"*) ;;
        *) unlisted="$unlisted $name" ;;
    esac
done
if [ -n "$unlisted" ]; then
    echo "packaging/deb/copyright.in does not account for:$unlisted" >&2
    echo "add them, with their licence, before shipping the package" >&2
    exit 1
fi

# The deployment tools travel with the package: the verifier is meant to run
# on the Jitsi host, which may have nothing but this .deb, and the sender lets
# a deployed bridge be exercised without Jitsi.
echo "==> installing the tools into the venv"
SITE_PACKAGES="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cp -r "$ROOT/tools" "$SITE_PACKAGES/tools"
find "$SITE_PACKAGES/tools" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
"$VENV/bin/python" -c 'import tools.verify_jitsi, tools.send_meeting'

# The build's umask must not leak into the package: normalise the payload
# (X keeps directories and already-executable files executable).
chmod -R u=rwX,go=rX "$PKGROOT/usr" "$PKGROOT/lib"

echo "==> installing the launchers, unit, config and documentation"
install -m 0755 "$ROOT/packaging/deb/$PKG.launcher" "$PKGROOT/usr/bin/$PKG"
install -m 0755 "$ROOT/packaging/deb/$PKG.launcher" "$PKGROOT/usr/bin/$PKG-verify"
install -m 0755 "$ROOT/packaging/deb/$PKG.launcher" "$PKGROOT/usr/bin/$PKG-send"
install -m 0644 "$ROOT/config.ini.example" "$PKGROOT$CONF_DIR/config.ini"
install -m 0644 "$ROOT/README.md" "$PKGROOT/usr/share/doc/$PKG/README.md"
install -m 0644 "$ROOT/docs/jitsi-integration.md" "$PKGROOT/usr/share/doc/$PKG/jitsi-integration.md"
# Debian Policy requires the copyright file, and it is the only place that
# accounts for the dependencies travelling in the bundled virtualenv.  The
# licence itself is appended rather than substituted: it is full of "&" and
# "\" as far as sed is concerned.  Indenting it a line at a time, blanks as
# " .", is what makes it one DEP-5 field.
install -m 0644 "$ROOT/LICENSE" "$PKGROOT/usr/share/doc/$PKG/LICENSE"
COPYRIGHT="$PKGROOT/usr/share/doc/$PKG/copyright"
{
    sed -e "s/@PKG@/$PKG/g" -e "/@AGPL@/d" "$ROOT/packaging/deb/copyright.in"
    awk '{ print ($0 == "") ? " ." : " " $0 }' "$ROOT/LICENSE"
} > "$COPYRIGHT"
chmod 0644 "$COPYRIGHT"

# The shipped unit targets the README's /opt install; rewrite the three paths
# it uses and fail loudly if the unit no longer has them.
sed -e "s|^Documentation=.*|Documentation=file:/usr/share/doc/$PKG/README.md|" \
    -e "s|^WorkingDirectory=.*|WorkingDirectory=/usr/lib/$PKG|" \
    -e "s|^ExecStart=.*|ExecStart=/usr/bin/$PKG --config $CONF_DIR/config.ini|" \
    "$ROOT/systemd/$PKG.service" > "$PKGROOT/lib/systemd/system/$PKG.service"
grep -q "^ExecStart=/usr/bin/$PKG " "$PKGROOT/lib/systemd/system/$PKG.service" || {
    echo "cannot rewrite ExecStart in systemd/$PKG.service" >&2
    exit 1
}

install -m 0644 /dev/null "$PKGROOT/DEBIAN/conffiles"
echo "$CONF_DIR/config.ini" > "$PKGROOT/DEBIAN/conffiles"

INSTALLED_SIZE="$(du -sk --exclude=DEBIAN "$PKGROOT" | cut -f1)"
render() {
    sed -e "s/@VERSION@/$DEB_VERSION/g" \
        -e "s/@ARCH@/$ARCH/g" \
        -e "s/@PYTHON_MINOR@/$PYTHON_MINOR/g" \
        -e "s/@INSTALLED_SIZE@/$INSTALLED_SIZE/g" \
        -e "s|@MAINTAINER@|$MAINTAINER|g" "$1"
}

render "$ROOT/packaging/deb/control.in" > "$PKGROOT/DEBIAN/control"
render "$ROOT/packaging/deb/postinst.in" > "$PKGROOT/DEBIAN/postinst"
install -m 0755 "$ROOT/packaging/deb/prerm" "$PKGROOT/DEBIAN/prerm"
install -m 0755 "$ROOT/packaging/deb/postrm" "$PKGROOT/DEBIAN/postrm"
chmod 0644 "$PKGROOT/DEBIAN/control" "$PKGROOT/DEBIAN/conffiles"
chmod 0755 "$PKGROOT/DEBIAN/postinst"

install -d "$OUT_DIR"
DEB="$OUT_DIR/${PKG}_${DEB_VERSION}_${ARCH}.deb"
dpkg-deb --root-owner-group --build "$PKGROOT" "$DEB" >/dev/null

echo
echo "==> built $DEB"
dpkg-deb --info "$DEB" | sed -n '1,14p'
