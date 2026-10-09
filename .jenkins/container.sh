#!/bin/sh
# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
#
# Run one step of the Jenkins pipeline inside one container.
#
#     container.sh IMAGE ARTIFACT_DIR SCRIPT
#
# The workspace is mounted **read-only** at /src and the tree is copied to
# /build inside the container, which then runs SCRIPT from there with the
# working directory set to it.  That is not tidiness: containers run as root,
# a Jenkins agent is long-lived, and anything a root process writes into the
# workspace belongs to root afterwards — the next build's `git clean -xffd`
# then fails on it and the agent stays broken until somebody logs in and
# removes it by hand.  `packaging/build-deb.sh` alone writes a `build/`
# directory and an `.egg-info` into the tree it runs from, and `pip install -e`
# writes more.
#
# Everything a step produces goes in the container's /out, which is copied back
# into ARTIFACT_DIR afterwards — through `docker cp`, which writes as the user
# running it, so the artifacts belong to Jenkins too.
#
# Environment:
#   CONTAINER_ENV  space-separated variables to carry into the container,
#                  defaulting to DEB_MAINTAINER, which the package build reads.
#   WORKSPACE      the agent's workspace, defaulting to the current directory,
#                  which is what a run from a checkout wants.

set -eu

if [ "$#" -lt 3 ]; then
    echo "usage: $0 IMAGE ARTIFACT_DIR SCRIPT" >&2
    echo "  SCRIPT runs inside the container with the tree at /build" >&2
    exit 2
fi

image="$1"
artifacts="$2"
script="$3"
workspace="${WORKSPACE:-$PWD}"

if [ ! -d "$workspace" ]; then
    echo "no such workspace: $workspace" >&2
    exit 2
fi

mkdir -p "$artifacts"

# Explicitly, so that an unreachable registry or a rate limit says so here
# rather than surfacing as a confusing failure to create the container.  The
# pipeline pulls them all in one stage first, with retries, so that seven
# cells do not meet the rate limit together.
echo "==> pulling $image"
docker pull "$image" >/dev/null

# shellcheck disable=SC2086  # word splitting is the point: a list of names
env_args=""
for name in ${CONTAINER_ENV:-DEB_MAINTAINER}; do
    env_args="$env_args -e $name"
done

echo "==> $image: $script"

container="$(
    # shellcheck disable=SC2086  # as above: the -e entries are separate words
    docker create \
        -v "$workspace:/src:ro" \
        -w / \
        $env_args \
        "$image" bash -lc "
set -e
mkdir -p /build /out
tar -C /src \
    --exclude=./.git --exclude=./dist --exclude=./build --exclude=./out \
    --exclude=.pytest_cache --exclude=.ruff_cache --exclude='*.egg-info' \
    -cf - . | tar -C /build -xf -
cd /build
$script"
)"

# On every path out, a failed step and a Ctrl-C included: a container left
# behind holds its writable layer, and the next build would make another.
# shellcheck disable=SC2064  # expanded now, on purpose, while $container is set
trap "docker rm -f '$container' >/dev/null 2>&1 || true" EXIT INT TERM

# -a so the step's output is the build's output.  Checked rather than assumed:
# `docker start -a` returns the container's own exit status, so a failing step
# is a failing build.
status=0
docker start -a "$container" || status=$?

# Copied whichever way the step ended, because a failing step is usually the
# one whose output somebody needs — a failing pytest run writes its JUnit
# report all the same, and Jenkins is where that report is read.
docker cp "$container:/out/." "$artifacts/" || true
echo "==> collected $(find "$artifacts" -type f | wc -l) file(s) into $artifacts"

exit "$status"
