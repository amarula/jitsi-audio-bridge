# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
#
# The unit suite, inside a python container with the tree at /build.  Which
# python is the container's business, so nothing here has to be told.
#
# Run by .jenkins/container.sh.  The report goes to /out, which is copied back
# into the workspace for Jenkins to read — including when tests fail, which is
# when it matters most.  The Opus round-trip at the end of the suite loads the
# real library rather than a stub, so libopus has to be installed; and the
# [s3] extra is installed so those tests run rather than skip.
#
# PYTEST_PREFIX comes from the cell: four cells write suites whose test classes
# are all named the same, and without a prefix per interpreter the JUnit view
# merges them into one incoherent run.
set -e

apt-get update -qq
apt-get install -y -qq libopus0

if [ -n "${PYTEST_PREFIX:-}" ]; then
    prefix="--junit-prefix=$PYTEST_PREFIX"
else
    prefix=""
fi

pip install --quiet --root-user-action=ignore -e '.[dev,s3]'
# shellcheck disable=SC2086  # empty, or one argument
pytest --junitxml=/out/report.xml $prefix -q
