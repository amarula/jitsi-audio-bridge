# SPDX-FileCopyrightText: 2026 Amarula Solutions
# SPDX-License-Identifier: AGPL-3.0-only
#
# Lint, inside a python container with the tree at /build.  Run by
# .jenkins/container.sh, which is what mounts the tree and collects /out.
set -e

pip install --quiet --root-user-action=ignore ruff
ruff check src tests tools
