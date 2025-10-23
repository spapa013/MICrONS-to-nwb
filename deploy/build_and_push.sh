#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./build_and_push.sh <SERVICE> [args for `docker compose build`]
#
# Examples:
#   ./build_and_push.sh microns-to-nwb
#   ./build_and_push.sh microns-to-nwb --no-cache --pull
#   ./build_and_push.sh microns-to-nwb --build-arg FOO=bar

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <SERVICE> [build-args...]" >&2
  exit 2
fi

SERVICE="$1"; shift

# Build with any extra args you’d normally pass after `docker compose build`
docker compose build "$@" "$SERVICE"

# Push the image defined by `image:` for that service
docker compose push "$SERVICE"

echo "Done."
