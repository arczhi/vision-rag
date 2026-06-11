#!/usr/bin/env bash
# Compatibility wrapper for the earlier "naive" spelling.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/start_native.sh" "$@"
