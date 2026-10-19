#!/usr/bin/env bash
# Resolve the active environment's Python, never the original workstation path.
set -euo pipefail
python_bin="$(command -v python3 || command -v python)"
if [[ ${1:-} == --print && $# == 1 ]]; then
  printf '%s\n' "$python_bin"
  exit 0
fi
exec "$python_bin" "$@"
