#!/usr/bin/env bash
# Optional desktop entry. Without this wrapper the original external default stays.
set -euo pipefail
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$task_dir/run_nandu1.sh" --command-source mouse "$@"
