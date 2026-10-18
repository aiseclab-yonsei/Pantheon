#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime.sh
pantheon_environment gpu
exec "$PANTHEON_PYTHON" -m pantheon.infer "$@"
