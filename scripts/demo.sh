#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime.sh
pantheon_environment cpu
exec "$PANTHEON_PYTHON" -m pantheon.pipeline --datasets Celeb-DF-v1 --output outputs/cpu_demo "$@"
