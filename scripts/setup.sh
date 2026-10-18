#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${PYTHON:-}" ]]; then
  interpreter="$PYTHON"
else
  interpreter=""
  for candidate in python3.12 python3.13 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
      'import ensurepip, venv, sys; sys.exit(not ((3,11) <= sys.version_info[:2] <= (3,13)))' \
      >/dev/null 2>&1; then
      interpreter="$candidate"
      break
    fi
  done
fi
if [[ -z "$interpreter" ]]; then
  echo 'Python 3.11–3.13 with venv is required. Install it or set PYTHON=/path/to/python.' >&2
  exit 1
fi
exec "$interpreter" scripts/setup_env.py "$@"
