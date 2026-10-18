#!/usr/bin/env bash
# Sourced after the caller has changed to the project root.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

pantheon_environment() {
  local kind="$1"
  local env_root="${PANTHEON_ENV_ROOT:-$PWD/environment}"
  if [[ ! -x "$env_root/$kind/bin/python" || ! -f "$env_root/$kind/.pantheon-setup.json" ]]; then
    if [[ "$kind" == cpu ]]; then
      bash setup.sh cpu --env-root "$env_root"
    else
      bash setup.sh auto --env-root "$env_root"
      if [[ ! -f "$env_root/gpu/.pantheon-setup.json" ]]; then
        bash setup.sh inference-cpu --env-root "$env_root"
      fi
    fi
  fi
  PANTHEON_PYTHON="$env_root/$kind/bin/python"
}
