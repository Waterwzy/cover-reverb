#!/usr/bin/env bash
# ============================================================
#  cover-reverb launcher (Linux / macOS)
#
#  Usage:
#    ./scripts/launch.sh                  start the web UI
#    ./scripts/launch.sh --port 8080      use another port
#    ./scripts/launch.sh --check          environment check only
#    ./scripts/launch.sh --selftest       run the pipeline on synthetic audio
#    ./scripts/launch.sh --batch --roots "/path/to/songs"
#    ./scripts/launch.sh --setup          reinstall dependencies
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${COVER_REVERB_ENV:-cover-reverb}"

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  local candidates=(
    "${HOME}/anaconda3/bin/conda"
    "${HOME}/miniconda3/bin/conda"
    "/opt/conda/bin/conda"
    "/usr/local/Caskroom/miniconda/base/bin/conda"
    "/usr/local/anaconda3/bin/conda"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -x "${candidate}" ]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

conda_env_python() {
  local conda_bin="$1"
  local env_path
  env_path="$("${conda_bin}" env list --json 2>/dev/null \
    | tr ',' '\n' | tr -d '"[] ' | grep "/${ENV_NAME}\$" | head -n1 || true)"
  if [ -n "${env_path}" ] && [ -x "${env_path}/bin/python" ]; then
    echo "${env_path}/bin/python"
    return 0
  fi
  return 1
}

PY=""

if [ -n "${COVER_REVERB_PYTHON:-}" ] && [ -x "${COVER_REVERB_PYTHON}" ]; then
  PY="${COVER_REVERB_PYTHON}"
elif CONDA_BIN="$(find_conda)"; then
  if PY="$(conda_env_python "${CONDA_BIN}")"; then
    :
  else
    PY=""
  fi
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PY="${REPO_ROOT}/.venv/bin/python"
fi

if [ -z "${PY}" ]; then
  echo
  echo "  No environment found - running bootstrap_env.py ..."
  echo
  python3 "${REPO_ROOT}/bootstrap_env.py" --with-dev

  if CONDA_BIN="$(find_conda)"; then
    PY="$(conda_env_python "${CONDA_BIN}" || true)"
  fi
  if [ -z "${PY}" ] && [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PY="${REPO_ROOT}/.venv/bin/python"
  fi
fi

if [ -z "${PY}" ]; then
  echo "  [FAIL] could not locate an interpreter after setup." >&2
  exit 1
fi

echo
echo "  python: ${PY}"
echo
exec "${PY}" -u "${REPO_ROOT}/launch.py" "$@"
