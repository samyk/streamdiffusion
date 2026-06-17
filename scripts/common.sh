#!/usr/bin/env bash

# Resolve a Python interpreter. Prefer the active venv, then project .venv, then python3.
if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python3"
elif [[ -n "${ROOT:-}" && -x "${ROOT}/.venv/bin/python3" ]]; then
  PYTHON="${ROOT}/.venv/bin/python3"
elif [[ -x ".venv/bin/python3" ]]; then
  PYTHON="$(pwd)/.venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "python3 not found. Activate the project venv or install Python 3." >&2
  exit 1
fi
