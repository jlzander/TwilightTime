#!/usr/bin/env bash
set -euo pipefail

# Adjust if your PROJECT_DIR is different
PROJECT_DIR="$HOME/absreality_pipeline"
VENV_DIR="$PROJECT_DIR/absreality_env"

if [ ! -d "$VENV_DIR" ]; then
  echo "Virtualenv not found at: $VENV_DIR" >&2
  exit 1
fi

echo "Activating virtualenv at $VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# Optional: cd into tt root for convenience
cd "$HOME/tt"

echo "Environment active. Python: $(python --version)"
