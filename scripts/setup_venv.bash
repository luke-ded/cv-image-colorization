#!/bin/bash
# Creates a Python virtual environment in the repo root and installs the dependencies.

set -e

cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements.txt

echo "Setup complete. Activate the environment with: source .venv/bin/activate"
