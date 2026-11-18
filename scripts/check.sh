#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# The one gate before every commit: format, lint, tests.
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -X utf8 -m pytest
echo "check: all green"
