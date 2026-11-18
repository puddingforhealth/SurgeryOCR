#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
missing=0
check() { # name cmd fix
  if command -v "$2" >/dev/null 2>&1; then
    printf "  ok      %-12s %s\n" "$1" "$("$2" --version 2>/dev/null | head -n1)"
  else
    printf "  MISSING %-12s fix: %s\n" "$1" "$3"; missing=$((missing+1))
  fi
}
echo "doctor: SurgeryOCR"
check uv uv "curl -LsSf https://astral.sh/uv/install.sh | sh"
check git git "apt install git / brew install git"
if [ ! -d .venv ]; then echo "  MISSING .venv        fix: ./scripts/setup.sh"; missing=$((missing+1)); else echo "  ok      .venv"; fi
if [ "$missing" -gt 0 ]; then echo "$missing prerequisite(s) missing. Run ./scripts/setup.sh"; exit 1; fi
echo "All prerequisites present."
