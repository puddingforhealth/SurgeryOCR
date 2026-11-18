# SurgeryOCR

This project aims to automate the extraction and labeling of critical numerical data displayed within ophthalmological s

## Intent

(One paragraph: what this project is for, who uses it, why it exists. Fill in after scaffolding — Claude should NOT leave this as a placeholder in the first real commit.)

## Conventions

- Package manager: **uv** only. Never `pip install` at user level.
- Run: `./scripts/dev.sh` or `./scripts/dev.ps1`.
- Setup: `./scripts/setup.sh` or `./scripts/setup.ps1` (idempotent).
- Tests: `pytest tests/`. Structure mirrors `src/surgeryocr/`.
- Format + lint: `ruff format && ruff check --fix`.
- Full check before every commit: `./scripts/check.sh` or `./scripts/check.ps1` (lint + typecheck + tests). Never commit on red.
- Wiring in an AI agent, provider or model routing? Read the matching `agentic-*` skills first (the global routing block lists them).

## Taboos

- No comments explaining WHAT (names do that). Comments only for WHY — hidden constraints.
- No speculative abstractions. Three similar lines is fine.
- No `try/except` around code that cannot fail.
- No backwards-compat shims unless the user explicitly asks.
