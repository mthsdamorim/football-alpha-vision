#!/usr/bin/env bash
set -euo pipefail

uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
uv run python -m football_alpha_video --version
uv run python -m football_alpha_player_detection --help
uv run python -m football_alpha_person_tracking --help
uv run python -c "import football_alpha_pitch_projection"
