set shell := ["bash", "-euo", "pipefail", "-c"]

default: setup

# Install the dependencies on your machine (add your own with `uv add`).
setup:
    uv sync --locked

# Start the viewer at http://localhost:${APP_PORT:-8000}, and wait until it's up.
up:
    docker compose up -d --wait
    @echo "Viewer: http://localhost:${APP_PORT:-8000}"

# Stop the viewer.
down:
    docker compose down -v

# Write summaries/<version>.jsonl: v0-template, v1-naive, v2-structured or v3-grounded. Add --offline to use the cache only.
summarize version *args:
    uv run python -m summarizer.generate {{version}} {{args}}

# Rebuild every version from saved responses, with no API calls.
rebuild:
    for v in v0-template v1-naive v2-structured v3-grounded; do uv run python -m summarizer.generate $v --offline; done

# Score every version in summaries/ (no model calls) -> eval/results.md, eval/scores.csv
evaluate:
    uv run python -m summarizer.evaluate

# Test the evaluator itself by planting wrong numbers and deleting red flags.
check-evaluator:
    uv run python -m summarizer.check_evaluator

# LLM judge on a fixed sample of 20 companies -> eval/judge.md
judge *versions:
    uv run python -m summarizer.judge {{versions}}

# Check the viewer's code (app/) with Ruff and Pyrefly.
lint:
    uv run ruff check
    uv run ruff format --check
    uv run pyrefly check
