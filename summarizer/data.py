"""Loading the company records."""

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPANIES = ROOT / "data" / "companies.jsonl"
SUMMARIES = ROOT / "summaries"
RUNS = ROOT / "runs"

# The README says the records are "as of 31 December 2025". Every "recent" or "stale" judgement is measured from here.
AS_OF = date(2025, 12, 31)


def load_companies(path: Path = COMPANIES) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_id(companies: list[dict]) -> dict[str, dict]:
    return {c["registrationNumber"]: c for c in companies}
