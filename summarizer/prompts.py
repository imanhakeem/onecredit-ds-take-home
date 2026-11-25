"""The prompt for each LLM version. Each version changes one thing from the one before, so the scores say what helped.

- v1-naive:      the raw record and a one-line instruction. What you'd write first.
- v2-structured: the raw record, plus what the team needs, a fixed format, risk definitions and rules against guessing.
- v3-grounded:   the same instructions, but the model gets the facts and flags computed by facts.py *instead of* the
                 raw record, and is told the risk tier. It writes; it no longer reads ratios or does arithmetic.
"""

import json

from .facts import Facts, facts_brief

AUDIENCE = (
    "You write summaries for a team that vets Malaysian companies before choosing them as suppliers or project "
    "partners. A reviewer reads your summary first, in under a minute, to decide whether the company deserves a closer "
    "look. Missing a red flag is the worst mistake you can make; a wrong number is the second worst. "
    "The record is as of 31 December 2025."
)

FORMAT = """Write the summary in Markdown, at most 170 words, in exactly this shape:

**Risk: <High|Elevated|Low>**: <one sentence on the main reason>

<One sentence: what the company does, where, how old it is, and how big (latest revenue).>

**Red flags**
- <most serious first, at most 5 bullets, each specific: give the amount, year or date>
(If there are none, write "- None found in the record.")

**Financials (FY<latest year>):** <one or two sentences: revenue and its trend, profit, cash flow, debt.>

**Ownership & litigation:** <one or two sentences: who controls it, corporate shareholders, recent director changes, court cases.>

Rules:
- Use only numbers that appear in the input. Do not calculate new ones. Write amounts like RM4.42m or RM356k.
- If something is missing (no accounts, a missing year), say so. Never fill the gap with a guess.
- Directors and shareholders have placeholder names like "Director 1"; use them as given.
- No advice beyond the risk level, no preamble, no closing remarks."""

RISK_DEFINITIONS = """Risk levels:
- High: any critical issue: status not Active; a pending winding-up petition, or one with judgment entered, against the
  company; an audit disclaimer, adverse opinion or going-concern uncertainty in any year; negative equity; or no
  financial statements at all.
- Elevated: no critical issue, but a material one: a net loss or negative operating cash flow in the latest year;
  current ratio below 1; interest cover below 1.5x; revenue down more than 20% over the years on record; liabilities
  above 80% of assets; a judgment entered against it; pending claims against it of RM500k or more (or 5% of revenue);
  a past winding-up petition; or half the board (or the majority shareholder) appointed in the last 12 months.
- Low: none of the above."""


def _record_json(company: dict) -> str:
    return json.dumps(company, ensure_ascii=False, separators=(",", ":"))


def v1_naive(company: dict, _facts: Facts) -> list[dict]:
    return [
        {
            "role": "user",
            "content": "Summarise this company for a team vetting suppliers. Keep it short.\n\n" + _record_json(company),
        }
    ]


def v2_structured(company: dict, _facts: Facts) -> list[dict]:
    system = f"{AUDIENCE}\n\n{FORMAT}\n\n{RISK_DEFINITIONS}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Company record (JSON):\n" + _record_json(company)},
    ]


def v3_grounded(_company: dict, facts: Facts) -> list[dict]:
    system = (
        f"{AUDIENCE}\n\n{FORMAT}\n\n"
        "You are given facts already checked and computed from the record, including the red flags and the risk level. "
        "Use the risk level as given. Every [critical] and [major] flag must appear in Red flags; mention [minor] ones "
        "only if there is room, in the last section. Copy numbers exactly as written in the facts."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Facts:\n" + facts_brief(facts)},
    ]


VERSIONS = {
    "v1-naive": v1_naive,
    "v2-structured": v2_structured,
    "v3-grounded": v3_grounded,
}
