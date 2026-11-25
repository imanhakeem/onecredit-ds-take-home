"""A second opinion from an LLM judge, on a sample, for what regexes can't see.

    python -m summarizer.judge v1-naive v2-structured v3-grounded

The automatic checks in evaluate.py only see numbers and keywords. A judge reads the raw record and the summary and
lists factual errors in words ("profitable" when it made a loss), risks the summary missed, and anything misleading.
It never sees the version name or the computed flags, so it can't just agree with facts.py.

It's the same model family that wrote the summaries, which may make it lenient, so its scores are a second opinion,
not the main metric. The sample is every High-risk company plus five Elevated and five Low ones, fixed in advance.
"""

import json
import os
import re
import statistics
import sys

from .data import SUMMARIES, load_companies
from .evaluate import EVAL, load_version
from .facts import extract
from .llm import MODEL, LLMError, chat

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", MODEL)
JUDGE_REASONING_EFFORT = os.environ.get("JUDGE_REASONING_EFFORT", "")

RUBRIC = """You check summaries written for a team that vets Malaysian companies before choosing them as suppliers or
partners. The reviewer reads the summary first to decide whether the company deserves a closer look. The record is as
of 31 December 2025.

Compare the SUMMARY with the RECORD (JSON). Be strict and specific. Do all your checking privately first.
Each list item must be a confirmed problem, stated in one sentence. If you check a claim and it is correct, do NOT list
it. An empty list is the right answer when there are no problems. Rounded or truncated numbers (RM5.35m for
5,352,856) are not errors. Mentions of other companies' registration numbers come from other records; ignore them.

Return only JSON, no prose, in this shape:
{
  "factual_errors": ["each statement in the summary that the record contradicts or does not support"],
  "missed_risks": ["each risk a vetting reviewer would need that the summary leaves out"],
  "misleading": ["anything technically true but likely to mislead, e.g. calling a loss-making firm 'stable'"],
  "accuracy": 1-5,       // 5 = no errors at all
  "risk_coverage": 1-5,  // 5 = every material risk is there
  "usefulness": 1-5,     // 5 = a reviewer could decide what to do next from this alone
  "concision": 1-5       // 5 = nothing wasted, readable in under a minute
}"""

SCORES = ("accuracy", "risk_coverage", "usefulness", "concision")


def sample(companies: list[dict]) -> list[str]:
    facts = [extract(c, companies) for c in companies]
    pick = [f.registration_number for f in facts if f.tier == "High"]
    pick += [f.registration_number for f in facts if f.tier == "Elevated"][:5]
    pick += [f.registration_number for f in facts if f.tier == "Low"][:5]
    return pick


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in judge reply")
    return json.loads(re.sub(r"//[^\n]*", "", m.group(0)))


def main() -> None:
    versions = sys.argv[1:] or ["v1-naive", "v2-structured", "v3-grounded"]
    companies = load_companies()
    by_id = {c["registrationNumber"]: c for c in companies}
    ids = sample(companies)
    results: dict[str, list[dict]] = {}

    for v in versions:
        summaries = load_version(SUMMARIES / f"{v}.jsonl")
        results[v] = []
        for rn in ids:
            if rn not in summaries:
                continue
            record = json.dumps(by_id[rn], ensure_ascii=False, separators=(",", ":"))
            messages = [
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content": f"RECORD:\n{record}\n\nSUMMARY:\n{summaries[rn]}"},
            ]
            try:
                r = chat(messages, model=JUDGE_MODEL, temperature=0, reasoning_effort=JUDGE_REASONING_EFFORT)
                verdict = parse(r["text"])
            except (LLMError, ValueError, json.JSONDecodeError) as e:
                print(f"{v} {rn}: judge failed: {e}", file=sys.stderr)
                continue
            results[v].append({"registrationNumber": rn, **verdict})
            print(f"{v} {rn}: " + " ".join(f"{k}={verdict.get(k)}" for k in SCORES))

    EVAL.mkdir(exist_ok=True)
    (EVAL / "judge.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# LLM judge results",
        "",
        f"Judge model: `{JUDGE_MODEL}`. Sample: {len(ids)} companies (all High-risk, 5 Elevated, 5 Low).",
        "",
        "| Metric | " + " | ".join(versions) + " |",
        "|---|" + "---|" * len(versions),
    ]
    for k in SCORES:
        cells = [f"{statistics.mean(x[k] for x in results[v]):.2f}" if results[v] else "n/a" for v in versions]
        lines.append(f"| {k} (mean, 1-5) | " + " | ".join(cells) + " |")
    for k in ("factual_errors", "missed_risks", "misleading"):
        lines.append(f"| {k} (total) | " + " | ".join(str(sum(len(x.get(k, [])) for x in results[v])) for v in versions) + " |")
    lines += ["", "## Every issue the judge raised", ""]
    for v in versions:
        lines += [f"### {v}", ""]
        for x in results[v]:
            issues = [f"{k.replace('_', ' ')}: {i}" for k in ("factual_errors", "missed_risks", "misleading") for i in x.get(k, [])]
            if issues:
                lines.append(f"- **{x['registrationNumber']}**")
                lines += [f"  - {i}" for i in issues]
        lines.append("")
    (EVAL / "judge.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6 + len(SCORES) + 3]))


if __name__ == "__main__":
    main()
