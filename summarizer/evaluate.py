"""Score every version in summaries/ against the facts, with no model calls.

    python -m summarizer.evaluate            # all versions -> eval/results.md and eval/scores.csv

What it measures, per summary (CASE_STUDY.md explains why these):
1. Flag recall: of the critical and major flags facts.py found, how many does the summary mention?
2. False alarms: does it claim a critical issue (winding-up, disclaimer, going concern, negative equity, no accounts)
   that the record doesn't have?
3. Number check: every RM amount, percentage and "x" multiple in the summary is compared with the numbers in the record
   and the derived facts. One that matches nothing is "unverified": a wrong number, or one the model calculated.
4. Risk tier: does the stated tier match the rules?
5. Length and format: word count, and whether the expected sections are there.

The mention checks are regular expressions, so they can be wrong. They were checked by hand against a sample of
summaries; the misses found are listed in CASE_STUDY.md.
"""

import csv
import json
import re
import statistics
from pathlib import Path

from .data import AS_OF, ROOT, SUMMARIES, load_companies
from .facts import CRITICAL, MAJOR, Facts, extract

EVAL = ROOT / "eval"

# How a summary shows it has mentioned each flag.
MENTION = {
    "STATUS_NOT_ACTIVE": r"status[^.\n]{0,40}winding|winding[\s-]up status|being wound up|under(going)? winding|"
    r"in (the process of )?winding[\s-]up|not active|liquidat|is winding[\s-]up|\(winding[\s-]up\)|winding[\s-]up company",
    "WINDING_UP_PETITION": r"winding[\s-]*up petition|petition",
    "PAST_WINDING_UP_PETITION": r"winding[\s-]*up petition|petition",
    "AUDIT_DISCLAIMER": r"disclaim",
    "GOING_CONCERN": r"going[\s-]concern",
    "AUDIT_QUALIFIED": r"qualified opinion",
    "NEGATIVE_EQUITY": r"negative (total |shareholders'? )?equity|liabilities exceed|net liabilit|insolven|"
    r"equity[^.\n]{0,25}(-|−)\s?RM|equity[^.\n]{0,25}RM\s?(-|−)|deficit",
    "NO_FINANCIALS": r"no (financial|accounts|audited)|financ[^.\n]{0,40}(missing|unavailable|not (available|on record|provided))|"
    r"without (any )?(financial|accounts)|absence of (financial|accounts)|lack of (financial|accounts)",
    "LOSS_MAKING": r"loss",
    "NEGATIVE_OPERATING_CASH": r"(negative|outflow)[^.\n]{0,40}cash|operating cash[^.\n]{0,40}(negative|outflow|(-|−)\s?RM|RM\s?(-|−))|"
    r"cash burn|burn(ing|ed)? cash",
    "WEAK_LIQUIDITY": r"current ratio|quick ratio|liquidity|short-term (debts|liabilities|obligations) exceed",
    "WEAK_INTEREST_COVER": r"interest cover",
    "HIGH_LEVERAGE": r"leverage|geared|gearing|liabilities[^.\n]{0,40}\d+% of (total )?assets",
    "REVENUE_DECLINE": r"(revenue|sales)[^.\n]{0,60}(fell|fall|declin|drop|down|decreas|shr[au]nk|contract)|"
    r"(declin|drop|fall|fell)[^.\n]{0,30}(revenue|sales)",
    "JUDGMENT_AGAINST": r"judge?ment",
    "PENDING_CLAIMS_AGAINST": r"pending",
    "RECENT_BOARD_CHANGES": r"appointed|new director|board (change|turnover|reshuffle|overhaul)|recent(ly)? (appoint|join|added)|joined",
}

# Claims that would be false alarms if the record doesn't support them.
ALARMS = {
    "winding-up": (r"winding[\s-]*up|wound up|liquidat", {"STATUS_NOT_ACTIVE", "WINDING_UP_PETITION", "PAST_WINDING_UP_PETITION"}),
    "disclaimer": (r"disclaim", {"AUDIT_DISCLAIMER"}),
    "going concern": (r"going[\s-]concern", {"GOING_CONCERN"}),
    # "short-term liabilities exceed assets" is a liquidity point (current ratio < 1), not negative equity
    "negative equity": (r"negative (total )?equity|(?<!short-term )(?<!current )liabilities exceed (total )?assets|insolven",
                        {"NEGATIVE_EQUITY"}),
    # "No accounts for FY2023" is a missing year, not a missing history, so a year straight after is not an alarm.
    "no accounts": (r"(no (financial|accounts)|financial (statements|data|information) (are )?(missing|unavailable|not available))"
                    r"(?![^.\n]{0,15}FY\d{4})", {"NO_FINANCIALS"}),
}
NEGATION = re.compile(r"\b(no|not|nor|without|never|none|free of|clean)\b[^.\n]{0,40}$", re.I)

NUMBER = re.compile(
    # "RM2 576 829" (space as thousands separator) and "RM 3.2-3.8 m" (a range; the unit belongs to both ends)
    r"(?P<rm>(?-i:\bRM)\s?(?:-|−)?\s?(?P<rmv>\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?-\s?\d+(?:\.\d+)?)?\s?(?P<unit>bn|billion|b|m|mil|million|k)?\b)"
    r"|(?P<pct>(?P<pv>\d+(?:\.\d+)?)\s?%)"
    r"|(?P<mult>(?P<mv>\d+(?:\.\d+)?)\s?x\b)",
    re.I,
)
# Thresholds the summaries may quote when explaining a flag ("interest cover below 1.5x"). They are not record values.
RULE_THRESHOLDS = [1.5, 1.0, 0.8, 0.2, 0.05, 500_000.0]
UNITS = {"k": 1e3, "m": 1e6, "mil": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


def _leaves(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _leaves(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in _leaves(x)]
    return []


def allowed_numbers(company: dict, facts: Facts) -> list[float]:
    vals = _leaves(company) + [float(v) for v in facts.numbers.values() if isinstance(v, int | float)]
    vals += [v / 100 for v in _leaves(company["relationships"])]  # shareholdings are stored as 58.25, not 0.5825
    vals += RULE_THRESHOLDS
    return sorted({abs(v) for v in vals})


def parse_numbers(text: str) -> list[tuple[str, str, float, float]]:
    """Return (as written, kind, value, tolerance). Kind is "money" or "ratio", so an RM amount is only compared with
    amounts and a percentage only with ratios. Tolerance is half a unit of the last digit shown (rounding), or 0.5%."""
    out = []
    for m in NUMBER.finditer(text):
        if m.group("rm"):
            raw = m.group("rmv").replace(",", "").replace(" ", "")
            scale, kind = UNITS.get((m.group("unit") or "").lower(), 1), "money"
        elif m.group("pct"):
            raw, scale, kind = m.group("pv"), 0.01, "ratio"
        else:
            raw, scale, kind = m.group("mv"), 1, "ratio"
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        value = float(raw) * scale
        # One unit of the last digit shown: accepts rounding *and* truncation (RM138k for 138,894), which a reviewer
        # wouldn't call wrong. Anything further off is flagged.
        out.append((m.group(0), kind, value, max(1.0 * 10**-decimals * scale * 1.05, 0.005 * value)))
    return out


def _kind(v: float) -> str:
    # Amounts in this data are whole ringgit in the thousands or more; ratios, margins and multiples are small.
    return "money" if abs(v) >= 1000 else "ratio"


def number_check(text: str, allowed: list[float]) -> tuple[int, list[str]]:
    found = parse_numbers(text)
    bad = [s for s, kind, v, tol in found if not any(_kind(a) == kind and abs(v - a) <= tol for a in allowed)]
    return len(found), bad


DATE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
DIRECTOR_DATE = re.compile(r"\b(Director \d+)\b[^.;\n]{0,30}?\b((?:19|20)\d{2}-\d{2}-\d{2})\b")


def date_check(text: str, company: dict) -> list[str]:
    """Dates that aren't in the record at all, and "Director N ... <date>" pairs where the date belongs to someone else.
    The second catches a model swapping two directors' appointment dates, which the number check can't see."""
    record_dates = set(re.findall(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", json.dumps(company))) | {AS_OF.isoformat()}
    appointed = {d["name"]: d["appointedOn"] for d in company["relationships"]["directors"]}
    bad = [m.group(0) for m in DATE.finditer(text) if m.group(0) not in record_dates]
    for m in DIRECTOR_DATE.finditer(text):
        name, d = m.group(1), m.group(2)
        if d in appointed.values() and appointed.get(name) != d:
            bad.append(f"{name} {d} (record: {appointed.get(name)})")
    return bad


# "[^.\n]" in the patterns means "in the same sentence", but a decimal point ("RM4.51m, down 22%") isn't the end of one.
SAME_SENTENCE = r"(?:[^.\n]|\.(?=\d))"


def _p(pattern: str) -> str:
    return pattern.replace(r"[^.\n]", SAME_SENTENCE)


def mentioned(code: str, text: str) -> bool:
    return bool(re.search(_p(MENTION.get(code, r"(?!x)x")), text, re.I))


def false_alarms(text: str, facts: Facts) -> list[str]:
    codes = {f.code for f in facts.flags}
    out = []
    for name, (pattern, supported_by) in ALARMS.items():
        if codes & supported_by:
            continue
        for m in re.finditer(_p(pattern), text, re.I):
            if not NEGATION.search(text[max(0, m.start() - 60) : m.start()]):
                out.append(name)
                break
    return out


def stated_tier(text: str) -> str | None:
    m = re.search(r"risk(?: level| rating| tier)?\**\s*[:\-]\s*\**\s*(high|elevated|medium|moderate|low)", text, re.I)
    if not m:
        return None
    t = m.group(1).capitalize()
    return "Elevated" if t in ("Medium", "Moderate") else t


def normalize(text: str) -> str:
    """Models write "winding‑up" with a non-breaking hyphen, "4.90×" with a times sign, ‘curly’ quotes, and so on.
    Map them to plain ASCII so the patterns above see one spelling."""
    text = re.sub(r"[‐-―−]", "-", text)
    text = text.replace("×", "x").replace(" ", " ").replace(" ", " ")
    return re.sub(r"[‘’]", "'", text)


def score(text: str, company: dict, facts: Facts) -> dict:
    text = normalize(text)
    crit =[f.code for f in facts.flags if f.severity == CRITICAL]
    major = [f.code for f in facts.flags if f.severity == MAJOR]
    n_numbers, unverified = number_check(text, allowed_numbers(company, facts))
    tier = stated_tier(text)
    return {
        "registrationNumber": facts.registration_number,
        "rule_tier": facts.tier,
        "stated_tier": tier,
        "tier_match": None if tier is None else tier == facts.tier,
        "critical_flags": len(crit),
        "critical_hit": sum(mentioned(c, text) for c in crit),
        "major_flags": len(major),
        "major_hit": sum(mentioned(c, text) for c in major),
        "missed": [c for c in crit + major if not mentioned(c, text)],
        "false_alarms": false_alarms(text, facts),
        "numbers": n_numbers,
        "unverified_numbers": unverified,
        "wrong_dates": date_check(text, company),
        "words": len(text.split()),
        "format_ok": all(
            re.search(p, text, re.I)
            for p in (r"\*\*risk", r"\*\*red flags", r"\*\*ownership", *([r"\*\*financials"] if facts.latest else []))
        ),
    }


def summarise(rows: list[dict]) -> dict:
    def ratio(a: str, b: str) -> float | None:
        tb = sum(r[b] for r in rows)
        return sum(r[a] for r in rows) / tb if tb else None

    stated = [r for r in rows if r["tier_match"] is not None]
    high = [r for r in rows if r["rule_tier"] == "High"]
    return {
        "n": len(rows),
        "critical_recall": ratio("critical_hit", "critical_flags"),
        "major_recall": ratio("major_hit", "major_flags"),
        "high_risk_fully_caught": sum(r["critical_hit"] == r["critical_flags"] for r in high) / len(high) if high else None,
        "false_alarm_summaries": sum(bool(r["false_alarms"]) for r in rows),
        "numbers": sum(r["numbers"] for r in rows),
        "unverified_count": sum(len(r["unverified_numbers"]) for r in rows),
        "wrong_date_summaries": sum(bool(r["wrong_dates"]) for r in rows),
        "tier_stated": len(stated) / len(rows),
        "tier_agreement": sum(r["tier_match"] for r in stated) / len(stated) if stated else None,
        "median_words": statistics.median(r["words"] for r in rows),
        "max_words": max(r["words"] for r in rows),
        "within_170_words": sum(r["words"] <= 170 for r in rows) / len(rows),
        "format_ok": sum(r["format_ok"] for r in rows) / len(rows),
    }


def load_version(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return {(r := json.loads(line))["registrationNumber"]: r["summary"] for line in f if line.strip()}


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.0%}" if v <= 1 else f"{v:.0f}"
    return str(v)


def main() -> None:
    companies = load_companies()
    facts = {c["registrationNumber"]: extract(c, companies) for c in companies}
    by_id = {c["registrationNumber"]: c for c in companies}
    versions = sorted(p.stem for p in SUMMARIES.glob("*.jsonl"))
    EVAL.mkdir(exist_ok=True)

    all_rows, totals = [], {}
    for v in versions:
        summaries = load_version(SUMMARIES / f"{v}.jsonl")
        rows = [score(text, by_id[rn], facts[rn]) | {"version": v} for rn, text in summaries.items() if rn in by_id]
        totals[v] = summarise(rows)
        all_rows += rows

    with (EVAL / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["version", *[k for k in all_rows[0] if k != "version"]])
        w.writeheader()
        w.writerows({k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()} for r in all_rows)

    metrics = [
        ("Companies scored", "n"),
        ("Critical flags mentioned", "critical_recall"),
        ("Major flags mentioned", "major_recall"),
        ("High-risk companies with every critical flag mentioned", "high_risk_fully_caught"),
        ("Summaries with a false critical alarm", "false_alarm_summaries"),
        ("Numbers quoted", "numbers"),
        ("Numbers that match nothing in the record", "unverified_count"),
        ("Summaries with a wrong or swapped date", "wrong_date_summaries"),
        ("Summaries that state a risk tier", "tier_stated"),
        ("Stated tier agrees with the rules", "tier_agreement"),
        ("Median words", "median_words"),
        ("Longest summary (words)", "max_words"),
        ("Summaries within 170 words", "within_170_words"),
        ("Follows the format", "format_ok"),
    ]
    def table(t: dict) -> list[str]:
        out = ["| Metric | " + " | ".join(versions) + " |", "|---|" + "---|" * len(versions)]
        return out + [f"| {label} | " + " | ".join(_fmt(t[v][key]) for v in versions) + " |" for label, key in metrics]

    lines = ["# Evaluation results", "", "Generated by `python -m summarizer.evaluate`. No model calls.", ""]
    lines += ["## All summaries", "", *table(totals)]

    # If a version is incomplete, also compare every version on the companies they all cover, so it's like for like.
    covered = set.intersection(*({r["registrationNumber"] for r in all_rows if r["version"] == v} for v in versions))
    if len(covered) < len(companies):
        same = {v: summarise([r for r in all_rows if r["version"] == v and r["registrationNumber"] in covered]) for v in versions}
        lines += ["", f"## Only the {len(covered)} companies every version covers", "", *table(same)]
    lines += ["", "## Where each version went wrong", ""]
    for v in versions:
        rows = [r for r in all_rows if r["version"] == v]
        lines += [f"### {v}", ""]
        problems = 0
        for r in rows:
            bits = []
            if r["missed"]:
                bits.append("missed " + ", ".join(r["missed"]))
            if r["false_alarms"]:
                bits.append("false alarm: " + ", ".join(r["false_alarms"]))
            if r["unverified_numbers"]:
                bits.append("unverified numbers: " + ", ".join(r["unverified_numbers"]))
            if r["wrong_dates"]:
                bits.append("wrong dates: " + ", ".join(r["wrong_dates"]))
            if r["tier_match"] is False:
                bits.append(f"tier {r['stated_tier']} vs rules {r['rule_tier']}")
            if bits:
                problems += 1
                lines.append(f"- **{r['registrationNumber']}** ({r['rule_tier']}): " + "; ".join(bits))
        if not problems:
            lines.append("- No problems found by the automatic checks.")
        lines.append("")
    (EVAL / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4 + 2 + len(metrics)]))


if __name__ == "__main__":
    main()
