"""Test the evaluator by planting known errors in the v0-template summaries and counting how many it catches.

    python -m summarizer.check_evaluator

1. Wrong numbers: each RM amount / percentage / multiple is changed by a random 8-40% and re-checked. The share caught
   is the number check's sensitivity. (It can't reach 100%: a wrong number can land on another real number by chance.)
2. Dropped flags: one red-flag bullet is deleted at a time. The share the recall check notices is its sensitivity.
"""

import random
import re

from .data import SUMMARIES, load_companies
from .evaluate import NUMBER, allowed_numbers, load_version, mentioned, number_check
from .facts import CRITICAL, MAJOR, extract

random.seed(7)


def perturb(match: re.Match) -> str:
    s = match.group(0)
    num = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    assert num
    raw = num.group(0).replace(",", "")
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    factor = 1 + random.choice([-1, 1]) * random.uniform(0.08, 0.40)
    new = f"{float(raw) * factor:.{decimals}f}"
    return s[: num.start()] + new + s[num.end() :]


def main() -> None:
    companies = load_companies()
    by_id = {c["registrationNumber"]: c for c in companies}
    summaries = load_version(SUMMARIES / "v0-template.jsonl")

    caught = total = 0
    for rn, text in summaries.items():
        facts = extract(by_id[rn], companies)
        allowed = allowed_numbers(by_id[rn], facts)
        for m in NUMBER.finditer(text):
            if "x" in m.group(0).lower() and m.group("mult") and float(m.group("mv")) == 1.5:
                continue  # the 1.5x threshold, not a record value
            wrong = perturb(m)
            _, bad = number_check(wrong, allowed)
            total += 1
            caught += bool(bad)
    print(f"Wrong numbers caught: {caught}/{total} = {caught / total:.0%}")

    noticed = dropped = 0
    for rn, text in summaries.items():
        facts = extract(by_id[rn], companies)
        for flag in [f for f in facts.flags if f.severity in (CRITICAL, MAJOR)]:
            without = text.replace(f"- {flag.text}", "- (removed)")
            dropped += 1
            noticed += not mentioned(flag.code, without)
    print(f"Dropped flags noticed: {noticed}/{dropped} = {noticed / dropped:.0%}")


if __name__ == "__main__":
    main()
