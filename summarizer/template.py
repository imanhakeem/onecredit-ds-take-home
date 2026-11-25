"""v0: a summary written by a fixed template, with no model at all.

It is the baseline every LLM version has to beat. It can't be wrong about a number or miss a flag, because it is built
from the same facts the evaluator checks against, but it reads like a form and can't weigh one risk against another.
"""

from .facts import CRITICAL, MAJOR, Facts, fmt_pct, fmt_rm

TIER_LINE = {
    "High": "**Risk: High**: critical issue on record, look closely before engaging.",
    "Elevated": "**Risk: Elevated**: material concerns worth checking.",
    "Low": "**Risk: Low**: no material red flags in the record.",
}


def render(f: Facts) -> str:
    out = [TIER_LINE[f.tier], ""]
    size = f", revenue {fmt_rm(f.latest['revenue'])} in FY{f.latest['year']}" if f.latest else ""
    out.append(f"{f.nature_of_business} ({f.industry}, {f.state}). {f.status}, incorporated {f.incorporated[:4]}{size}.")
    out += ["", "**Red flags**"]
    serious = [x for x in f.flags if x.severity in (CRITICAL, MAJOR)]
    out += [f"- {x.text}" for x in serious] or ["- None found in the record."]
    if f.latest:
        L = f.latest
        trend = f", {fmt_pct(f.trend['revenue_change'])} since FY{f.trend['from_year']}" if f.trend else ""
        out += [
            "",
            f"**Financials (FY{L['year']}):** revenue {fmt_rm(L['revenue'])}{trend}; "
            f"profit {fmt_rm(L['profit_after_tax'])}; equity {fmt_rm(L['total_equity'])}; "
            f"borrowings {fmt_rm(L['borrowings'])}.",
        ]
    out += ["", f"**Ownership:** {f.ownership} {len(f.directors)} directors."]
    lit = f.litigation
    if lit["total"]:
        out.append(
            f"**Litigation:** {lit['total']} cases, {lit['as_defendant']} as defendant, "
            f"{lit['pending_as_defendant']} pending against it."
        )
    else:
        out.append("**Litigation:** none on record.")
    minor = [x.text for x in f.flags if x.severity not in (CRITICAL, MAJOR)]
    if minor:
        out.append("**Also note:** " + " ".join(minor))
    return "\n".join(out)
