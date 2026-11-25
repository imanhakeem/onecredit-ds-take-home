"""Turn a raw company record into the facts and red flags a vetting team cares about.

This is plain code, not a model, on purpose: the flags are the ground truth the summaries are scored against, so they have
to be reproducible and easy to audit. Every threshold is written down here and explained in CASE_STUDY.md.
"""

from dataclasses import asdict, dataclass, field
from datetime import date

from .data import AS_OF

CRITICAL, MAJOR, MINOR = "critical", "major", "minor"
SEVERITY_ORDER = {CRITICAL: 0, MAJOR: 1, MINOR: 2}
TIERS = ("High", "Elevated", "Low")  # High = a critical flag, Elevated = a major flag, Low = neither

# A pending claim against the company counts as major once it is this big, absolutely or relative to revenue.
BIG_CLAIM_RM = 500_000
BIG_CLAIM_SHARE_OF_REVENUE = 0.05
STALE_ACCOUNTS_MONTHS = 18
RECENT_MONTHS = 12


@dataclass
class Flag:
    code: str
    severity: str
    text: str


@dataclass
class Facts:
    registration_number: str
    name: str
    industry: str
    state: str
    nature_of_business: str
    status: str
    incorporated: str
    age_years: int
    share_capital: int
    former_names: list[dict]
    directors: list[dict]
    shareholders: list[dict]
    ownership: str
    financial_years: list[int]
    latest: dict | None
    trend: dict
    audit: list[dict]
    litigation: dict
    flags: list[Flag] = field(default_factory=list)
    tier: str = "Low"
    numbers: dict[str, float] = field(default_factory=dict)  # every derived number, so the evaluator can check claims

    def to_dict(self) -> dict:
        return asdict(self)


# --- formatting: one place, so the prompt, the template and the evaluator all agree on how numbers look -------------


def fmt_rm(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}RM{x / 1e6:.2f}m"
    if x >= 1e3:
        return f"{sign}RM{x / 1e3:.0f}k"
    return f"{sign}RM{x:.0f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _d(s: str) -> date:
    return date.fromisoformat(s)


# --- the extractor ---------------------------------------------------------------------------------------------------


def extract(company: dict, all_companies: list[dict] | None = None) -> Facts:
    p = company["profile"]
    rel = company["relationships"]
    years = sorted(company["financials"]["years"], key=lambda y: y["financialYear"])
    cases = company["litigation"]["cases"]
    flags: list[Flag] = []
    numbers: dict[str, float] = {}

    inc = _d(p["incorporationDate"])
    age = AS_OF.year - inc.year - ((AS_OF.month, AS_OF.day) < (inc.month, inc.day))

    # Status -----------------------------------------------------------------------------------------------------------
    if p["status"] != "Active":
        flags.append(Flag("STATUS_NOT_ACTIVE", CRITICAL, f"Company status is '{p['status']}', not Active."))

    # People and ownership ---------------------------------------------------------------------------------------------
    directors = rel["directors"]
    shareholders = sorted(rel["shareholders"], key=lambda s: -s["percentage"])
    director_names = {d["name"] for d in directors}
    recent_cutoff = date(AS_OF.year - 1, AS_OF.month, AS_OF.day)
    recent = [d for d in directors if _d(d["appointedOn"]) > recent_cutoff]
    top = shareholders[0] if shareholders else None

    if top is None:
        ownership = "No shareholders listed."
    elif top["type"] == "company":
        ownership = f"{top['name']} (a company) holds {top['percentage']:g}%."
    elif top["percentage"] > 50:
        ownership = f"{top['name']} holds a controlling {top['percentage']:g}%."
    else:
        ownership = f"No majority holder; largest is {top['name']} with {top['percentage']:g}%."
    numbers |= {f"share_{i}": s["percentage"] / 100 for i, s in enumerate(shareholders)}

    corporate = [s for s in shareholders if s["type"] == "company"]
    for s in corporate:
        others = [
            o["registrationNumber"]
            for o in all_companies or []
            if o["registrationNumber"] != company["registrationNumber"]
            and any(x["name"] == s["name"] for x in o["relationships"]["shareholders"])
        ]
        also = f" It also holds shares in {', '.join(others)}." if others else ""
        flags.append(
            Flag(
                "CORPORATE_SHAREHOLDER",
                MINOR,
                f"{s['name']} holds {s['percentage']:g}%; the people behind it are not in this record.{also}",
            )
        )

    if recent:
        majority_new = top is not None and top["name"] in {d["name"] for d in recent} and top["percentage"] > 50
        big_turnover = len(recent) * 2 >= len(directors)
        names = ", ".join(f"{d['name']} ({d['appointedOn']})" for d in recent)
        severity = MAJOR if majority_new or big_turnover else MINOR
        why = " including the majority shareholder" if majority_new else ""
        flags.append(
            Flag(
                "RECENT_BOARD_CHANGES",
                severity,
                f"{len(recent)} of {len(directors)} directors appointed in the last 12 months{why}: {names}.",
            )
        )
    non_director_holders = [s["name"] for s in shareholders if s["type"] == "individual" and s["name"] not in director_names]
    numbers["n_directors"] = len(directors)
    numbers["n_recent_directors"] = len(recent)

    for fn in p["formerNames"]:
        if months_between(_d(fn["changedOn"]), AS_OF) <= 24:
            flags.append(Flag("RECENT_NAME_CHANGE", MINOR, f"Renamed on {fn['changedOn']}; formerly {fn['name']}."))

    # Financials -------------------------------------------------------------------------------------------------------
    fys = [y["financialYear"] for y in years]
    latest = None
    trend: dict = {}
    audit = [{"year": y["financialYear"], "firm": y["audit"]["firm"], "opinion": y["audit"]["opinion"]} for y in years]

    if not years:
        flags.append(Flag("NO_FINANCIALS", CRITICAL, "No financial statements on record, so size and health are unknown."))
    else:
        y = years[-1]
        s = y["statements"]
        r = y["ratiosAndMetrics"]
        current_ratio = s["currentAssets"] / s["currentLiabilities"] if s["currentLiabilities"] else None
        debt_to_assets = s["totalLiabilities"] / s["totalAssets"] if s["totalAssets"] else None
        latest = {
            "year": y["financialYear"],
            "year_end": y["financialYearEnd"],
            "accounts_age_months": months_between(_d(y["financialYearEnd"]), AS_OF),
            "revenue": s["revenue"],
            "profit_after_tax": s["profitAfterTax"],
            "net_margin": r["profitability"]["netProfitMargin"],
            "total_equity": s["totalEquity"],
            "total_assets": s["totalAssets"],
            "borrowings": s["borrowings"],
            "cash": s["cash"],
            "operating_cash_flow": s["operatingCashFlow"],
            "current_ratio": current_ratio,
            "liabilities_to_assets": debt_to_assets,
            "interest_cover": r["capitalStructure"]["interestCoverageRatio"],
            "debt_service_cover": r["capitalStructure"]["debtServiceCoverageRatio"],
            "days_sales_outstanding": r["operationalEfficiency"]["daysSalesOutstanding"],
        }
        numbers |= {f"latest_{k}": v for k, v in latest.items() if isinstance(v, int | float) and v is not None}

        # Gaps and staleness
        missing = [fy for fy in range(fys[0], fys[-1] + 1) if fy not in fys]
        if missing:
            flags.append(Flag("MISSING_YEARS", MINOR, f"No accounts for FY{', FY'.join(map(str, missing))}."))
        if latest["accounts_age_months"] >= STALE_ACCOUNTS_MONTHS:
            flags.append(
                Flag(
                    "STALE_ACCOUNTS",
                    MINOR,
                    f"Latest accounts are for the year to {y['financialYearEnd']}, "
                    f"{latest['accounts_age_months']} months before the record date.",
                )
            )
        if len(years) == 1:
            flags.append(Flag("SHORT_HISTORY", MINOR, f"Only one year of accounts (FY{fys[0]}), so no trend."))

        # Audit opinions: anything other than a clean opinion is serious
        for a in audit:
            op = a["opinion"].lower()
            if "disclaimer" in op or "adverse" in op:
                flags.append(
                    Flag("AUDIT_DISCLAIMER", CRITICAL, f"FY{a['year']} audit: '{a['opinion']}' ({a['firm']}).")
                )
            elif "going concern" in op:
                flags.append(Flag("GOING_CONCERN", CRITICAL, f"FY{a['year']} audit: '{a['opinion']}' ({a['firm']})."))
            elif "qualified" in op and not op.startswith("unqualified"):
                flags.append(Flag("AUDIT_QUALIFIED", MAJOR, f"FY{a['year']} audit: '{a['opinion']}' ({a['firm']})."))
        if len({a["firm"] for a in audit}) > 1:
            flags.append(Flag("AUDITOR_CHANGE", MINOR, "Auditor changed: " + ", ".join(f"FY{a['year']} {a['firm']}" for a in audit)))

        # Solvency
        if s["totalEquity"] <= 0:
            flags.append(
                Flag("NEGATIVE_EQUITY", CRITICAL, f"Liabilities exceed assets: equity {fmt_rm(s['totalEquity'])} in FY{fys[-1]}.")
            )
        elif debt_to_assets is not None and debt_to_assets > 0.8:
            flags.append(
                Flag("HIGH_LEVERAGE", MAJOR, f"Liabilities are {fmt_pct(debt_to_assets, 0)} of assets in FY{fys[-1]}.")
            )

        # Profitability
        loss_years = [yy["financialYear"] for yy in years if yy["statements"]["profitAfterTax"] < 0]
        if s["profitAfterTax"] < 0:
            streak = "every year on record" if len(loss_years) == len(years) and len(years) > 1 else f"FY{', FY'.join(map(str, loss_years))}"
            flags.append(
                Flag("LOSS_MAKING", MAJOR, f"Net loss of {fmt_rm(s['profitAfterTax'])} in FY{fys[-1]}; losses in {streak}.")
            )
        elif loss_years:
            flags.append(
                Flag("PAST_LOSSES", MINOR, f"Profitable in FY{fys[-1]} after losses in FY{', FY'.join(map(str, loss_years))}.")
            )

        # Cash and liquidity
        if s["operatingCashFlow"] < 0:
            flags.append(
                Flag("NEGATIVE_OPERATING_CASH", MAJOR, f"Operating cash flow was {fmt_rm(s['operatingCashFlow'])} in FY{fys[-1]}.")
            )
        if current_ratio is not None and current_ratio < 1:
            flags.append(
                Flag("WEAK_LIQUIDITY", MAJOR, f"Current ratio {current_ratio:.2f}: short-term debts exceed short-term assets.")
            )
        ic = latest["interest_cover"]
        if s["interestExpense"] > 0 and ic is not None and ic < 1.5:
            flags.append(Flag("WEAK_INTEREST_COVER", MAJOR, f"Interest cover {ic:.2f}x in FY{fys[-1]} (below 1.5x)."))

        # Trend
        if len(years) >= 2:
            first, prev = years[0]["statements"], years[-2]["statements"]
            trend = {
                "from_year": fys[0],
                "prev_year": fys[-2],
                "revenue_change": s["revenue"] / first["revenue"] - 1,
                "revenue_change_last_year": s["revenue"] / prev["revenue"] - 1,
                "profit_first": first["profitAfterTax"],
                "loss_years": loss_years,
            }
            numbers |= {f"trend_{k}": v for k, v in trend.items() if isinstance(v, int | float)}
            if trend["revenue_change"] < -0.2:
                flags.append(
                    Flag(
                        "REVENUE_DECLINE",
                        MAJOR,
                        f"Revenue fell {fmt_pct(-trend['revenue_change'], 0)} from FY{fys[0]} to FY{fys[-1]} "
                        f"({fmt_rm(first['revenue'])} to {fmt_rm(s['revenue'])}).",
                    )
                )

    # Litigation -------------------------------------------------------------------------------------------------------
    as_def = [c for c in cases if c["role"] == "defendant"]
    pending_def = [c for c in as_def if c["status"] == "Pending"]
    judged_def = [c for c in as_def if c["status"] == "Judgment entered"]
    wup = [c for c in as_def if c["caseType"] == "Winding-up petition"]
    wup_open = [c for c in wup if c["status"] in ("Pending", "Judgment entered")]
    pending_claims = sum(c["claimAmount"] for c in pending_def)
    after_accounts = [c for c in cases if latest and c["filedOn"] > latest["year_end"]]
    litigation = {
        "total": len(cases),
        "as_defendant": len(as_def),
        "as_plaintiff": len(cases) - len(as_def),
        "pending_as_defendant": len(pending_def),
        "pending_claims_against": pending_claims,
        "judgments_against": len(judged_def),
        "winding_up_petitions_against": len(wup),
        "filed_after_latest_accounts": len(after_accounts),
    }
    numbers |= {f"lit_{k}": v for k, v in litigation.items()}
    numbers |= {f"case_{i}": c["claimAmount"] for i, c in enumerate(cases)}
    numbers["lit_claims_against_total"] = sum(c["claimAmount"] for c in as_def)
    numbers["lit_claims_by_total"] = sum(c["claimAmount"] for c in cases if c["role"] == "plaintiff")

    if wup_open:
        detail = "; ".join(f"filed {c['filedOn']}, {c['status'].lower()}, {fmt_rm(c['claimAmount'])}" for c in wup_open)
        flags.append(Flag("WINDING_UP_PETITION", CRITICAL, f"Winding-up petition(s) against the company: {detail}."))
    elif wup:
        detail = "; ".join(f"filed {c['filedOn']}, {c['status'].lower()}" for c in wup)
        flags.append(Flag("PAST_WINDING_UP_PETITION", MAJOR, f"Creditor(s) have petitioned to wind it up before: {detail}."))
    other_judgments = [c for c in judged_def if c["caseType"] != "Winding-up petition"]
    if other_judgments:
        total = sum(c["claimAmount"] for c in other_judgments)
        flags.append(
            Flag("JUDGMENT_AGAINST", MAJOR, f"{len(other_judgments)} judgment(s) entered against it, claims totalling {fmt_rm(total)}.")
        )
    if pending_def:
        revenue = latest["revenue"] if latest else None
        big = pending_claims >= BIG_CLAIM_RM or (revenue and pending_claims >= BIG_CLAIM_SHARE_OF_REVENUE * revenue)
        rel_txt = f" ({fmt_pct(pending_claims / revenue)} of FY{latest['year']} revenue)" if revenue else ""
        if revenue:
            numbers["lit_pending_share_of_revenue"] = pending_claims / revenue
        flags.append(
            Flag(
                "PENDING_CLAIMS_AGAINST",
                MAJOR if big else MINOR,
                f"{len(pending_def)} pending case(s) as defendant, claims totalling {fmt_rm(pending_claims)}{rel_txt}.",
            )
        )

    flags.sort(key=lambda f: SEVERITY_ORDER[f.severity])
    severities = {f.severity for f in flags}
    tier = "High" if CRITICAL in severities else "Elevated" if MAJOR in severities else "Low"

    return Facts(
        registration_number=company["registrationNumber"],
        name=p["companyName"],
        industry=p["industry"],
        state=p["state"],
        nature_of_business=p["natureOfBusiness"],
        status=p["status"],
        incorporated=p["incorporationDate"],
        age_years=age,
        share_capital=p["shareCapital"],
        former_names=p["formerNames"],
        directors=directors,
        shareholders=shareholders + [{"note": f"not a director: {n}"} for n in non_director_holders],
        ownership=ownership,
        financial_years=fys,
        latest=latest,
        trend=trend,
        audit=audit,
        litigation=litigation,
        flags=flags,
        tier=tier,
        numbers=numbers | {"age_years": age, "share_capital": p["shareCapital"]},
    )


def facts_brief(f: Facts) -> str:
    """The facts as a compact, pre-formatted text block for the prompt. Numbers are already rounded and labelled, so
    the model copies them rather than computing them."""
    lines = [
        f"Company: {f.name} ({f.registration_number})",
        f"Business: {f.nature_of_business}. Industry: {f.industry}. State: {f.state}.",
        f"Status: {f.status}. Incorporated {f.incorporated} ({f.age_years} years old). Share capital {fmt_rm(f.share_capital)}.",
        f"Rule-based risk tier: {f.tier}",
        "",
        "Red flags (most severe first):" if f.flags else "Red flags: none found by the rules.",
    ]
    lines += [f"- [{x.severity}] {x.text}" for x in f.flags]
    lines += ["", "Financials:"]
    if f.latest is None:
        lines.append("- None on record.")
    else:
        L = f.latest
        lines.append(
            f"- Years on record: {', '.join(f'FY{y}' for y in f.financial_years)}. "
            f"Latest FY{L['year']} (year to {L['year_end']})."
        )
        lines.append(
            f"- FY{L['year']}: revenue {fmt_rm(L['revenue'])}, profit after tax {fmt_rm(L['profit_after_tax'])} "
            f"(net margin {fmt_pct(L['net_margin'])}), operating cash flow {fmt_rm(L['operating_cash_flow'])}."
        )
        cr = f"{L['current_ratio']:.2f}" if L["current_ratio"] is not None else "n/a"
        ic = f"{L['interest_cover']:.2f}x" if L["interest_cover"] is not None else "n/a"
        lines.append(
            f"- FY{L['year']} balance sheet: equity {fmt_rm(L['total_equity'])}, borrowings {fmt_rm(L['borrowings'])}, "
            f"cash {fmt_rm(L['cash'])}, liabilities/assets {fmt_pct(L['liabilities_to_assets'], 0)}, "
            f"current ratio {cr}, interest cover {ic}."
        )
        if f.trend:
            t = f.trend
            change = f"- Revenue change FY{t['from_year']}->FY{L['year']}: {fmt_pct(t['revenue_change'])}"
            if t["prev_year"] != t["from_year"]:
                change += f"; FY{t['prev_year']}->FY{L['year']}: {fmt_pct(t['revenue_change_last_year'])}"
            lines.append(change + ".")
        lines.append("- Audit: " + "; ".join(f"FY{a['year']} {a['opinion']} ({a['firm']})" for a in f.audit))
    lines += ["", "People and ownership:"]
    lines.append(f"- {len(f.directors)} directors. {f.ownership}")
    lines.append(
        "- Shareholders: "
        + ", ".join(f"{s['name']} {s['percentage']:g}%" for s in f.shareholders if "percentage" in s)
    )
    if f.former_names:
        lines.append("- Former names: " + "; ".join(f"{n['name']} (until {n['changedOn']})" for n in f.former_names))
    lit = f.litigation
    lines += ["", "Litigation:"]
    if lit["total"] == 0:
        lines.append("- No court cases on record.")
    else:
        lines.append(
            f"- {lit['total']} cases: {lit['as_defendant']} as defendant, {lit['as_plaintiff']} as plaintiff. "
            f"{lit['pending_as_defendant']} pending against it (claims {fmt_rm(lit['pending_claims_against'])}), "
            f"{lit['judgments_against']} judgment(s) entered against it, "
            f"{lit['filed_after_latest_accounts']} filed after the latest accounts."
        )
    return "\n".join(lines)
