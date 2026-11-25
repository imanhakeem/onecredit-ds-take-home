# Case study: company summaries for supplier vetting

**One idea, taken all the way:** code works out the facts and the red flags; the LLM only turns them into readable
text; and every summary is scored, automatically, against those same facts. I then checked that the scoring itself
works by planting errors and seeing how many it catches.

**Models (all free tier, on Groq):** summaries by **`openai/gpt-oss-120b`** (`reasoning_effort=low`, temperature 0.2);
the LLM judge is **`qwen/qwen3.8-27b`**, a different model family, so it is less inclined to like its own writing.
All 50 companies are used for v0, v2 and v3; v1 covers 17 (see [what's left](#whats-left)). Every model response is
saved in `runs/cache/`, so everything here can be rebuilt and re-scored with no API calls:
`just rebuild && just evaluate`.

**Results in one line:** the grounded version (v3) mentioned 100% of critical and major red flags, matched the rule-based
risk tier 100% of the time, quoted 687 numbers with none unverifiable, and used less than half the tokens of the
structured prompt (v2). v2, which reads the raw record, called the company with an audit disclaimer "Low" risk and two
companies facing pending winding-up petitions "Elevated".

## 1. What a good summary is for this team, and why

The team reads the summary *first*, to decide whether a company is worth a closer look. So the summary is a triage
tool, not a description. That gives an order of priorities:

1. **It must not miss a red flag.** A missed winding-up petition or audit disclaimer can lead the team to sign with a
   company that fails. That is far more costly than a few extra seconds of reading, so recall of red flags matters
   more than anything else.
2. **It must not be wrong.** A wrong number, or a "clean audit" that wasn't, is nearly as bad: the reviewer stops
   trusting the summary and reads the whole record anyway, and the summary has saved nothing.
3. **It must lead with the decision.** The first line says how risky the company looks and why, so a reviewer can
   stop there for a clean company.
4. **It must be short and the same shape every time.** At most ~170 words, in fixed sections (risk, what it is, red
   flags, financials, ownership & litigation), so a reviewer reading 50 of them knows where to look.
5. **It must say what's missing.** "No accounts on record" is a finding. A summary that silently skips the
   financials looks clean when it isn't.

What counts as a red flag, from the data (see `summarizer/facts.py` for the exact rules):

| Severity | Red flags |
|---|---|
| **Critical** → risk *High* | status not Active; a pending winding-up petition (or one with judgment entered) against the company; an audit disclaimer, adverse opinion or going-concern uncertainty in any year; negative equity; no financial statements at all |
| **Major** → risk *Elevated* | net loss or negative operating cash flow in the latest year; current ratio < 1; interest cover < 1.5x; revenue down > 20% over the years on record; liabilities > 80% of assets; a judgment entered against it; pending claims against it ≥ RM500k or ≥ 5% of revenue; a past winding-up petition; half the board, or the majority shareholder, appointed in the last 12 months |
| **Minor** (mentioned if there is room) | a missing year of accounts; latest accounts ≥ 18 months old; only one year of accounts; past losses; a company as shareholder; a renamed company; smaller board changes or claims |

Why these: they are the things that make a supplier fail to deliver or fail to exist (insolvency, court-ordered
winding up), the things that make its numbers untrustworthy (audit opinions, missing years), and the things that
change who you are actually dealing with (new controlling directors, holding companies). The thresholds are
conventional credit-analysis rules of thumb (current ratio 1, interest cover 1.5x), chosen before looking at the
summaries, and checked against the data so they don't fire on everyone (10 companies come out High, 20 Elevated,
20 Low).

Some things I found in the data that shaped this:

- **SYN-001569** looks healthy on every ratio, but its FY2022 auditor gave a *disclaimer of opinion*: the auditor
  could not vouch for the accounts at all. A summary built from the financials alone calls it fine.
- **SYN-000286** is *Active*, but a winding-up petition against it has been pending since 2022.
- **SYN-000029**'s 58% shareholder only joined the board on 30 Nov 2025, a month before the record date: control may
  have just changed hands.
- **Gemilang Holdings Sdn. Bhd.** owns 51% of SYN-000381 and 70% of SYN-001203; the summary of one should say the
  owner also controls the other.
- 3 companies have no accounts, 4 are missing FY2023, and the latest accounts are 12–21 months old for everyone.
- The statements are internally consistent (gross profit = revenue − cost of sales, assets = liabilities + equity,
  and so on); I checked, so data errors are not part of this problem.

## 2. What you built

```
data/companies.jsonl
      │
      ▼
summarizer/facts.py ──► facts + red flags + risk tier   (plain Python rules, no model)
      │                         │
      ▼                         ▼
summarizer/prompts.py      summarizer/template.py
      │                    (v0: no model)
      ▼
summarizer/llm.py ◄──► any OpenAI-compatible API: Groq here (cached in runs/cache/)
      │
      ▼
summaries/<version>.jsonl ──► summarizer/evaluate.py ──► eval/results.md, eval/scores.csv
                          └─► summarizer/judge.py    ──► eval/judge.md      (LLM second opinion)
```

Four versions, each changing one thing, so the scores show what each change bought:

| Version | Input to the model | Instructions | Who decides the risk tier |
|---|---|---|---|
| `v0-template` | none (no model) | a fill-in-the-blanks template | the rules |
| `v1-naive` | raw JSON record | "Summarise this company for a team vetting suppliers. Keep it short." | the model, unguided |
| `v2-structured` | raw JSON record | audience, fixed format, risk definitions, "don't calculate, don't guess" | the model, using written definitions |
| `v3-grounded` | **computed facts only** (no raw record) | same as v2 | the rules (given to the model) |

`v3` is the system I'd ship. The model never sees the raw record: it gets a ~850-token brief of checked facts
(`facts_brief()` in `facts.py`) with numbers already rounded and labelled, and the list of flags with severities. It
only has to write. Arithmetic and judgement about what counts as a risk stay in code, where they can be tested and
explained to an auditor. Its prompt is about a third the size of v2's, which matters on a free tier.

`v0` matters too: it is the honest baseline. If an LLM version can't beat a template on the things the team cares
about, the LLM isn't worth its cost or its risk.

## 3. How you know whether it's good

Three layers, from cheapest and most trusted to most expensive and least trusted:

1. **Automatic checks against the facts** (`summarizer/evaluate.py`): run on all 50, no model, a few seconds.
2. **A test of those checks** (`summarizer/check_evaluator.py`): plant errors, count what gets caught.
3. **An LLM judge** (`summarizer/judge.py`) on a fixed sample of 20, plus **my own reading** of every summary the
   first two layers flagged.

### 3.1 The automatic checks

Each one maps to a priority from section 1:

| Priority | Check |
|---|---|
| Don't miss a red flag | **Flag recall:** for each critical and major flag, does the summary mention it? (one regex per flag type) |
| Don't raise a false alarm | **False alarms:** does it claim a winding-up, disclaimer, going concern, negative equity or "no accounts" that the record doesn't have? (negations like "no winding-up petitions" are ignored) |
| Don't be wrong | **Number check:** every RM amount, % and "x" multiple is matched against the record and the derived facts, money only against money, ratios only against ratios, within one unit of the last digit shown. A number that matches nothing is *unverified*. **Date check:** every date must be in the record, and "Director N … date" must be that director's date. |
| Lead with the decision | **Tier agreement:** does the stated risk tier match the rules? |
| Short, same shape | **Length** (≤ 170 words) and **format** (the expected sections are there) |

### 3.2 Testing the checks before trusting them

A metric nobody has tested is just a guess. Three things:

- **The template must score perfectly.** v0 is built from the same facts, so any failure is a bug in the evaluator.
  The first run showed 4 "false alarms" and 19 "unverified numbers", all evaluator bugs: "No accounts for FY2023"
  read as "no accounts at all", "Audit Fi**rm 3**" read as "RM3", rule thresholds like "1.5x" counted as record
  values. After fixing them: zero problems.
- **Plant errors and count what's caught.** Every number in the 50 template summaries was moved by a random 8–40%:
  the number check caught **87%** (357/411). The 13% it misses land, by chance, on another real number in the record.
  Every critical/major flag bullet was deleted one at a time: the recall check noticed **91%** (71/78). The rest
  are flags the template mentions twice, so the summary still covers them (not a miss).
- **Read every summary the checks flag.** When the LLM versions came in, I read every flagged item in context,
  and checked each unverified number against the record by hand. That found more evaluator bugs, all fixed and re-run: this model writes "winding‑up" with a
  non-breaking hyphen and "4.90×" with a times sign (now normalised); "RM 3.2–3.8 m" ranges and "RM2 576 829"
  spacing were misparsed; "short-term liabilities exceed assets" was read as negative equity; "Revenue RM4.51m, down
  22%" was missed because the pattern stopped at the decimal point; "quick ratio below 1" wasn't counted as a
  liquidity mention. What's left in the table below I believe are real model errors.

### 3.3 Results

All 50 companies (v1: the 17 it covers; a like-for-like table on those 17 is in `eval/results.md`, and tells the
same story):

| | v0-template | v1-naive (17) | v2-structured | **v3-grounded** |
|---|---|---|---|---|
| Critical flags mentioned | 100% | 88% | 100% | **100%** |
| Major flags mentioned | 100% | 92% | 95% | **100%** |
| High-risk companies with every critical flag mentioned | 100% | 83% | 100% | **100%** |
| False critical alarms | 0 | 0 | 0 | **0** |
| Numbers that match nothing in the record | 0 | 6 | 1 | **0** (of 687) |
| Summaries with a wrong or swapped date | 0 | 0 | 0 | 1 |
| States a risk tier | 100% | 0% | 100% | 100% |
| Tier agrees with the rules | 100% | n/a | 82% | **100%** |
| Median words | 82 | 333 | 121 | 120 |
| Within 170 words | 100% | 6% | 98% | 100% |
| Tokens per summary (prompt + output) | 0 | 2,255 | 2,583 | **1,145** |

**The LLM judge** (`qwen/qwen3.8-27b`, sample of 20 fixed in advance: all 10 High-risk companies, 5 Elevated, 5 Low).
Qwen's free daily cap ran out mid-run, so it scored v2 on all 20 and v3 on 18 (two replies weren't valid JSON),
the template on 5 and v1 on none. On the 18 companies both have:

| Judge, mean of 1–5 | v2-structured | v3-grounded |
|---|---|---|
| Accuracy | 3.94 | **4.28** |
| Risk coverage | 3.33 | 3.28 |
| Usefulness | 3.94 | 3.94 |
| Concision | 4.83 | **5.00** |

The judge's scores barely separate the two, but I don't think they should be trusted as scores, because I read every
"factual error" it listed for v3 (28 items):

- **7 items pointed at 5 real problems, in 5 summaries**, and 4 of them were invisible to my checks: the swapped director dates
  (SYN-000545); "no claim amount recorded" for a RM169,800 case (SYN-000165); "5 plaintiff lawsuits, none pending"
  when one is (SYN-000267); "no accounts *filed* for FY2023" when the record only shows they're missing (SYN-000286);
  and "1 judgment entered against it" for SYN-001212, which has 3. That last one is **my** wording in `facts.py`:
  the flag counts judgments outside winding-up petitions, but doesn't say so.
- **2 were the judge's own mistakes** (for one, it summed RM58,600 + RM721,100 to RM793,100).
- **The other 19 weren't errors:** the judge re-checked a claim inside the list and concluded "…which is actually
  correct", or gave an opinion (a Low rating "should" reflect leverage of 1.65).

For v2 it also found real errors my checks missed: an invented "pending winding-up petition" (SYN-001212, where the
case is an employment claim) and an operating loss of RM6.69m labelled as profit after tax (SYN-000067).

So the judge is useful as a **high-recall, low-precision error finder** to point a human at summaries worth reading,
and not useful as a scorer: 7 of its 28 "errors" for v3 (1 in 4) were real.

### 3.4 What the errors look like

The numbers above are counts. These are the cases behind them, and they matter more:

- **v2 rated the audit-disclaimer company "Low" risk** (SYN-001569): "No critical or material issues; finances
  stable and audit opinions clean after 2022". It even listed the disclaimer as a red flag, then padded the list
  with "None found in the record." twice. This is the failure the whole design exists to prevent: an LLM that
  reads the record and *knows* the fact can still weigh it wrongly.
- **v2's tier errors lean the dangerous way.** 7 of its 9 disagreements with the rules under-rate the risk: two
  pending winding-up petitions (SYN-000286, SYN-001389) and negative equity (SYN-000451) rated Elevated instead of
  High, and a new majority-owner director (SYN-000029) rated Low. For a vetting team, under-rating is the costly
  direction.
- **When the model does its own arithmetic, it's often wrong.** I checked every unverified number by hand. Wrong:
  "borrowings 55% of total assets" (actual 31%), "operating cash flow down 55%" (49.5%), "revenue declining 12–19%
  a year" (16% and 21%), "RM 2.05 M" revenue (RM2.08m), and total assets of "RM 22.6 **B**", a thousand times too
  large. Right: "equity ≈ 41% of assets", "revenue fell >30%". That's why v3 hands the model finished numbers.
- **v1 has no shape.** Median 333 words, tables and bullet trees, no risk level, and it skipped SYN-000545's
  going-concern warning altogether.
- **v3's errors are copying and gap-filling errors.** It swapped two directors' appointment dates (SYN-000545),
  writing that the majority owner joined on the other director's date. Every date existed in the record, so the
  number check couldn't see it; I noticed it reading the output and added the date check, which now catches it. The
  judge found two more: v3 said "none pending" about plaintiff cases and "no claim amount recorded" about a
  RM169,800 case. My brief spells out pending cases *against* the company but not cases it brought, and the model
  filled the gap with a guess. The lesson for the design: anything the brief leaves out, the model may invent, so
  the brief has to be complete for every field the summary format asks about.
- **v3 bends one rule.** For 16 of 20 Low-risk companies it put minor notes ("latest accounts are 18 months old")
  under "Red flags", when told to keep them for the last section. A clean company then looks flagged at a glance.
- **Nobody surfaced the cross-company link.** Gemilang Holdings owns 51% of SYN-000381 and 70% of SYN-001203; the
  fact is in the brief, but as a minor flag, and no version mentioned it.

### 3.5 Limits of this evaluation

- **The checks share the rules' blind spots.** Recall is measured against flags *I* defined. If a risk isn't in
  `facts.py`, no version is penalised for missing it. The judge, which reads the raw record, is the partial guard
  against that, and it did find a real wording flaw and a real gap in my facts (see 3.3).
- **The judge sample is small and unfinished:** 18–20 summaries per version, and none for v1, because of free-tier
  caps. Its scores are context, not evidence.
- **v3 is partly graded on its own answer key.** It receives the flags, so its recall shows that it *passes them on*,
  not that it *finds* them. That's the point of the design, but it means v2 vs v3 compares "find and write" with
  "write". The test that the flags are right is my manual check of them (section 1) plus the judge.
- **Regexes miss paraphrase.** I fixed every miss I found by reading, but a summary that says "the business is
  shrinking" instead of "revenue fell" would still count as a miss.
- **One model, one run, no human reviewers.** Temperature 0.2 and a single sample per company; I didn't measure
  run-to-run variation. Nobody from the vetting team has read these, so "useful" is my judgement, not theirs.

## 4. What you'd do next, and how you'd keep improving it

### What's left

- **v1 covers 17 companies, not 50.** Groq's free daily token cap for `gpt-oss-120b` (200k) ran out during the
  third version. `just summarize v1-naive` tomorrow finishes it from the cache. I don't expect it to change the
  conclusion: on the 17, v1 already misses critical flags and is 3× too long.
- **The judge didn't finish** the template (5 of 20) or v1 (0 of 20) before Qwen's daily cap ran out.
  `just judge v3-grounded v2-structured v0-template v1-naive` tomorrow completes it from the cache.
- **Provider churn cost real time.** I started on Gemini; its free tier now allows 20 requests a day per model, and
  its open Gemma models were overloaded. I moved to Groq. The client speaks the OpenAI-compatible API, so switching
  provider was a two-line change in `.env`, which is worth keeping.

### Next, in order

1. **Write the red-flag list in code, let the model write the rest.** The two v3 errors (a swapped date, minor flags
   in the wrong section) were both in parts that are pure copying. Render "Red flags" from `facts.py` directly and
   ask the model for the risk sentence, the description and the financial narrative only. It removes a whole class of
   error for free.
2. **Fix the gaps the judge found in my facts.** For SYN-001212, "1 judgment entered against it" is true only for
   non-winding-up cases; there are 3 judgments in all, and the wording should say so. Give the brief one line per
   court case (role, type, status, amount), so the model never has to guess about cases the company brought.
   Promote cross-company links (shared holding companies) from minor to a line of their own; it's the one finding no
   version reported. Teach the date check other formats ("3 May 2025"), which v2 used and the check can't read.
3. **Get the team to label 20 summaries.** Ask two reviewers, for each summary: would you have looked closer? Did it
   miss anything you needed? That turns "useful" from my opinion into a measured number, and their disagreements
   are where my flag definitions are wrong. Some are debatable already: is "no accounts on record" really *High*
   risk, or a data request? The judge thought the latter; I chose High because a company you can't assess isn't one
   you can approve, but that's the team's call.
4. **Measure consistency.** Run v3 five times per company at its current temperature and check that the tier, flags
   and numbers don't change. A summary that varies between runs can't be trusted even if each run looks fine.
5. **Check paraphrase recall with the judge, not regexes,** and measure the judge against the hand labels from
   step 3 before trusting it more.

### How I'd keep improving it in use

- **Every summary ships with its facts.** Store the facts and flags next to each summary, so a reviewer can see why
  it says "High", and an auditor can replay it.
- **Run the automatic checks on every summary before a reviewer sees it.** They're cheap and need no model. A summary
  that misses a critical flag, quotes an unverified number or contradicts the rules' tier is held back, and a
  template version is shown instead. The template is already a safe fallback: it scored perfectly on every check.
- **Track two numbers over time:** critical-flag recall (target 100%; any miss is an incident to review) and how often
  reviewers overrule the risk tier. Overrules show where the rules are wrong, and the rules are where
  improvements go first, because every version inherits them.
- **Treat the rules as the product.** New red flags (a director on another struggling company's board, a sudden
  auditor change) go into `facts.py` with a test; the prompt barely needs to change.

## Tools I used

- **Claude Code** (Anthropic): I worked with it as a pair programmer throughout: exploring the data, writing the code,
  running the experiments, and drafting this write-up. I reviewed the results and the error analysis above, and can
  explain every part of the code.
- **Groq API**, free tier: `openai/gpt-oss-120b` for summaries, `qwen/qwen3.8-27b` as judge.
- **Google Gemini API**, free tier: tried first; daily request caps and overloaded models made it unusable for
  a full run (details above).

## How to reproduce

```bash
cp .env.example .env              # add a free Groq or Gemini key
just rebuild                      # rebuild every summaries/*.jsonl from the saved responses, no API calls
just evaluate                     # -> eval/results.md, eval/scores.csv
just check-evaluator              # plant errors, see how many the checks catch
just summarize v3-grounded        # regenerate a version (cached responses are reused)
just judge v3-grounded v2-structured v0-template v1-naive   # -> eval/judge.md
```

The code needs only the Python standard library. Without Docker, the viewer also runs with
`uv run uvicorn app.main:app --port 8000`.
