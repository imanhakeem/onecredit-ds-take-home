"""Write summaries/<version>.jsonl for one version.

    python -m summarizer.generate v0-template
    python -m summarizer.generate v3-grounded            # all 50 companies
    python -m summarizer.generate v3-grounded --limit 5  # a quick look first
    python -m summarizer.generate v3-grounded --offline  # rebuild from the cache only, no API calls
"""

import argparse
import json
import re
import sys

from .data import RUNS, SUMMARIES, load_companies
from .facts import extract
from .llm import MODEL, LLMError, QuotaExhausted, chat
from .prompts import VERSIONS
from .template import render

ALL_VERSIONS = ["v0-template", *VERSIONS]


def clean(text: str) -> str:
    """Strip a ```markdown fence if the model wrapped its answer in one."""
    m = re.fullmatch(r"\s*```(?:markdown|md)?\s*\n(.*?)\n```\s*", text, flags=re.S)
    return (m.group(1) if m else text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", choices=ALL_VERSIONS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="*", help="only these registration numbers")
    ap.add_argument("--offline", action="store_true", help="use cached responses only")
    args = ap.parse_args()

    companies = load_companies()
    todo = [c for c in companies if not args.ids or c["registrationNumber"] in args.ids][: args.limit]

    out_rows, log_rows = [], []
    for i, company in enumerate(todo, 1):
        rn = company["registrationNumber"]
        facts = extract(company, companies)
        if args.version == "v0-template":
            text, meta = render(facts), {"model": None}
        else:
            messages = VERSIONS[args.version](company, facts)
            try:
                r = chat(messages, offline=args.offline)
            except QuotaExhausted as e:
                print(f"[{i}/{len(todo)}] {rn}: STOPPING, {e}", file=sys.stderr)
                break
            except LLMError as e:
                print(f"[{i}/{len(todo)}] {rn}: FAILED {e}", file=sys.stderr)
                continue
            text = clean(r["text"])
            meta = {"model": MODEL, "cache_key": r["key"], "cached": r["cached"], "usage": r["usage"]}
            print(f"[{i}/{len(todo)}] {rn}: {'cache' if r['cached'] else 'api'}, {len(text.split())} words", flush=True)
        out_rows.append({"registrationNumber": rn, "summary": text})
        log_rows.append({"registrationNumber": rn, "version": args.version, **meta})

    SUMMARIES.mkdir(exist_ok=True)
    (RUNS / "logs").mkdir(parents=True, exist_ok=True)
    with (SUMMARIES / f"{args.version}.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    with (RUNS / "logs" / f"{args.version}.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in log_rows)
    print(f"Wrote {len(out_rows)} summaries to summaries/{args.version}.jsonl")


if __name__ == "__main__":
    main()
