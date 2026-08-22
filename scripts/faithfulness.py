"""
What do the guardrails actually buy? (Day 3)

A claim like "guardrails reduce hallucination" is meaningless without the
unguarded baseline, so this runs the *same* queries through the pipeline twice --
`guardrails_enabled=True` and `False` -- and reports the difference.

Two query sets, and the negative set is the part that matters:

  ANSWERABLE   queries from the rows that were indexed, which have a gold
               passage in the corpus. Correct behaviour: answer.
  UNANSWERABLE real MS MARCO queries from rows *after* the indexed slice.
               They are on-domain and natural-sounding but their passages were
               never indexed, so nothing in the corpus supports an answer.
               Correct behaviour: abstain. Any answer here is ungrounded.

Using real out-of-slice queries rather than hand-written nonsense ("what is the
capital of Mars") matters: gibberish is trivially far from every embedding, so it
would flatter the abstention rate. These negatives are hard cases -- plausible
questions whose answers simply are not in the index.

The metric is behavioural (did it emit an answer?), not the grounding word-overlap
test itself, so it does not just re-measure the guardrail against its own rule.

Run:  python scripts/faithfulness.py --n 150
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.harness import Pipeline
from src.indexer import HybridIndex
from src.retriever import Retriever


def load_queries(n: int) -> tuple[list[str], list[str]]:
    """(answerable, unanswerable) -- in-slice vs out-of-slice real queries."""
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=config.data_files(),
                      split=config.SPLIT, streaming=True)
    answerable, unanswerable = [], []
    for i, row in enumerate(ds):
        q = (row.get("query") or "").strip()
        if not q:
            continue
        if i < config.MAX_ROWS:
            # only queries whose gold passage is actually in the index
            sel = (row.get("passages") or {}).get("is_selected") or []
            if any(int(x) == 1 for x in sel) and len(answerable) < n:
                answerable.append(q)
        else:
            if len(unanswerable) < n:
                unanswerable.append(q)
        if len(answerable) >= n and len(unanswerable) >= n:
            break
    return answerable, unanswerable


def run_set(pipe: Pipeline, queries: list[str], guardrails: bool,
            use_rerank: bool = False) -> dict:
    answered = 0
    for q in queries:
        r = pipe.run(text=q, guardrails_enabled=guardrails, use_rerank=use_rerank)
        if not r.abstained:
            answered += 1
    n = len(queries) or 1
    return {"n": len(queries), "answered": answered, "rate": answered / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="queries per set")
    ap.add_argument("--rerank", action="store_true",
                    help="enable the cross-encoder, which also switches the "
                         "abstention gate onto its score (see guardrails.py)")
    args = ap.parse_args()

    if not HybridIndex.exists():
        raise SystemExit("No index found. Run `python ingest.py` first.")
    pipe = Pipeline(Retriever(HybridIndex.load()))
    pipe.run(text="warmup")          # exclude cold start

    answerable, unanswerable = load_queries(args.n)
    print(f"rerank                   : {args.rerank}")
    print(f"answerable (in-slice)    : {len(answerable)}")
    print(f"unanswerable (out-slice) : {len(unanswerable)}")

    rows = []
    for label, on in [("guardrails OFF", False), ("guardrails ON", True)]:
        ans = run_set(pipe, answerable, on, args.rerank)
        una = run_set(pipe, unanswerable, on, args.rerank)
        rows.append((label, ans, una))

    print(f"\n{'':<16}{'answered (answerable)':>24}{'answered (unanswerable)':>26}")
    print("-" * 66)
    for label, ans, una in rows:
        print(f"{label:<16}{ans['answered']:>6}/{ans['n']:<3} = {ans['rate']:>6.1%}   "
              f"   {una['answered']:>6}/{una['n']:<3} = {una['rate']:>6.1%}")
    print("-" * 66)

    off_h, on_h = rows[0][2]["rate"], rows[1][2]["rate"]
    off_c, on_c = rows[0][1]["rate"], rows[1][1]["rate"]
    print(f"\nUngrounded answers on unanswerable queries: "
          f"{off_h:.1%}  ->  {on_h:.1%}")
    if off_h > 0:
        print(f"  relative reduction: {(off_h - on_h) / off_h:.0%}")
    print(f"Coverage on answerable queries:              "
          f"{off_c:.1%}  ->  {on_c:.1%}")
    print(f"  cost of the guardrail: {(off_c - on_c) * 100:.1f} points of coverage")
    print("\nThat trade is the honest framing -- abstention is not free, and a "
          "guardrail\nthat abstained on everything would score a perfect 0% "
          "hallucination rate.")


if __name__ == "__main__":
    main()
