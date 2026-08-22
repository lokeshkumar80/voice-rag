"""
Pick MIN_DENSE_SCORE from data instead of guessing.

The off-topic guardrail abstains when the top dense cosine falls below a
threshold. Set it too low and the system answers nonsense questions from
whatever the nearest chunk happens to be; too high and it abstains on questions
it can actually answer. So we measure both distributions and choose the cut.

  ON-TOPIC  : real queries from the same validation rows the index was built
              from -- these have a gold passage in the corpus by construction.
  OFF-TOPIC : real MS MARCO queries from rows *after* the indexed slice. Natural,
              on-domain questions whose passages were simply never indexed.

⚠ The off-topic set used to be hand-written nonsense ("what is the capital of
Mars", "asdf qwerty"). That was far too easy: gibberish sits far from everything
in embedding space, so it produced a threshold of 0.50 that looked excellent
(0.93 balanced accuracy) and then blocked only **26%** of realistic unanswerable
queries in scripts/faithfulness.py. Calibrating a guardrail on easy negatives
gives you a number, not a guarantee. Hard negatives are the whole point.

Run:  python scripts/calibrate_guardrail.py
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.indexer import HybridIndex
from src.retriever import Retriever

# Kept only as a sanity check -- these should be blocked by any sane threshold.
# They are NOT the calibration set; see the docstring.
TRIVIAL_NEGATIVES = [
    "asdf qwerty zxcv",
    "zzzz plugh xyzzy frobnicate",
    "What is the capital of Mars in the year 3000?",
]


def query_sets(n: int) -> tuple[list[str], list[str]]:
    """(on_topic, off_topic): in-slice queries vs real out-of-slice queries."""
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=config.data_files(),
                      split=config.SPLIT, streaming=True)
    on, off = [], []
    for i, row in enumerate(ds):
        q = (row.get("query") or "").strip()
        if not q:
            continue
        if i < config.MAX_ROWS:
            if len(on) < n:
                on.append(q)
        elif len(off) < n:
            off.append(q)
        if len(on) >= n and len(off) >= n:
            break
    return on, off


def best_dense(retr: Retriever, queries: list[str]) -> np.ndarray:
    scores = []
    for q in queries:
        res = retr.retrieve(q, top_k=config.TOP_K)
        scores.append(max((r.dense_score for r in res), default=0.0))
    return np.array(scores)


def main():
    if not HybridIndex.exists():
        raise SystemExit("No index found. Run `python ingest.py` first.")
    retr = Retriever(HybridIndex.load())

    on_q, off_q = query_sets(150)
    on = best_dense(retr, on_q)
    off = best_dense(retr, off_q)
    trivial = best_dense(retr, TRIVIAL_NEGATIVES)
    print(f"\n(sanity) trivial gibberish max dense score: {trivial.max():.3f} "
          f"-- easy negatives; not what we calibrate on")

    print(f"\non-topic  n={len(on):<4} mean={on.mean():.3f}  "
          f"p05={np.percentile(on,5):.3f}  p25={np.percentile(on,25):.3f}  min={on.min():.3f}")
    print(f"off-topic n={len(off):<4} mean={off.mean():.3f}  "
          f"p75={np.percentile(off,75):.3f}  p95={np.percentile(off,95):.3f}  max={off.max():.3f}")

    print(f"\n{'threshold':>10} {'answers on-topic':>18} {'abstains off-topic':>20} {'balanced':>10}")
    print("-" * 62)
    best_t, best_score = None, -1.0
    for t in np.arange(0.30, 0.86, 0.02):
        tpr = float((on >= t).mean())        # on-topic correctly answered
        tnr = float((off < t).mean())        # off-topic correctly abstained
        bal = (tpr + tnr) / 2
        if bal > best_score:
            best_t, best_score = float(t), bal
        print(f"{t:>10.2f} {tpr:>18.3f} {tnr:>20.3f} {bal:>10.3f}")

    print("-" * 62)
    print(f"\nBest balanced threshold: MIN_DENSE_SCORE={best_t:.2f} "
          f"(balanced accuracy {best_score:.3f})")
    print("The distributions overlap heavily on hard negatives, so this is a")
    print("recall/abstention trade-off, not a clean separation. Cross-check any")
    print("threshold change with scripts/faithfulness.py before trusting it.")


if __name__ == "__main__":
    main()
