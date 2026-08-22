"""
Pick MIN_DENSE_SCORE from data instead of guessing.

The off-topic guardrail abstains when the top dense cosine falls below a
threshold. Set it too low and the system answers nonsense questions from
whatever the nearest chunk happens to be; too high and it abstains on questions
it can actually answer. So we measure both distributions and choose the cut.

  ON-TOPIC  : real queries from the same validation rows the index was built
              from -- these have a gold passage in the corpus by construction.
  OFF-TOPIC : gibberish, and questions about things demonstrably not in a
              500-row MSMARCO slice.

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

OFF_TOPIC = [
    "asdf qwerty zxcv",
    "zzzz plugh xyzzy frobnicate",
    "What is the capital of Mars in the year 3000?",
    "Who won the 2047 Martian Grand Prix?",
    "मंगल ग्रह की राजधानी क्या है?",
    "What is my bank account password?",
    "Explain the plot of a movie that was never made.",
    "क्या आप मुझे कल का लॉटरी नंबर बता सकते हैं?",
    "How many purple elephants live inside the sun?",
    "Translate this sentence into a language that does not exist.",
]


def on_topic_queries(n: int) -> list[str]:
    """Queries from the same rows ingest.py indexed -> answerable by construction."""
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=config.data_files(),
                      split=config.SPLIT, streaming=True)
    out = []
    for i, row in enumerate(ds):
        if i >= config.MAX_ROWS or len(out) >= n:
            break
        q = (row.get("query") or "").strip()
        if q:
            out.append(q)
    return out


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

    on = best_dense(retr, on_topic_queries(150))
    off = best_dense(retr, OFF_TOPIC)

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
    print("Note: the two distributions overlap, so this is a recall/abstention")
    print("trade-off, not a clean separation. Lower it to answer more, raise it")
    print("to abstain more.")


if __name__ == "__main__":
    main()
