"""
Latency analytics (requirement #4): P50 / P70 / P100 across many queries,
broken down per stage plus the retrieval budget (embed + search + rerank) that
the <200ms target applies to.

By default it benchmarks the RETRIEVAL path with text queries (STT excluded,
since STT is a network call that can't hit 200ms). Use --with-generation to
include answer generation in the totals.

Run:  python benchmark.py --n 200
      python benchmark.py --n 200 --rerank
      python benchmark.py --n 100 --with-generation
"""
from __future__ import annotations
import argparse
import statistics as stats
from typing import List

from datasets import load_dataset

import config
from src.harness import Pipeline
from src.indexer import HybridIndex
from src.retriever import Retriever


def sample_queries(n: int) -> List[str]:
    ds = load_dataset("parquet", data_files=config.data_files(),
                  split=config.SPLIT, streaming=True)   # eval.py uses split="validation"
    qs = []
    for row in ds:
        q = config.row_query(row)
        if q:
            qs.append(q)
        if len(qs) >= n:
            break
    return qs


def pct(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def row(label: str, xs: List[float]) -> str:
    return (f"{label:<22} {pct(xs,50):>8.2f} {pct(xs,70):>8.2f} "
            f"{pct(xs,100):>8.2f} {stats.mean(xs):>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of test queries")
    ap.add_argument("--rerank", action="store_true", help="enable cross-encoder rerank")
    ap.add_argument("--with-generation", action="store_true", help="include generation in totals")
    args = ap.parse_args()

    if not HybridIndex.exists():
        raise SystemExit("No index found. Run `python ingest.py` first.")
    pipe = Pipeline(Retriever(HybridIndex.load()))

    queries = sample_queries(args.n)
    print(f"Benchmarking {len(queries)} queries "
          f"(rerank={args.rerank}, generation={args.with_generation})")

    # Warm up before timing: the first call pays lazy model load, CUDA context
    # init and first-kernel autotune (~10s). Left in, that one-off lands in P100
    # and misrepresents the steady-state retrieval budget we are measuring.
    print("Warming up (model load + CUDA init, excluded from timings) ...")
    for q in queries[:3]:
        pipe.run(text=q, use_rerank=args.rerank,
                 generation_mode=(None if args.with_generation else "extractive"))
    print()

    buckets = {"embed": [], "retrieve": [], "rerank": [],
               "retrieval_total": [], "generate": [], "total": []}
    for q in queries:
        r = pipe.run(text=q, use_rerank=args.rerank,
                     generation_mode=(None if args.with_generation else "extractive"))
        t = r.timing
        buckets["embed"].append(t.embed_ms)
        buckets["retrieve"].append(t.retrieve_ms)
        buckets["rerank"].append(t.rerank_ms)
        buckets["retrieval_total"].append(t.retrieval_total_ms)
        buckets["generate"].append(t.generate_ms)
        total = t.retrieval_total_ms + (t.generate_ms if args.with_generation else 0)
        buckets["total"].append(total)

    print(f"{'stage (ms)':<22} {'P50':>8} {'P70':>8} {'P100':>8} {'mean':>8}")
    print("-" * 60)
    for key in ["embed", "retrieve", "rerank", "retrieval_total", "generate", "total"]:
        if key == "generate" and not args.with_generation:
            continue
        print(row(key, buckets[key]))
    print("-" * 60)
    budget = pct(buckets["retrieval_total"], 100)
    verdict = "PASS ✅" if budget < 200 else "OVER ❌"
    print(f"Retrieval P100 = {budget:.2f} ms vs 200 ms target  ->  {verdict}")


if __name__ == "__main__":
    main()
