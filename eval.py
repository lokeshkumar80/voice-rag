"""
Evaluation harness — the part that makes "best model" a measurable claim.

MSMARCO-XI carries relevance labels: for each query, the passage(s) with
is_selected==1 are the GOLD passages, and `Answer` is the gold answer. So we can
compute real IR metrics instead of guessing:

  Recall@k  — did a gold passage make the top-k?
  MRR@10    — 1/rank of the first gold passage (MS MARCO's headline metric)
  nDCG@10   — rank-weighted relevance

Design: we build a self-contained eval index from a held-out slice of the
validation split, remembering which chunks came from gold passages. Then we can
sweep retrieval configs (hybrid alpha, rerank on/off) over the SAME index cheaply,
and rebuild for chunk-strategy comparisons.

Run:
  python eval.py --rows 500                 # sweep alpha + rerank
  python eval.py --rows 500 --chunk semantic
  python eval.py --rows 500 --answer-f1     # also score extractive answers
"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np

import config
from src.chunking import chunk_passage
from src.indexer import HybridIndex, embed
from src.retriever import Retriever
from src.schemas import Chunk


# --------------------------- metrics ---------------------------
def recall_at_k(ranked_ids: List[int], gold: Set[int], k: int) -> float:
    if not gold:
        return 0.0
    hit = len(set(ranked_ids[:k]) & gold)
    return hit / len(gold)


def hit_at_k(ranked_ids: List[int], gold: Set[int], k: int) -> float:
    return 1.0 if set(ranked_ids[:k]) & gold else 0.0


def mrr_at_k(ranked_ids: List[int], gold: Set[int], k: int = 10) -> float:
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: List[int], gold: Set[int], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r, cid in enumerate(ranked_ids[:k], 1) if cid in gold)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = pred.lower().split()
    g = gold.lower().split()
    if not p or not g:
        return 0.0
    common: Dict[str, int] = {}
    for w in p:
        if w in g:
            common[w] = common.get(w, 0) + 1
    overlap = sum(min(common[w], g.count(w)) for w in common)
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


# --------------------------- eval set ---------------------------
@dataclass
class EvalSet:
    index: HybridIndex
    queries: List[str]
    gold_chunk_ids: List[Set[int]]        # per query
    gold_answers: List[str]


def build_eval_set(rows: int, chunk_strategy: str) -> EvalSet:
    from datasets import load_dataset
    print(f"Building eval set: {rows} validation rows, chunk={chunk_strategy} ...")
    ds = load_dataset("parquet", data_files=config.data_files(),
                  split=config.SPLIT, streaming=True)   # eval.py uses split="validation"
    embed_fn = (lambda xs: embed(xs)) if chunk_strategy == "semantic" else None

    chunks: List[Chunk] = []
    queries, gold_ids, gold_answers = [], [], []
    cid = 0
    for r_i, row in enumerate(ds):
        if r_i >= rows:
            break
        passages = row.get("passages") or {}
        texts = passages.get(config.PASSAGE_FIELD) or passages.get("English_passages") or []
        selected = passages.get("is_selected") or [0] * len(texts)
        q = (row.get("query") or "").strip()
        if not q or not texts:
            continue

        gold_for_row: Set[int] = set()
        for i, passage in enumerate(texts):
            if not passage or not passage.strip():
                continue
            is_gold = i < len(selected) and int(selected[i]) == 1
            for piece in chunk_passage(passage.strip(), chunk_strategy,
                                       size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP,
                                       embed_fn=embed_fn,
                                       semantic_threshold=config.SEMANTIC_THRESHOLD,
                                       min_chars=config.MIN_CHUNK_CHARS):
                chunks.append(Chunk(id=cid, text=piece, query_id=row.get("query_id"),
                                    query_type=row.get("query_type"),
                                    is_selected=1 if is_gold else 0, source_row=r_i))
                if is_gold:
                    gold_for_row.add(cid)
                cid += 1
        if gold_for_row:                          # only keep queries with a known gold
            queries.append(q)
            gold_ids.append(gold_for_row)
            gold_answers.append((row.get("Answer") or "").strip())

    print(f"  corpus={len(chunks)} chunks, eval queries={len(queries)}")
    index = HybridIndex.build(chunks)
    return EvalSet(index, queries, gold_ids, gold_answers)


# --------------------------- run one config ---------------------------
def evaluate(es: EvalSet, alpha: float, use_rerank: bool,
             answer_f1: bool = False) -> Dict[str, float]:
    from src import generator
    retr = Retriever(es.index)
    agg = {"R@1": [], "R@5": [], "R@10": [], "Hit@5": [],
           "MRR@10": [], "nDCG@10": [], "F1": []}
    for q, gold, gans in zip(es.queries, es.gold_chunk_ids, es.gold_answers):
        results = retr.retrieve(q, top_k=10, alpha=alpha, use_rerank=use_rerank)
        ranked = [r.chunk.id for r in results]
        agg["R@1"].append(recall_at_k(ranked, gold, 1))
        agg["R@5"].append(recall_at_k(ranked, gold, 5))
        agg["R@10"].append(recall_at_k(ranked, gold, 10))
        agg["Hit@5"].append(hit_at_k(ranked, gold, 5))
        agg["MRR@10"].append(mrr_at_k(ranked, gold, 10))
        agg["nDCG@10"].append(ndcg_at_k(ranked, gold, 10))
        if answer_f1 and gans:
            pred = generator.extractive(q, results)
            agg["F1"].append(token_f1(pred, gans))
    out = {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}
    if not answer_f1:
        out.pop("F1", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--chunk", default=config.CHUNK_STRATEGY,
                    choices=["fixed", "sentence", "recursive", "semantic"])
    ap.add_argument("--answer-f1", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep HYBRID_ALPHA over a grid instead of the 4-config ablation")
    args = ap.parse_args()

    es = build_eval_set(args.rows, args.chunk)

    # Ablation grid: retrieval mode x rerank. alpha=1 dense-only, 0 BM25-only.
    if args.sweep:
        # Fusion weight is the knob that decides whether hybrid beats dense at
        # all; a single hand-picked alpha proves nothing either way.
        configs = [(f"alpha={a:.1f}", a, False)
                   for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    else:
        configs = [
            ("BM25 only",        0.0, False),
            ("Dense only",       1.0, False),
            ("Hybrid a=0.6",     0.6, False),
            ("Hybrid + rerank",  0.6, True),
        ]
    cols = ["R@1", "R@5", "R@10", "Hit@5", "MRR@10", "nDCG@10"]
    if args.answer_f1:
        cols.append("F1")

    print(f"\nchunk={args.chunk}  lang={config.LANG}  embed={config.EMBED_MODEL}")
    print(f"{'config':<18}" + "".join(f"{c:>9}" for c in cols))
    print("-" * (18 + 9 * len(cols)))
    for name, alpha, rr in configs:
        m = evaluate(es, alpha, rr, answer_f1=args.answer_f1)
        print(f"{name:<18}" + "".join(f"{m.get(c,0):>9.3f}" for c in cols))


if __name__ == "__main__":
    main()
