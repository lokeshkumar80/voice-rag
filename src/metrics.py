"""
Retrieval-quality and answer-overlap metrics.

Deliberately free of heavy imports -- stdlib only, no torch, faiss, datasets or
numpy. These are pure functions over ranked id lists, so they are the part of the
evaluation that can be unit-tested anywhere: CI runners without a GPU, without
the 460MB dataset, and without network access. `eval.py` imports them from here
rather than defining them, so there is exactly one implementation.

Conventions shared by the ranking metrics:
  ranked_ids  ids in descending relevance order, best first
  gold        set of ids that are actually relevant (is_selected == 1)
  k           cutoff; lists shorter than k are simply used in full

⚠ `recall_at_k` divides by the number of gold *chunks*, so it is not comparable
across chunking strategies that emit different chunk counts for the same passage
(see the caveat in README.md). `hit_at_k` and `mrr_at_k` are denominator-free and
are the fair cross-config comparisons.
"""
from __future__ import annotations

import math
from typing import Dict, List, Set


def recall_at_k(ranked_ids: List[int], gold: Set[int], k: int) -> float:
    """Fraction of gold ids appearing in the top k."""
    if not gold:
        return 0.0
    hit = len(set(ranked_ids[:k]) & gold)
    return hit / len(gold)


def hit_at_k(ranked_ids: List[int], gold: Set[int], k: int) -> float:
    """1.0 if any gold id is in the top k, else 0.0."""
    return 1.0 if set(ranked_ids[:k]) & gold else 0.0


def mrr_at_k(ranked_ids: List[int], gold: Set[int], k: int = 10) -> float:
    """Reciprocal rank of the first gold id, 0.0 if none in the top k."""
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: List[int], gold: Set[int], k: int = 10) -> float:
    """Rank-weighted relevance, normalized by the best achievable ordering."""
    dcg = sum(1.0 / math.log2(r + 1) for r, cid in enumerate(ranked_ids[:k], 1) if cid in gold)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-words F1 between predicted and gold answer text (SQuAD-style)."""
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
