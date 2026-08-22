"""
Hybrid retrieval:
  1. dense (FAISS) top-N + BM25 top-N candidates
  2. min-max normalize each score list, fuse: alpha*dense + (1-alpha)*bm25
  3. optional cross-encoder rerank on the fused top candidates
  4. optional metadata filter (e.g. query_type) applied before scoring

Returns RetrievedChunk objects with per-signal scores for transparency.
"""
from __future__ import annotations
import time
from functools import lru_cache
from typing import Callable, List, Optional

import numpy as np

import config
from src.indexer import HybridIndex, embed, _tokenize
from src.schemas import RetrievedChunk, StageTiming


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)


@lru_cache(maxsize=1)
def _get_reranker():
    """Cross-encoder, sharing the GPU with the bi-encoder. bge-reranker-v2-m3
    defaults to an 8192 max_length; capped to EMBED_MAX_SEQ for the same reason
    the embedder is (see config), and to leave room for both models on 8GB."""
    from sentence_transformers import CrossEncoder
    device = config.DEVICE
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    return CrossEncoder(config.RERANK_MODEL, device=device,
                        max_length=config.EMBED_MAX_SEQ)


class Retriever:
    def __init__(self, index: HybridIndex):
        self.index = index

    def retrieve(self, query: str, *, top_k: int = config.TOP_K,
                 candidates: int = config.CANDIDATES,
                 alpha: float = config.HYBRID_ALPHA,
                 use_rerank: bool = config.USE_RERANK,
                 metadata_filter: Optional[Callable[[dict], bool]] = None,
                 timing: Optional[StageTiming] = None
                 ) -> List[RetrievedChunk]:
        chunks = self.index.chunks
        n = len(chunks)
        if n == 0:
            return []
        candidates = min(candidates, n)

        # --- dense: embed query, then FAISS search ---
        t = time.perf_counter()
        qvec = embed([query])                       # normalized
        if timing is not None:
            timing.embed_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        dscores, didx = self.index.faiss.search(qvec, candidates)
        dscores, didx = dscores[0], didx[0]

        # --- lexical (BM25) over full corpus, then take top candidates ---
        bm25_all = np.asarray(self.index.bm25.get_scores(_tokenize(query)), dtype="float32")
        bidx = np.argsort(-bm25_all)[:candidates]

        # union of candidate ids
        cand_ids = list(dict.fromkeys([int(i) for i in didx if i >= 0] + [int(i) for i in bidx]))

        dense_map = {int(i): float(s) for i, s in zip(didx, dscores) if i >= 0}
        d_arr = np.array([dense_map.get(cid, 0.0) for cid in cand_ids], dtype="float32")
        b_arr = np.array([float(bm25_all[cid]) for cid in cand_ids], dtype="float32")
        if config.FUSION == "rrf":
            # Fuse RANKS, not scores. Min-max normalization is relative to the
            # candidate set -- the best candidate is pinned at 1.0 however weak
            # it actually is -- so a signal with a poor score distribution can
            # still dominate. RRF is scale-free and immune to that.
            k = config.RRF_K
            d_order = np.argsort(-d_arr)
            b_order = np.argsort(-b_arr)
            d_rank = np.empty(len(cand_ids), dtype="float32")
            b_rank = np.empty(len(cand_ids), dtype="float32")
            d_rank[d_order] = np.arange(len(cand_ids))
            b_rank[b_order] = np.arange(len(cand_ids))
            fused = (alpha / (k + d_rank + 1)) + ((1 - alpha) / (k + b_rank + 1))
        else:
            fused = alpha * _minmax(d_arr) + (1 - alpha) * _minmax(b_arr)

        results = []
        for cid, f, ds, bs in zip(cand_ids, fused, d_arr, b_arr):
            c = chunks[cid]
            if metadata_filter and not metadata_filter(c.model_dump()):
                continue
            results.append(RetrievedChunk(chunk=c, score=float(f),
                                          dense_score=float(ds), bm25_score=float(bs)))
        results.sort(key=lambda r: r.score, reverse=True)
        if timing is not None:
            timing.retrieve_ms = (time.perf_counter() - t) * 1000

        # --- optional rerank ---
        if use_rerank and results:
            t = time.perf_counter()
            pool = results[:min(len(results), candidates)]
            pairs = [(query, r.chunk.text) for r in pool]
            rr = _get_reranker().predict(pairs)
            for r, s in zip(pool, rr):
                r.rerank_score = float(s)
                r.score = float(s)
            pool.sort(key=lambda r: r.score, reverse=True)
            results = pool
            if timing is not None:
                timing.rerank_ms = (time.perf_counter() - t) * 1000

        return results[:top_k]
