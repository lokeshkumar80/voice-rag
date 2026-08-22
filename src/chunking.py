"""
Multiple chunking strategies (requirement #2: "chunking strategy should be vast").

Each function takes a raw passage string and returns a list of chunk strings.
The ingest pipeline attaches metadata (query_id, query_type, is_selected...) to
every produced chunk, so retrieval is metadata-aware regardless of strategy.

Strategies
----------
fixed      : fixed-size character windows with overlap
sentence   : one chunk per sentence, greedily packed to a target size
recursive  : split on a hierarchy of separators (para -> sentence -> word)
             so we never cut mid-word, honoring size + overlap
semantic   : embedding-based — start a new chunk when adjacent sentences'
             similarity drops below SEMANTIC_THRESHOLD (topic shift)
"""
from __future__ import annotations
import re
from typing import Callable, List, Optional

import numpy as np

# Sentence splitter that tolerates Latin + Devanagari/Indic danda (।) punctuation.
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+|\n+")


def _sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def fixed_chunks(text: str, size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def sentence_chunks(text: str, size: int, overlap_sents: int = 1) -> List[str]:
    sents = _sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur and cur_len + len(s) > size:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []      # sentence overlap
            cur_len = sum(len(x) for x in cur)
        cur.append(s)
        cur_len += len(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def recursive_chunks(text: str, size: int, overlap: int,
                     seps: Optional[List[str]] = None) -> List[str]:
    """Recursively split on the coarsest separator that keeps pieces <= size."""
    seps = seps if seps is not None else ["\n\n", "\n", "। ", ". ", " "]
    text = text.strip()
    if len(text) <= size or not seps:
        return [text] if text else []

    sep, rest = seps[0], seps[1:]
    pieces = text.split(sep) if sep else list(text)

    out, buf = [], ""
    for p in pieces:
        candidate = (buf + sep + p) if buf else p
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            # piece itself too big -> recurse with finer separators
            out.extend(recursive_chunks(p, size, overlap, rest) if len(p) > size else [p])
            buf = ""
    if buf:
        out.append(buf)

    # add character overlap between neighbours to preserve context at edges
    if overlap and len(out) > 1:
        stitched = [out[0]]
        for prev, nxt in zip(out, out[1:]):
            tail = prev[-overlap:]
            stitched.append((tail + " " + nxt).strip())
        out = stitched
    return [c for c in out if c.strip()]


def semantic_chunks(text: str, embed_fn: Callable[[List[str]], np.ndarray],
                    threshold: float, max_size: int) -> List[str]:
    """
    Split at topic boundaries: embed sentences, start a new chunk when the
    cosine similarity between consecutive sentences falls below `threshold`
    (or the running chunk would exceed max_size).
    """
    sents = _sentences(text)
    if len(sents) <= 1:
        return sents
    vecs = embed_fn(sents)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    chunks, cur = [], [sents[0]]
    cur_len = len(sents[0])
    for i in range(1, len(sents)):
        sim = float(np.dot(vecs[i], vecs[i - 1]))
        if sim < threshold or cur_len + len(sents[i]) > max_size:
            chunks.append(" ".join(cur))
            cur, cur_len = [sents[i]], len(sents[i])
        else:
            cur.append(sents[i])
            cur_len += len(sents[i])
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def merge_small(chunks: List[str], min_chars: int) -> List[str]:
    """Fold sub-`min_chars` fragments into their neighbour.

    Every strategy can emit degenerate fragments -- a stray "3.", a trailing
    "क्या है?" -- and they are actively harmful in retrieval: BM25 length
    normalization *rewards* very short documents, so a two-word fragment
    outranks the real passage and becomes the top-1 the answer is generated
    from. We merge rather than drop so no source text is lost; the merge can
    push a chunk slightly over `size`, which is the cheaper trade.
    """
    if min_chars <= 0 or not chunks:
        return chunks
    out: List[str] = []
    for c in chunks:
        if out and len(c) < min_chars:
            out[-1] = f"{out[-1]} {c}".strip()
        else:
            out.append(c)
    # the first chunk can still be short if it had no predecessor to merge into
    if len(out) > 1 and len(out[0]) < min_chars:
        out[1] = f"{out[0]} {out[1]}".strip()
        out.pop(0)
    return [c for c in out if c.strip()]


def chunk_passage(text: str, strategy: str, *, size: int, overlap: int,
                  embed_fn: Optional[Callable] = None,
                  semantic_threshold: float = 0.55,
                  min_chars: int = 0) -> List[str]:
    """Dispatch to the configured strategy, then clean up micro-chunks."""
    if strategy == "fixed":
        chunks = fixed_chunks(text, size, overlap)
    elif strategy == "sentence":
        chunks = sentence_chunks(text, size)
    elif strategy == "recursive":
        chunks = recursive_chunks(text, size, overlap)
    elif strategy == "semantic":
        if embed_fn is None:
            chunks = recursive_chunks(text, size, overlap)   # graceful fallback
        else:
            chunks = semantic_chunks(text, embed_fn, semantic_threshold, size)
    else:
        raise ValueError(f"unknown chunk strategy: {strategy}")
    return merge_small(chunks, min_chars)
