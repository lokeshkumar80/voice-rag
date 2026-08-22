"""
Ingest MSMARCO-XI -> chunked corpus -> hybrid index on disk.

The dataset is ~55 GB / 11.5M rows, so we STREAM and cap at MAX_ROWS. Each row's
`passages` is a dict of parallel lists (English_passages / Translated_passages /
is_selected). We flatten those into a passage corpus, chunk each passage with the
configured strategy, and attach metadata for metadata-aware retrieval.

Run:  python ingest.py
"""
from __future__ import annotations
import sys

from datasets import load_dataset

import config
from src.chunking import chunk_passage
from src.indexer import HybridIndex, embed
from src.schemas import Chunk


def _passage_texts(row: dict) -> list[tuple[str, int]]:
    """Return [(passage_text, is_selected), ...] for one row."""
    passages = row.get("passages") or {}
    texts = passages.get(config.PASSAGE_FIELD) or passages.get("English_passages") or []
    selected = passages.get("is_selected") or [0] * len(texts)
    out = []
    for i, txt in enumerate(texts):
        if txt and txt.strip():
            out.append((txt.strip(), int(selected[i]) if i < len(selected) else 0))
    return out


def build_corpus() -> list[Chunk]:
    print(f"Streaming {config.DATASET_ID} [{config.LANG}] split={config.SPLIT} "
          f"(cap {config.MAX_ROWS} rows) ...")
    ds = load_dataset("parquet", data_files=config.data_files(),
                  split=config.SPLIT, streaming=True)   # eval.py uses split="validation"

    # semantic chunking needs an embed function
    embed_fn = (lambda xs: embed(xs)) if config.CHUNK_STRATEGY == "semantic" else None

    chunks: list[Chunk] = []
    seen: set[str] = set()
    cid = 0
    for r_i, row in enumerate(ds):
        if r_i >= config.MAX_ROWS:
            break
        qtype = row.get("query_type")
        qid = row.get("query_id")
        for passage, is_sel in _passage_texts(row):
            for piece in chunk_passage(passage, config.CHUNK_STRATEGY,
                                       size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP,
                                       embed_fn=embed_fn,
                                       semantic_threshold=config.SEMANTIC_THRESHOLD,
                                       min_chars=config.MIN_CHUNK_CHARS):
                key = piece[:120]
                if key in seen:            # cheap dedup of near-identical chunks
                    continue
                seen.add(key)
                chunks.append(Chunk(id=cid, text=piece, query_id=qid,
                                    query_type=qtype, is_selected=is_sel, source_row=r_i))
                cid += 1
        if (r_i + 1) % 500 == 0:
            print(f"  processed {r_i + 1} rows -> {len(chunks)} chunks")
    print(f"Built {len(chunks)} chunks from {min(r_i + 1, config.MAX_ROWS)} rows.")
    return chunks


def main():
    chunks = build_corpus()
    if not chunks:
        print("No chunks produced. Check LANG / PASSAGE_FIELD / network access.")
        sys.exit(1)
    print(f"Embedding + indexing with {config.EMBED_MODEL} ...")
    index = HybridIndex.build(chunks)
    index.save()
    print(f"Saved index to {config.INDEX_DIR}/  ({len(chunks)} chunks). Ready.")


if __name__ == "__main__":
    main()
