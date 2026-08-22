"""
Builds and persists the retrieval index:
  - dense: sentence-transformer embeddings in a FAISS inner-product index
  - lexical: BM25 over the same chunks
Both are saved to INDEX_DIR so the server loads them instantly at startup.
"""
from __future__ import annotations
import json
import os
import pickle
from functools import lru_cache
from typing import List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

import config
from src.schemas import Chunk


@lru_cache(maxsize=2)
def get_embedder(model_name: str = config.EMBED_MODEL):
    """Cached so we load the model once per process. Imported lazily so the
    server (and tests) don't pull in torch until embeddings are actually used.
    Uses GPU when config.DEVICE='cuda' and a GPU is present."""
    from sentence_transformers import SentenceTransformer
    device = config.DEVICE
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    model = SentenceTransformer(model_name, device=device)
    # Cap sequence length to what a chunk actually needs (see config.EMBED_MAX_SEQ).
    model.max_seq_length = min(model.max_seq_length, config.EMBED_MAX_SEQ)
    return model


def embed(texts: List[str], normalize: bool = True,
          show_progress: bool = False) -> np.ndarray:
    """Encode texts, backing off the batch size if the GPU runs out of memory
    rather than losing a long ingest run to a single OOM."""
    model = get_embedder()
    batch = config.EMBED_BATCH
    while True:
        try:
            vecs = model.encode(texts, batch_size=batch,
                                convert_to_numpy=True, normalize_embeddings=normalize,
                                show_progress_bar=show_progress)
            return vecs.astype("float32")
        except Exception as e:
            if "out of memory" not in str(e).lower() or batch <= 1:
                raise
            import torch
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            print(f"  [embed] CUDA OOM -> retrying at batch_size={batch}")


def _tokenize(text: str) -> List[str]:
    # simple unicode-aware tokenizer, works for Latin + Indic scripts
    return [t for t in text.lower().replace("।", " ").split() if t]


class HybridIndex:
    def __init__(self, chunks: List[Chunk], faiss_index: faiss.Index, bm25: BM25Okapi):
        self.chunks = chunks
        self.faiss = faiss_index
        self.bm25 = bm25

    # ---- build ----
    @classmethod
    def build(cls, chunks: List[Chunk]) -> "HybridIndex":
        texts = [c.text for c in chunks]
        vecs = embed(texts, show_progress=True)
        dim = vecs.shape[1]
        if config.INDEX_TYPE == "hnsw":
            # Approximate NN graph: scales to millions of vectors with a small,
            # tunable recall/latency tradeoff (HNSW_M, HNSW_EF_SEARCH).
            index = faiss.IndexHNSWFlat(dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = config.HNSW_EF_SEARCH
        else:
            index = faiss.IndexFlatIP(dim)   # exact cosine (vectors normalized)
        index.add(vecs)
        bm25 = BM25Okapi([_tokenize(t) for t in texts])
        return cls(chunks, index, bm25)

    # ---- persist ----
    def save(self, dir_: str = config.INDEX_DIR) -> None:
        os.makedirs(dir_, exist_ok=True)
        faiss.write_index(self.faiss, os.path.join(dir_, "dense.faiss"))
        with open(os.path.join(dir_, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)
        with open(os.path.join(dir_, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        with open(os.path.join(dir_, "meta.json"), "w") as f:
            json.dump({"n_chunks": len(self.chunks),
                       "embed_model": config.EMBED_MODEL,
                       "chunk_strategy": config.CHUNK_STRATEGY}, f, indent=2)

    @classmethod
    def load(cls, dir_: str = config.INDEX_DIR) -> "HybridIndex":
        faiss_index = faiss.read_index(os.path.join(dir_, "dense.faiss"))
        with open(os.path.join(dir_, "bm25.pkl"), "rb") as f:
            bm25 = pickle.load(f)
        chunks = []
        with open(os.path.join(dir_, "chunks.jsonl"), encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk.model_validate_json(line))
        return cls(chunks, faiss_index, bm25)

    @staticmethod
    def exists(dir_: str = config.INDEX_DIR) -> bool:
        return os.path.exists(os.path.join(dir_, "dense.faiss"))
