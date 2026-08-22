"""
Central configuration. Everything tunable lives here so the rest of the code
stays declarative. Values can be overridden with environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Dataset (ai4bharat/MSMARCO-XI)
# ---------------------------------------------------------------------------
# The dataset is ~55 GB / 11.5M rows. We STREAM a subset and cap it.
DATASET_ID = "ai4bharat/MSMARCO-XI"
LANG = os.getenv("LANG_CODE", "hi")           # hi, en(->English_passages), te, ta, bn, ...
SPLIT = os.getenv("SPLIT", "validation")      # "train" or "validation"
MAX_ROWS = int(os.getenv("MAX_ROWS", "4000")) # how many source rows to pull (each has ~10 passages)

LANG_FILE = {
    "as": "asm", "bn": "ben", "gu": "gu",  "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "or",  "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}

# Where `scripts/fetch_dataset.py` parks a local copy of the parquet. Streaming
# straight from hf:// re-downloads on every run and, with no read timeout, a
# dropped connection hangs the process indefinitely (it sits in CLOSE-WAIT
# forever). Ablations re-read the same split many times, so cache it once.
DATA_CACHE_DIR = os.getenv("DATA_CACHE_DIR", "data/raw")


def data_files(prefer_local: bool = True) -> dict:
    """Parquet paths per split -- a local cached copy when present, else hf://."""
    prefix = LANG_FILE[LANG]
    base = f"hf://datasets/{DATASET_ID}"
    remote = {"train": f"{base}/train/{prefix}train.parquet",
              "validation": f"{base}/validation/{prefix}val.parquet"}
    if not prefer_local:
        return remote
    out = {}
    for split, url in remote.items():
        local = os.path.join(DATA_CACHE_DIR, os.path.basename(url))
        out[split] = local if os.path.exists(local) else url
    return out

# Which passage text to index. For an Indic demo use "Translated_passages"
# (matches Sarvam Indic STT). For English use "English_passages".
PASSAGE_FIELD = os.getenv("PASSAGE_FIELD", "Translated_passages")

# ---------------------------------------------------------------------------
# Embeddings + index
# ---------------------------------------------------------------------------
# BGE-M3: multilingual (100+ langs incl. Indic), 568M params (comfortable on an
# 8GB GPU), and the current open-weight leader for multilingual hybrid retrieval.
# It natively produces dense + sparse (lexical) signals from ONE model.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("DEVICE", "cuda")            # "cuda" on your lab box, "cpu" to fall back
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))  # lower for BGE-M3 on 8GB
# BGE-M3 ships with max_seq_length=8192. Batches are padded to their longest
# member, so leaving it at 8192 reserves ~30x more sequence room than a
# CHUNK_SIZE-char chunk can use and OOMs an 8GB card. Cap it to fit chunks.
EMBED_MAX_SEQ = int(os.getenv("EMBED_MAX_SEQ", "512"))  # tokens, not chars
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
# FAISS index type: "flat" (exact, fine <~200k vectors) or "hnsw" (ANN, scales to
# millions with tiny latency cost). Learn the tradeoff by benchmarking both.
INDEX_TYPE = os.getenv("INDEX_TYPE", "hnsw")
HNSW_M = int(os.getenv("HNSW_M", "32"))         # graph degree; higher = better recall, more RAM
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "128"))  # search breadth; recall/latency knob

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Strategy applied at ingest time. See src/chunking.py.
# Measured, not assumed: `for s in fixed sentence recursive semantic; do
#   python eval.py --rows 500 --chunk $s --answer-f1; done`
# "semantic" -- the sophisticated-sounding option -- came LAST on every
# denominator-free metric (MRR@10 0.578 vs 0.606, Hit@5 0.839 vs 0.871) while
# also being ~10x slower to build, since it embeds every sentence. "sentence"
# ties "fixed" on the headline metrics, wins nDCG@10, yields the smallest index
# (5,592 chunks), and never cuts mid-sentence -- which matters because the
# extractive generator returns whole sentences.
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "sentence")  # fixed | sentence | recursive | semantic
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))          # target chars (fixed/recursive)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))     # char overlap
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.55"))  # cosine break point
# Fragments below this are merged into their neighbour. BM25 length
# normalization rewards very short documents, so a stray "3." outranks real
# passages and becomes the top-1 the answer is generated from.
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "80"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))            # final chunks handed to generator
CANDIDATES = int(os.getenv("CANDIDATES", "30")) # candidates pulled before rerank/fusion
# 1.0 = pure dense. This is a measured negative result, not an oversight.
#
# At 5,442 chunks BM25 added a small real lift and 0.9 was optimal. At 21,657
# chunks BM25 weakens relatively (MRR@10 0.259 vs dense 0.494) and alpha=1.0 wins
# on 5 of 6 metrics -- mixing in the weaker signal only costs accuracy.
#
# Before accepting that, we tested the obvious alternative explanation: that
# min-max score fusion was the problem, not BM25. Reciprocal Rank Fusion (the
# standard fix) helped mid-range (alpha=0.6: MRR 0.425 vs 0.401) but still peaked
# at 1.0. So the fusion was mildly suboptimal AND BM25 genuinely does not help
# here. Both signals were measured; dense carries this corpus.
#
# Keep the hybrid path: it is a config change away, and the answer is corpus
# dependent -- it already flipped once between 5k and 21k chunks.
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "1.0"))  # weight on dense vs BM25 (0..1)
# How the two signals are combined.
#   "minmax" -- normalize each score list to [0,1] over the candidate set, then
#               weight by alpha. Simple, but the normalization is relative to
#               whatever candidates showed up, so the top of each list is pinned
#               at 1.0 regardless of absolute quality.
#   "rrf"    -- Reciprocal Rank Fusion: combine RANKS, not scores
#               (sum of 1/(k+rank)). Scale-free and the standard choice, because
#               it cannot be skewed by one signal's score distribution.
FUSION = os.getenv("FUSION", "rrf")             # minmax | rrf (rrf >= minmax measured)
RRF_K = int(os.getenv("RRF_K", "60"))           # standard smoothing constant
USE_RERANK = os.getenv("USE_RERANK", "false").lower() == "true"  # cross-encoder; off for 200ms path
# bge-reranker-v2-m3 is the multilingual cross-encoder that pairs with BGE-M3.
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
# Off-topic abstain threshold, applied to the top result's DENSE COSINE.
# It cannot be applied to the fused score: fusion min-max normalizes each signal
# over the candidate set, which pins the best candidate at ~1.0 for every query
# -- pure gibberish scores 1.0000 -- so a fused threshold can never fire. The
# cosine is absolute and comparable across queries.
#
# 0.58 comes from calibrating against HARD negatives: real MS MARCO queries whose
# passages were never indexed. An earlier 0.50 was calibrated against gibberish
# ("asdf qwerty"), which scores at most 0.434 here -- so it scored 0.93 balanced
# accuracy and then blocked only 26% of realistic unanswerable queries. Easy
# negatives give you a number, not a guarantee. Re-derive with
# scripts/calibrate_guardrail.py and always cross-check scripts/faithfulness.py.
MIN_DENSE_SCORE = float(os.getenv("MIN_DENSE_SCORE", "0.58"))

# When the cross-encoder has already run (USE_RERANK=true) its score is a far
# better abstention signal than cosine and costs nothing extra -- the work is
# already done. Measured at 103k chunks, 150+150 queries:
#   bi-encoder cosine : answerable 0.669 vs unanswerable 0.590 (gap 0.079)
#   cross-encoder     : answerable 0.860 vs unanswerable 0.350 (gap 0.510)
# a 6.5x wider separation -> balanced accuracy 0.710 vs 0.813.
# A cascade (cosine first, cross-encoder only when ambiguous) was tried and
# rejected: the distributions overlap so heavily that 71% of queries land in the
# ambiguous band, so it costs ~236ms to buy *less* accuracy than always running
# the cross-encoder. There is no cheap version of this.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.85"))
MIN_TRANSCRIPT_CHARS = int(os.getenv("MIN_TRANSCRIPT_CHARS", "3"))
# Grounding check: fraction of answer content words that must appear in context.
MIN_GROUNDING_OVERLAP = float(os.getenv("MIN_GROUNDING_OVERLAP", "0.35"))

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
# "extractive" = no LLM, returns grounded span (fast, always grounded).
# "llm"        = Sarvam chat completion, grounded by retrieved context.
GENERATION_MODE = os.getenv("GENERATION_MODE", "extractive")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-m")
SARVAM_LANG = os.getenv("SARVAM_LANG", "hi-IN")  # STT hint; "unknown" to auto-detect

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
STAGE_RETRIES = int(os.getenv("STAGE_RETRIES", "2"))
STAGE_TIMEOUT_S = float(os.getenv("STAGE_TIMEOUT_S", "15"))
