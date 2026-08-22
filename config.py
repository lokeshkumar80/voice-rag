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

def data_files() -> dict:
    prefix = LANG_FILE[LANG]
    base = f"hf://datasets/{DATASET_ID}"
    return {"train": f"{base}/train/{prefix}train.parquet",
            "validation": f"{base}/validation/{prefix}val.parquet"}

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
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "semantic")  # fixed | sentence | recursive | semantic
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
# 0.9 is the measured optimum, not a guess: `python eval.py --rows 500 --sweep`
# peaks here on 5 of 6 IR metrics. Dense carries this corpus (alpha=0.0 scores
# MRR 0.323 vs 0.510 at 0.9); BM25 contributes a small but real lift on top.
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.9"))  # weight on dense vs BM25 (0..1)
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
# cosine is absolute and comparable across queries. Calibrated in
# scripts/calibrate_guardrail.py against on-topic vs off-topic queries.
MIN_DENSE_SCORE = float(os.getenv("MIN_DENSE_SCORE", "0.50"))
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
