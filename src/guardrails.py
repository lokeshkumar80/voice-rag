"""
Guardrails (requirement #6). Two gates:

INPUT gate  (before retrieval):
  - empty / too-short transcript  -> reject
  - unsafe/inappropriate content  -> refuse
OUTPUT gate (after retrieval + generation):
  - off-topic: best dense cosine below MIN_DENSE_SCORE -> abstain
  - grounding: generated answer must overlap the retrieved context; if not,
    the answer is likely hallucinated -> abstain

"Show that your system knows when NOT to answer, not just how to answer."
"""
from __future__ import annotations
import re
from typing import List, Tuple

import config
from src.schemas import RetrievedChunk

# Minimal, transparent unsafe-content list. In production swap for a classifier
# (e.g. Sarvam chat with a safety prompt, or Llama-Guard). Kept lightweight here
# so it costs ~0 ms and never blocks the latency budget.
_UNSAFE_PATTERNS = [
    r"\bhow to (make|build|synthesi[sz]e)\b.*\b(bomb|explosive|meth|nerve agent)\b",
    r"\b(child|minor)\b.*\b(sexual|explicit)\b",
    r"\bkill (myself|yourself)\b|\bself[- ]harm\b",
    r"\b(credit card|social security)\b.*\b(steal|dump|generate)\b",
]
_UNSAFE_RE = [re.compile(p, re.I) for p in _UNSAFE_PATTERNS]

_STOP = set("the a an of to in on for and or is are was were be been with by at from "
            "क्या है की का के में और या को से पर हैं था थे यह वह".split())


def _content_words(text: str) -> set[str]:
    toks = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return {t for t in toks if t not in _STOP and len(t) > 1}


# ---------------- INPUT gate ----------------
def check_input(transcript: str) -> Tuple[bool, str]:
    """Return (ok, reason). ok=False means refuse before doing any work."""
    t = (transcript or "").strip()
    if len(t) < config.MIN_TRANSCRIPT_CHARS:
        return False, "empty_or_too_short"
    for rx in _UNSAFE_RE:
        if rx.search(t):
            return False, "unsafe_content"
    return True, ""


# ---------------- OUTPUT gate ----------------
def check_retrieval(contexts: List[RetrievedChunk]) -> Tuple[bool, str]:
    """Off-topic detection: is anything relevant enough to answer from?

    Thresholds the DENSE COSINE, not the fused score. Fusion min-max normalizes
    each signal across the candidate set, so the best candidate is pinned near
    1.0 for every query -- gibberish included -- and a fused threshold can never
    fire. The cosine is an absolute similarity, comparable across queries.
    """
    if not contexts:
        return False, "no_context"

    # If the cross-encoder already ran, prefer its score: it does full
    # query-document attention instead of comparing independent embeddings, and
    # separates answerable from unanswerable ~6.5x more widely than cosine
    # (gap 0.510 vs 0.079 at 103k chunks). Free here -- the work is already done.
    rr = [c.rerank_score for c in contexts if c.rerank_score is not None]
    if rr:
        best = max(rr)
        if best < config.MIN_RERANK_SCORE:
            return False, f"off_topic_rerank_{best:.3f}"
        return True, ""

    best = max(c.dense_score for c in contexts)
    if best < config.MIN_DENSE_SCORE:
        return False, f"off_topic_low_score_{best:.3f}"
    return True, ""


def check_grounding(answer: str, contexts: List[RetrievedChunk]) -> Tuple[bool, str]:
    """
    Cheap hallucination check: what fraction of the answer's content words
    actually appear in the retrieved context? Below threshold => not grounded.
    (Extractive answers score ~1.0 by construction; this mainly guards LLM mode.)
    """
    ans_words = _content_words(answer)
    if not ans_words:
        return False, "empty_answer"
    ctx_words = set()
    for c in contexts:
        ctx_words |= _content_words(c.chunk.text)
    overlap = len(ans_words & ctx_words) / len(ans_words)
    if overlap < config.MIN_GROUNDING_OVERLAP:
        return False, f"ungrounded_overlap_{overlap:.2f}"
    return True, ""


ABSTAIN_MESSAGE = {
    "en": "I don't have enough information in my knowledge base to answer that.",
    "hi": "मेरे पास इसका उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
}


def abstain_text(lang: str) -> str:
    return ABSTAIN_MESSAGE.get(lang.split("-")[0], ABSTAIN_MESSAGE["en"])
