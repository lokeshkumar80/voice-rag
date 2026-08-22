"""
Answer generation, grounded in retrieved context.

Two modes (config.GENERATION_MODE):
  - "extractive": pick the best sentence(s) from the top chunk. No LLM, ~0 ms,
    guaranteed grounded. This is what keeps the query-time path fast.
  - "llm": Sarvam chat completion, instructed to answer ONLY from context and
    say it doesn't know otherwise. Higher quality, adds network latency.

LLM mode degrades gracefully to extractive if the API fails (harness contract).
"""
from __future__ import annotations
import re
from typing import List

from tenacity import (retry, retry_if_exception, stop_after_attempt,
                      wait_exponential)

import config
from src.indexer import embed
from src.schemas import RetrievedChunk

_SENT = re.compile(r"(?<=[.!?।॥])\s+")


def _best_sentences(query: str, text: str, k: int = 2) -> str:
    sents = [s.strip() for s in _SENT.split(text) if s.strip()] or [text]
    if len(sents) <= k:
        return " ".join(sents)
    qv = embed([query])[0]
    sv = embed(sents)
    scores = sv @ qv
    top = sorted(range(len(sents)), key=lambda i: -scores[i])[:k]
    return " ".join(sents[i] for i in sorted(top))   # keep original order


def extractive(query: str, contexts: List[RetrievedChunk]) -> str:
    if not contexts:
        return ""
    return _best_sentences(query, contexts[0].chunk.text, k=2)


def _context_block(contexts: List[RetrievedChunk], limit: int = 4) -> str:
    return "\n".join(f"[{i+1}] {c.chunk.text}" for i, c in enumerate(contexts[:limit]))


def _is_transient(exc: BaseException) -> bool:
    """Retry blips, not timeouts.

    A timeout means the model is slow, and it will still be slow on attempt
    three -- retrying just multiplies the wait. Measured: STAGE_TIMEOUT_S=15 with
    3 attempts took 46s to fall back to extractive, which is indistinguishable
    from a hang. Timeouts should fail fast into graceful degradation; connection
    resets and 5xx are worth another go.
    """
    name = type(exc).__name__.lower()
    return "timeout" not in name and "deadline" not in name


@retry(reraise=True, stop=stop_after_attempt(config.STAGE_RETRIES + 1),
       wait=wait_exponential(multiplier=0.3, max=3),
       retry=retry_if_exception(_is_transient))
def _sarvam_chat(query: str, contexts: List[RetrievedChunk]) -> str:
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
    system = ("You are a retrieval-grounded assistant. Answer ONLY using the "
              "provided context. If the answer is not in the context, reply "
              "exactly: I don't know. Be concise. Answer in the user's language.")
    user = f"Context:\n{_context_block(contexts)}\n\nQuestion: {query}\nAnswer:"
    # Enforce STAGE_TIMEOUT_S. It was declared in config and wired to nothing,
    # so a hosted call could block for as long as it liked -- measured at 79s
    # against sarvam-105b. A timeout the harness advertises but does not apply is
    # worse than none: it buys false confidence.
    resp = client.chat.completions(
        model=config.SARVAM_CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        request_options={"timeout_in_seconds": int(config.STAGE_TIMEOUT_S)},
    )
    # SDK returns an OpenAI-style object
    return resp.choices[0].message.content.strip()


def generate(query: str, contexts: List[RetrievedChunk], mode: str = None) -> str:
    mode = mode or config.GENERATION_MODE
    if mode == "llm" and config.SARVAM_API_KEY:
        try:
            out = _sarvam_chat(query, contexts)
            if out and out.lower() != "i don't know":
                return out
            return out or extractive(query, contexts)
        except Exception:
            return extractive(query, contexts)   # graceful degradation
    return extractive(query, contexts)
