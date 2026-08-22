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

from tenacity import retry, stop_after_attempt, wait_exponential

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


@retry(reraise=True, stop=stop_after_attempt(config.STAGE_RETRIES + 1),
       wait=wait_exponential(multiplier=0.3, max=3))
def _sarvam_chat(query: str, contexts: List[RetrievedChunk]) -> str:
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
    system = ("You are a retrieval-grounded assistant. Answer ONLY using the "
              "provided context. If the answer is not in the context, reply "
              "exactly: I don't know. Be concise. Answer in the user's language.")
    user = f"Context:\n{_context_block(contexts)}\n\nQuestion: {query}\nAnswer:"
    resp = client.chat.completions(
        model=config.SARVAM_CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
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
