---
title: Voice RAG Hindi
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app_gradio.py
python_version: "3.12.12"
suggested_hardware: zero-a10g
pinned: false
license: mit
short_description: Ask in Hindi by voice; answers grounded in MS MARCO-XI
---

# 🎙️ Voice RAG — Hindi · MS MARCO-XI

Ask a question in Hindi by voice or text. The pipeline transcribes it (Sarvam
STT), retrieves from a chunked MS MARCO-XI corpus (BGE-M3 dense + FAISS HNSW),
and answers **only** from what it retrieved.

```
audio ─▶ STT ─▶ [input guardrail] ─▶ embed + retrieve ─▶ [off-topic guardrail]
      ─▶ generate ─▶ [grounding check] ─▶ answer      (+ per-stage timing)
```

## Try making it refuse
Ask something the corpus can't answer — *"What is the capital of Mars?"*, or
gibberish. It should **abstain** rather than invent an answer. That behaviour is
measured, not decorative: ungrounded answers on genuinely unanswerable questions
drop from **100% → 14.7%** with the cross-encoder gate enabled.

The corpus is a 103,068-chunk slice of MS MARCO-XI, not general knowledge, so
plenty of reasonable questions will be refused too. Both directions are the point.

## What the timings show
| stage | typical |
|---|---|
| speech-to-text (network call to Sarvam) | 1,000–2,500 ms |
| **retrieval (embed + search)** | **~11 ms** |
| generation (extractive, no LLM) | ~30 ms |

The STT hop is roughly **100x** the entire local retrieval segment. That is the
honest framing of the project's "<200 ms" target: it applies to the query-time
retrieval path, which is local and genuinely fast. No pipeline built on a hosted
STT call reaches 200 ms end to end, and this one reports both numbers rather than
quoting the flattering one.

## Measured, not assumed
Every default here came out of an ablation, including the ones that contradicted
the obvious choice:

| result | |
|---|---|
| MRR@10 / Recall@5 | 0.551 / 0.744 (1,219 gold-labelled queries) |
| Retrieval P50 / P100 | 11.4 ms / 57 ms (RTX 4060) |
| Ungrounded answers, guardrails on | 100% → 14.7% |
| Chunking | plain sentence splitting **beat** semantic chunking |
| Fusion | dense-only **beat** hybrid BM25 at this corpus size |

Full methodology, ablations and the reasoning behind each default are in the
source repository.

> Requires a `SARVAM_API_KEY` Space secret for the voice path; the text box works
> without it.
