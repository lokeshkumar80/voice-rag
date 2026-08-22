---
title: Voice RAG (Hindi) — MSMARCO-XI
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Speak a Hindi question, get a grounded answer from MS MARCO-XI
---

# Voice-Enabled RAG · Hindi · MSMARCO-XI

Hold the mic button and ask a question in Hindi. The pipeline is:

```
audio ─▶ Sarvam STT ─▶ [input guardrail] ─▶ BGE-M3 embed + FAISS/BM25 hybrid
      ─▶ [off-topic guardrail] ─▶ grounded answer ─▶ [grounding check]
```

Every stage is timed and the per-stage breakdown is returned with each answer.

## ⚠️ About the latency you'll see here
This Space runs on the **free CPU tier**. The published benchmark — retrieval
**P50 15.3 ms / P100 90.7 ms** — was measured on an **RTX 4060**. BGE-M3 is a
568M-parameter model; on 2 shared vCPUs the embedding step is far slower, so
expect the retrieval segment here to land in the hundreds of milliseconds.

That gap is the point, not an excuse: the same code, index and model on GPU vs
CPU is a clean demonstration of where the time actually goes. The
speech-to-text hop dominates either way — it is a network call to Sarvam at
~1–2.5 s, roughly **100x** the local retrieval segment on GPU.

## Try it
- **Hindi questions about the indexed corpus** work best — it is a 500-row slice
  of the MS MARCO-XI validation split (~7.7k chunks), not general knowledge.
- **Ask something off-topic** (or nonsense) and it should *abstain* rather than
  invent an answer. That is the guardrail doing its job.
- There is a text box too, if you'd rather not use the mic. Mic capture needs
  microphone permission.

## What's measured, not assumed
| | value |
|---|---|
| Retrieval P50 / P100 (RTX 4060) | 15.3 ms / 90.7 ms |
| End-to-end voice, warm | 1.2–2.6 s (STT dominates) |
| MRR@10 (hybrid + rerank) | 0.578 |
| Recall@5 (hybrid + rerank) | 0.678 |
| Off-topic abstention | 0.88 balanced accuracy |

Full methodology, ablations and the reasoning behind each default are in the
source repo's README.
