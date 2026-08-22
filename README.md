# Voice-Enabled RAG · MSMARCO-XI

Voice in → Sarvam STT → hybrid chunk retrieval (FAISS + BM25) → grounded answer,
run inside a timed, guardrailed harness with P50/P70/P100 latency analytics.

```
audio ─▶ STT ─▶ [input guardrail] ─▶ embed+retrieve ─▶ [off-topic guardrail]
      ─▶ generate ─▶ [grounding guardrail] ─▶ answer   (+ per-stage timing)
```

## How each task requirement is met
| Requirement | Where |
|---|---|
| 1. Speech-to-text (Sarvam) | `src/stt.py` (Saaras v3, retries) |
| 2. Vast chunking | `src/chunking.py` (fixed / sentence / recursive / semantic + overlap + metadata) |
| 3. <200 ms latency | local embed + FAISS; retrieval budget measured separately from STT/LLM |
| 4. P50/P70/P100 analytics | `benchmark.py` |
| 5. Harness | `src/harness.py` (staged orchestration, retries, structured Pydantic I/O, graceful degradation) |
| 6. Guardrails | `src/guardrails.py` (input safety, off-topic abstain, hallucination/grounding check) |

## About the 200 ms target — read this
The full voice path can't hit 200 ms: STT is a network call (~300 ms–1 s) and any
hosted LLM adds hundreds of ms. So the budget is applied to the **query-time
retrieval segment** — `embed + vector search + rerank` — which runs locally and
stays well under 200 ms. The harness times every stage separately, and
`benchmark.py` prints the retrieval P100 against the 200 ms line **and** the
end-to-end numbers, so the write-up is transparent about exactly what is being
measured. To keep the whole non-STT path fast, generation defaults to an
**extractive** mode (best grounded sentence, no LLM). Flip `GENERATION_MODE=llm`
for Sarvam-generated answers and report that latency separately.

---

## Step-by-step

### 0. Prerequisites
- Python 3.10+ on Linux
- A Sarvam API key → https://dashboard.sarvam.ai

### 1. Clone / open in VS Code, create a venv
```bash
cd voice-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> First install pulls torch (for `sentence-transformers`); it's a few hundred MB.

### 2. Configure
```bash
cp .env.example .env       # then paste your SARVAM_API_KEY into .env
```
Key knobs live in `config.py` (or set as env vars):
`LANG_CODE` (hi/te/ta/bn…), `PASSAGE_FIELD` (`Translated_passages` for Indic,
`English_passages` for English), `MAX_ROWS`, `CHUNK_STRATEGY`, `USE_RERANK`,
`GENERATION_MODE`, `HYBRID_ALPHA`, `MIN_CHUNK_CHARS`, `MIN_DENSE_SCORE`.

Two naming traps worth knowing:
- The variable is **`LANG_CODE`**, not `LANG`. `LANG` is the system locale
  variable — setting it does nothing here and can break your shell's locale.
- There is **no `en` language code.** English is not a separate file; it lives
  *inside* each language's parquet as the `English_passages` field. For an
  English corpus keep `LANG_CODE=hi` and set `PASSAGE_FIELD=English_passages`.
  (`LANG_CODE=en` raises `KeyError` — it isn't in `config.LANG_FILE`.)

### 3. Build the index (streams a dataset subset, chunks, embeds)
```bash
python ingest.py
```
This streams `MAX_ROWS` rows of MSMARCO-XI (default 4000 → tens of thousands of
chunks), applies the chunking strategy, embeds locally, and writes
`data/index/`. Start small (`MAX_ROWS=1000`) to smoke-test, then scale up.

### 4. Check the latency numbers
```bash
python benchmark.py --n 200                 # retrieval path, extractive
python benchmark.py --n 200 --rerank        # with cross-encoder rerank
python benchmark.py --n 100 --with-generation
```
Prints a P50/P70/P100/mean table per stage and a PASS/OVER verdict vs 200 ms.
The numbers under **Measured results** below came from this.

### 5. Run the app (live link + demo)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 — hold the button to record a question, or type one.
You'll see the transcript, grounded answer, per-stage timing, and the retrieved
context. `POST /ask` (audio) and `POST /ask_text` (JSON) are the raw endpoints.

---

## Measured results
Hindi (`LANG_CODE=hi`), 500 validation rows -> 7,670 chunks, BGE-M3 on an RTX 4060.

### Latency — `python benchmark.py --n 200`
| stage (ms) | P50 | P70 | P100 |
|---|---|---|---|
| embed | 9.71 | 10.02 | 47.32 |
| retrieve | 5.68 | 6.74 | 44.58 |
| **retrieval_total** | **15.34** | **16.57** | **90.74** |

**Retrieval P100 = 90.74 ms vs the 200 ms target — PASS.**
The benchmark warms up before timing; the first query pays model load and CUDA
init (~10 s), and leaving that in the timed loop puts a one-off cold start into
P100 and misrepresents steady-state.

With `--rerank` the cross-encoder adds ~332 ms at P50, so it **breaks** the
budget. That is why `USE_RERANK=false` is the default for the latency path — it
buys accuracy, and the table below shows exactly how much.

### End-to-end voice — `POST /ask` with browser-style WebM/Opus
Measured with a Sarvam TTS→STT round trip (synthesized Hindi question in, grounded answer out),
warm process:

| stage | STT | retrieval_total | generate | **total** |
|---|---|---|---|---|
| ms | 1139–2566 | **13.6** | 32 | **1185–2612** |

This is the honest version of the 200 ms discussion above. The hosted STT call alone is roughly
**100x** the entire local retrieval segment. No pipeline built on a network STT hop reaches
200 ms end-to-end; what *is* under 200 ms — and what the target sensibly applies to — is the
query-time retrieval path, at 13.6 ms warm.

### Retrieval quality — `python eval.py --rows 500 --answer-f1`
248 queries that have a gold passage (`is_selected==1`) in the corpus.

| config | R@1 | R@5 | Hit@5 | MRR@10 | nDCG@10 | Answer F1 |
|---|---|---|---|---|---|---|
| BM25 only | 0.132 | 0.400 | 0.524 | 0.323 | 0.343 | 0.190 |
| Dense only | 0.227 | 0.617 | 0.786 | 0.505 | 0.525 | 0.232 |
| Hybrid a=0.6 | 0.197 | 0.601 | 0.786 | 0.464 | 0.486 | 0.214 |
| **Hybrid + rerank** | **0.285** | **0.678** | **0.839** | **0.578** | **0.574** | **0.248** |

Read `R@1` with care: it divides by the number of gold *chunks*, so when a gold
passage splits into several chunks it is capped well below 1.0 by construction.
`Hit@5` and `MRR@10` are the fair cross-config comparisons.

### Why HYBRID_ALPHA is 0.9 — `python eval.py --rows 500 --sweep`
| alpha | 0.0 | 0.4 | 0.6 | 0.8 | **0.9** | 1.0 |
|---|---|---|---|---|---|---|
| MRR@10 | 0.323 | 0.419 | 0.465 | 0.492 | **0.510** | 0.505 |
| nDCG@10 | 0.343 | 0.442 | 0.488 | 0.512 | **0.529** | 0.526 |

Dense carries this corpus. BM25 adds a small but real lift on top — the peak is
at 0.9, not at the 0.6 that looks like a sensible default. The sweep is what
justifies hybrid here; a single hand-picked alpha would have proved nothing.

## Experiments worth running
- Compare chunking strategies: re-run `ingest.py` with `CHUNK_STRATEGY=fixed` vs
  `semantic`, benchmark each, and show retrieval-quality/latency trade-offs.
- Sweep the fusion weight with `python eval.py --rows 500 --sweep` (see above).
- Show the guardrails firing: ask something off-topic and something unsafe; the
  system abstains instead of hallucinating.

## Deploying the live link
- **Hugging Face Spaces (Docker or Gradio):** simplest for this dataset. Commit the
  repo, add `SARVAM_API_KEY`
  as a Space secret, and either ship a prebuilt `data/index/` or run `ingest.py`
  in the build step (keep `MAX_ROWS` modest so the build finishes).
- **Render / Railway / Fly.io:** deploy the FastAPI app; set the env vars; build
  the index at deploy time or bake it into the image.
- Mic capture in the browser needs HTTPS (all the above provide it) or localhost.

## Layout
```
config.py          all tunables
ingest.py          dataset -> chunks -> index
benchmark.py       P50/P70/P100 analytics
eval.py            IR metrics vs gold labels (+ --sweep for the alpha grid)
scripts/
  calibrate_guardrail.py   picks MIN_DENSE_SCORE from on- vs off-topic scores
app.py             FastAPI server + endpoints
static/index.html  mic-record web UI
src/
  chunking.py      4 chunking strategies
  indexer.py       embeddings + FAISS + BM25 (+ save/load)
  retriever.py     hybrid fusion + rerank + metadata filter
  stt.py           Sarvam STT (retries)
  generator.py     extractive + grounded-LLM generation
  guardrails.py    input safety, off-topic, grounding checks
  harness.py       staged orchestrator with per-stage timing
  schemas.py       Pydantic contracts
```
