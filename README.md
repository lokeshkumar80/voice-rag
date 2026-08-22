# Voice-Enabled RAG · MSMARCO-XI

Voice in → Sarvam STT → chunk retrieval (FAISS HNSW dense + BM25, fusion
configurable) → grounded answer, inside a timed, guardrailed harness with
P50/P70/P100 latency analytics.

Every default here is a measured result, not a convention — including the ones
that came out against the obvious choice. Dense-only beats hybrid on this corpus,
plain sentence splitting beats semantic chunking, and the abstention threshold
that scored 0.93 on easy negatives was worth almost nothing on realistic ones.

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
| 6. Guardrails | `src/guardrails.py` (input safety, off-topic abstain, hallucination/grounding check); measured in `scripts/faithfulness.py` |

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
Hindi (`LANG_CODE=hi`), 10,000 validation rows -> **103,068 chunks**, BGE-M3 on an RTX 4060.

### Latency — `python benchmark.py --n 200`
| stage (ms) | P50 | P70 | P100 |
|---|---|---|---|
| embed | 9.49 | 9.73 | 40.23 |
| retrieve | 2.06 | 2.33 | 14.68 |
| **retrieval_total** | **11.54** | **11.86** | **53.57** |

**Retrieval P100 = 53.57 ms vs the 200 ms target — PASS**, on a corpus 19x larger
than the earlier 5,442-chunk build *and faster than it was* (P50 14.00 → 11.54 ms).

#### Why it got faster while getting 19x bigger
Scaling to 103k chunks first made it *much* slower — retrieval P50 91 ms, P100
525 ms, **OVER budget**. Profiling split the blame cleanly:

| component | 103k chunks |
|---|---|
| FAISS HNSW search | 0.57 ms |
| `rank_bm25` `get_scores` | **63.75 ms** |

BM25 was **99%** of retrieval time. `rank_bm25` is pure Python and degrades
*superlinearly*: extrapolating linearly from the 5.4k-chunk corpus predicted
17 ms, and the real cost was 63.75 ms — **3.7x worse than the extrapolation**.
Swapping to `bm25s` (sparse-matrix backed) took the same query from 63.75 ms to
**0.06 ms — 652x** — at 0.9983 rank correlation with the old implementation.

The lesson is the one the whole project keeps re-learning: *extrapolate to form a
hypothesis, then measure.* The linear estimate was confident and wrong.
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

### Retrieval quality — `python eval.py --rows 2000 --answer-f1`
**1,219** queries with a gold passage (`is_selected==1`), over a 21,657-chunk
eval index. (`eval.py` builds its own labelled index, separate from the serving
index — see Layout.)

| config | R@1 | R@5 | Hit@5 | MRR@10 | nDCG@10 | Answer F1 |
|---|---|---|---|---|---|---|
| BM25 only | 0.136 | 0.402 | 0.424 | 0.259 | 0.311 | 0.196 |
| Dense only | 0.312 | 0.701 | 0.727 | 0.494 | 0.562 | 0.281 |
| Hybrid a=0.6 | 0.212 | 0.612 | 0.641 | 0.400 | 0.478 | 0.240 |
| **Hybrid + rerank** | **0.364** | **0.744** | **0.773** | **0.551** | **0.610** | **0.293** |

`sentence` chunking — the measured default. `Hybrid a=0.6` is the ablation grid's
fixed point, not the tuned value; see the alpha sweep below.

#### These numbers are *lower* than the small-corpus run, on purpose
An earlier 500-row eval (5,592 chunks, 248 queries) scored MRR@10 **0.606** and
Hit@5 **0.871**. Quadrupling the haystack drops those to **0.551** and **0.773**.

Nothing got worse — the earlier task was simply easier. Retrieval metrics are a
property of the *corpus size as much as the retriever*, so a headline MRR quoted
without its corpus size means very little. These are the more credible numbers,
and they are the ones worth reporting.

Read `R@1` with care: it divides by the number of gold *chunks*, so when a gold
passage splits into several chunks it is capped well below 1.0 by construction.
`Hit@5` and `MRR@10` are the fair cross-config comparisons.

### Which chunking strategy? — `python eval.py --rows 500 --chunk <s> --answer-f1`
All four strategies, same 248 gold-labelled queries, best config (hybrid + rerank):

| strategy | chunks | R@5 | Hit@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| **sentence** | 5,592 | **0.843** | 0.871 | 0.606 | **0.673** |
| fixed | 5,602 | 0.832 | 0.875 | **0.612** | 0.669 |
| recursive | 5,958 | 0.827 | **0.879** | 0.604 | 0.663 |
| semantic | 7,975 | 0.676 | 0.839 | 0.578 | 0.574 |

**The sophisticated option loses.** `semantic` chunking — embed every sentence,
split at topic boundaries — came last on every metric *and* costs ~10x more to
build, since it runs the embedder over each sentence before indexing.

Be careful reading `R@5` across strategies: it divides by the number of gold
*chunks*, and `semantic` emits ~40% more chunks, which inflates its denominator.
The denominator-free metrics are the fair comparison — and `semantic` is still
last on both (Hit@5 0.839 vs 0.871, MRR@10 0.578 vs 0.606), so the conclusion
holds. The honest margin is the smaller one, not the R@5 gap.

The top three are within noise of each other on 248 queries. `sentence` is the
default because it wins nDCG@10, produces the smallest index, and never cuts
mid-sentence — which matters here because the extractive generator returns whole
sentences.

### What the guardrails buy — `python scripts/faithfulness.py --n 150`
Same queries, run twice: `guardrails_enabled=True` vs `False`.

At **103k chunks**, both gates:

| gate | answers *answerable* | answers *unanswerable* | extra latency |
|---|---|---|---|
| guardrails OFF | 100.0% | 100.0% | — |
| cosine (default, fast path) | 88.7% | 52.7% | 0 ms |
| **cross-encoder** (`USE_RERANK=true`) | 76.7% | **14.7%** | ~332 ms |

**Ungrounded answers: 100% → 52.7% on the fast path, → 14.7% (an 85% reduction)
with the cross-encoder.** Both halves matter: a guardrail that abstained on
everything would post a perfect 0% hallucination rate.

#### Guardrails get *harder* as the corpus grows
On the 5.4k-chunk corpus the cosine gate blocked **80%** of unanswerable queries.
At 103k chunks the same threshold blocks **52.7%**. Nothing regressed — a larger
corpus simply means any query, answerable or not, is likelier to find a plausible
near-match. Gibberish that scored 0.427 at 5.4k scores 0.499 at 103k.

**An abstention threshold is only valid for the corpus size it was tuned on.**
That is invisible until you scale, which is the argument for scaling before
believing a safety number.

#### Why the cross-encoder, and why there is no cheap version
| discriminator | answerable | unanswerable | gap | best bal. acc |
|---|---|---|---|---|
| bi-encoder cosine | 0.669 | 0.590 | 0.079 | 0.710 |
| **cross-encoder** | 0.860 | 0.350 | **0.510** | **0.813** |

The cross-encoder separates the two **6.5x more widely**, because it does full
query-document attention instead of comparing independently-encoded vectors.
When `USE_RERANK=true` its score is already computed, so the better gate is free.

A cascade — cosine first, cross-encoder only for ambiguous cases — was built and
**rejected on measurement**: the distributions overlap so heavily that 71% of
queries land in the ambiguous band, costing ~236 ms to buy *less* accuracy
(0.793) than simply always running the cross-encoder (0.813).

The negative set is the part that makes this real. It is *not* hand-written
nonsense; it is real MS MARCO queries from rows **after** the indexed slice —
natural, on-domain questions whose passages simply were never indexed.

#### The mistake worth reading
The abstention threshold was originally calibrated against gibberish
("asdf qwerty", "capital of Mars"). It scored **0.93 balanced accuracy** and
looked finished. Measured against realistic negatives — real MS MARCO queries
whose passages were never indexed — that same threshold blocked only **26%**.

Gibberish sat at a **0.434** cosine; real unanswerable queries reached **0.719**.
Easy negatives sit so far from everything in embedding space that they validate
almost any threshold. Recalibrating on hard negatives moved `MIN_DENSE_SCORE`
0.50 → 0.58 and took the block rate from 26% to 80% (on the 5.4k corpus).

Calibrating a safety mechanism on easy cases gives you a number, not a guarantee.
`scripts/calibrate_guardrail.py` now draws its negatives from validation rows
past `MAX_ROWS`, and prints the trivial-gibberish score purely as a reminder of
how easy that old test was.

### Does hybrid retrieval actually help? — a measured *no*
`python eval.py --rows 2000 --sweep` (1,219 queries, 21,657-chunk eval index)

| alpha | 0.0 | 0.6 | 0.8 | 0.9 | **1.0** |
|---|---|---|---|---|---|
| MRR@10 (min-max fusion) | 0.259 | 0.401 | 0.458 | 0.485 | **0.496** |
| MRR@10 (RRF fusion) | 0.259 | 0.425 | 0.464 | 0.483 | **0.495** |

`alpha=1.0` is pure dense. **Adding BM25 at any weight makes retrieval worse**, so
the default is dense-only — and BM25 stays wired in because the answer is
corpus-dependent, not permanent.

Two things worth noting, because the naive read of this table is wrong:

**1. This flipped with scale.** On the 5,442-chunk corpus BM25 *did* add a real
lift and `alpha=0.9` was optimal. At 21,657 chunks BM25 weakens relatively
(MRR@10 0.259 vs dense 0.494) and mixing it in only costs accuracy. The tuned
value from the small corpus was actively wrong at the larger one.

**2. We tested the obvious alternative explanation before accepting it.** The
suspicion was that *min-max fusion* was at fault rather than BM25 — it normalizes
over the candidate set, pinning the best candidate at 1.0 however weak it is.
(The exact same flaw made the original fused-score guardrail unable to fire, so
it had form.) Reciprocal Rank Fusion, the standard remedy, combines ranks instead
and is scale-free. It **did** help mid-range — alpha=0.6: 0.425 vs 0.401 — but
still peaked at 1.0.

So both are true: the fusion was mildly suboptimal (RRF is now the default), *and*
BM25 genuinely does not help on this corpus. A negative result that survives its
own best counter-argument is worth more than a positive one that was never tested.

## Experiments worth running
- Compare chunking strategies: re-run `ingest.py` with `CHUNK_STRATEGY=fixed` vs
  `semantic`, benchmark each, and show retrieval-quality/latency trade-offs.
- Sweep the fusion weight with `python eval.py --rows 2000 --sweep`, and compare
  fusion methods with `FUSION=rrf` vs `FUSION=minmax` (see above).
- Re-run `scripts/faithfulness.py` after **any** corpus size change — abstention
  thresholds do not survive scaling.
- Show the guardrails firing: ask something off-topic and something unsafe; the
  system abstains instead of hallucinating.

## Deploying (not currently deployed — this runs locally)

**Read this before reaching for Spaces.** As of August 2026 Hugging Face
[gates compute Spaces behind a paid plan](https://huggingface.co/docs/hub/en/spaces-overview):

> Static Spaces are free for everyone. Gradio and Docker Spaces run on compute
> and require a paid plan to create: PRO for personal accounts.

Creating a Docker Space on a free account returns **HTTP 402**. Note what this
does *and does not* mean: the gate is the **Space type**, not resource usage —
so shrinking or quantizing the model does **not** get you a free Docker Space.

Free options that do work:

| Option | Compute | Cost | Catch |
|---|---|---|---|
| **Gradio + [ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu)** | Real GPU (48GB) | Free, 2 Spaces | Gradio SDK only; 5 min/day GPU quota; account >30 days + verified email |
| Static Space | None | Free | Showcase page only, no backend |
| HF model/dataset repo | None | Free | Git-backed tracking, like GitHub |
| Google Cloud Run | 4GB CPU | Free tier | Needs billing enabled; CPU-only, so latency ≫ the GPU numbers above |

**ZeroGPU is the best free path** — it gives a real GPU, so the published
latency numbers roughly hold instead of needing a CPU caveat.

That front end is **already written**: [`app_gradio.py`](app_gradio.py) runs the
same pipeline, and `gr.Audio(sources=["microphone"])` replaced the hand-rolled
MediaRecorder JS. The `@spaces.GPU` decorator degrades to a no-op off-Spaces, so
the identical file runs locally with `python app_gradio.py`.

```bash
export HF_TOKEN=hf_xxx                       # a WRITE token
./deploy/push_to_zerogpu.sh <your-username>
```
Then, in the Space UI (neither is scriptable): **Settings → Hardware → ZeroGPU**,
and add `SARVAM_API_KEY` as a **secret**.

Two gotchas worth knowing before you try: ZeroGPU only supports specific PyTorch
versions (2.8.0–2.11.0 as of Aug 2026), so `deploy/requirements_space.txt` pins
it — leave it unpinned and pip may install a version outside that window and
break GPU allocation. And "good standing" means a verified email **and** an
account older than 30 days.

`deploy/` holds a portable Dockerfile (honours `$PORT`, so one image serves
Cloud Run's 8080 and Spaces' 7860), plus scripts for both targets. The Cloud Run
script refuses to run if the HF/GCP identity doesn't match the target owner, so
a demo can't land on the wrong account.

- Mic capture in the browser needs HTTPS or localhost.
- Never bake `SARVAM_API_KEY` into an image — pass it as a runtime secret.

## Layout
```
config.py          all tunables
ingest.py          dataset -> chunks -> index
benchmark.py       P50/P70/P100 analytics
eval.py            IR metrics vs gold labels (+ --sweep for the alpha grid)
scripts/
  fetch_dataset.py         cache the parquet locally (resumable, stall-safe)
  calibrate_guardrail.py   picks MIN_DENSE_SCORE from HARD negatives
  faithfulness.py          what the guardrails buy: guardrails on vs off
app.py             FastAPI server + endpoints (/ask, /ask_text)
app_gradio.py      Gradio UI -- the deployable demo (HF Spaces + ZeroGPU)
static/index.html  mic-record web UI (used by app.py)
deploy/            Dockerfile + push scripts (Cloud Run / Spaces / ZeroGPU)
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
