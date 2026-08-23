# Voice-Enabled RAG · MSMARCO-XI

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/lokeshkumar79/voice-rag-hindi)
[![ZeroGPU](https://img.shields.io/badge/hardware-ZeroGPU-orange)](https://huggingface.co/docs/hub/en/spaces-zerogpu)
[![Retrieval P50](https://img.shields.io/badge/retrieval%20P50-11.4%20ms-brightgreen)](#latency--python-benchmarkpy---n-200)
[![MRR@10](https://img.shields.io/badge/MRR%4010-0.551-blue)](#retrieval-quality--python-evalpy---rows-2000---answer-f1)

**▶ Live demo: [huggingface.co/spaces/lokeshkumar79/voice-rag-hindi](https://huggingface.co/spaces/lokeshkumar79/voice-rag-hindi)**
— speak a Hindi question, or type one. Try asking something the corpus *can't*
answer; it should refuse. Note the free ZeroGPU tier allows 5 minutes of GPU per
day across all visitors, so the demo may report a quota error if it has been busy.

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
| 1. Speech-to-text (Sarvam) | `src/stt.py` — Saaras v3, retries, per-extension codec, enforced timeout |
| 2. Vast chunking | `src/chunking.py` — fixed / sentence / recursive / semantic, overlap, metadata. All four **benchmarked**; `sentence` won |
| 3. <200 ms latency | **11.5 ms P50 / 53.6 ms P100** over 103k chunks. See the framing note below — it is the retrieval segment, not the STT hop |
| 4. P50/P70/P100 analytics | `benchmark.py`, warmed up so cold start doesn't pollute P100 |
| 5. Harness | `src/harness.py` — staged orchestration, Pydantic I/O, retries, enforced stage timeouts, graceful degradation |
| 6. Guardrails | `src/guardrails.py` — input safety, off-topic abstain, grounding check. **Measured**, not asserted: `scripts/faithfulness.py` |
| Live demo | [HF Space on ZeroGPU](https://huggingface.co/spaces/lokeshkumar79/voice-rag-hindi) (`app_gradio.py`, `deploy/`) |

## About the 200 ms target — read this
The full voice path cannot hit 200 ms, and the measurements say so plainly:

| stage | measured |
|---|---|
| Sarvam STT (network round trip) | **1,100–2,600 ms** |
| retrieval — embed + search | **11.5 ms** P50 |
| generation (extractive) | ~30 ms |

**The STT hop alone is roughly 100x the entire local retrieval segment.** So the
budget is applied to the **query-time retrieval segment** — `embed + vector
search + rerank` — which is local, and genuinely fast. The harness times every
stage separately and `benchmark.py` reports retrieval P100 against the 200 ms
line *and* the end-to-end total, so nothing hides behind a favourable average.

Generation defaults to **extractive** (best grounded sentences, no LLM, ~30 ms).
`GENERATION_MODE=llm` works but sarvam-105b measured a **40 s median** — see
[Generation](#generation-why-extractive-and-a-bug-that-hid-for-weeks) below.

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
> First install pulls torch via `sentence-transformers` — on Linux that is the
> **CUDA build, ~2.5 GB**. For a CPU-only box, install
> `torch --index-url https://download.pytorch.org/whl/cpu` first (see
> `deploy/Dockerfile`, which does exactly this).

### 2. Configure
```bash
cp .env.example .env       # then paste your SARVAM_API_KEY into .env
```
Key knobs live in `config.py` (or set as env vars):
`LANG_CODE` (hi/te/ta/bn…), `PASSAGE_FIELD`, `USE_ENGLISH`, `MAX_ROWS`,
`CHUNK_STRATEGY`, `MIN_CHUNK_CHARS`, `HYBRID_ALPHA`, `FUSION`, `USE_RERANK`,
`MIN_DENSE_SCORE`, `MIN_RERANK_SCORE`, `GENERATION_MODE`, `EMBED_MAX_SEQ`,
`STAGE_TIMEOUT_S`.

Current measured defaults (each justified in **Measured results**):

| knob | value | why |
|---|---|---|
| `CHUNK_STRATEGY` | `sentence` | beat fixed/recursive/semantic |
| `HYBRID_ALPHA` | `1.0` | dense-only; BM25 hurt at this corpus size |
| `FUSION` | `rrf` | ≥ min-max at every alpha |
| `MIN_DENSE_SCORE` | `0.58` | calibrated on hard negatives |
| `USE_RERANK` | `false` | +332 ms breaks the budget; big accuracy/abstention win when on |
| `GENERATION_MODE` | `extractive` | LLM mode measured a 40 s median |

Two naming traps worth knowing:
- The variable is **`LANG_CODE`**, not `LANG`. `LANG` is the system locale
  variable — setting it does nothing here and can break your shell's locale.
- There is **no `en` language code.** English is not a separate file; it lives
  *inside* each language's parquet as `English_passages` / `Eng_Query` /
  `Eng_Answer`. (`LANG_CODE=en` raises `KeyError` — it isn't in
  `config.LANG_FILE`.) Use **`USE_ENGLISH=true`**, which swaps the passage,
  query *and* answer fields together — changing only `PASSAGE_FIELD` leaves
  Hindi queries pointed at English passages, which doesn't error, it just
  quietly returns bad results. `config.py` refuses to start on a half-applied
  switch, since values pinned in `.env` take precedence over the flag.

### 3. Cache the dataset, then build the index
```bash
python scripts/fetch_dataset.py     # ~460 MB, resumable — do this first
python ingest.py
```
**Fetch first.** Streaming straight from `hf://` on every run has no read
timeout, so a dropped connection parks the process at 0% CPU indefinitely rather
than failing — it cost us a stalled ablation and a stalled download before we
cached it. `config.data_files()` prefers `data/raw/` automatically once present.

`ingest.py` chunks `MAX_ROWS` rows, embeds locally and writes `data/index/`.
Reference points on an RTX 4060:

| `MAX_ROWS` | chunks | index size | embed time |
|---|---|---|---|
| 500 (smoke test) | ~5.4k | ~30 MB | ~1 min |
| **10,000 (what's measured here)** | **103,068** | **555 MB** | **~25 min** |

### 4. Check the latency numbers
```bash
python benchmark.py --n 200                 # retrieval path, extractive
python benchmark.py --n 200 --rerank        # with cross-encoder rerank
python benchmark.py --n 100 --with-generation
```
Prints a P50/P70/P100/mean table per stage and a PASS/OVER verdict vs 200 ms.
The numbers under **Measured results** below came from this.

### 5. Check retrieval quality and what the guardrails buy
```bash
python eval.py --rows 2000 --answer-f1          # IR metrics vs gold labels
python eval.py --rows 2000 --sweep              # fusion-weight sweep
python eval.py --rows 2000 --chunk fixed        # one chunking strategy
python scripts/faithfulness.py --n 150          # guardrails on vs off
python scripts/calibrate_guardrail.py           # re-derive MIN_DENSE_SCORE
```
⚠ **Re-run the last two after any corpus size change.** Abstention thresholds do
not survive scaling — the same threshold blocked 80% of unanswerable queries at
5.4k chunks and 52.7% at 103k. See [What the guardrails
buy](#what-the-guardrails-buy--python-scriptsfaithfulnesspy---n-150).

### 6. Run the app
Two front ends over the same pipeline:

```bash
python app_gradio.py                            # Gradio UI — what's deployed
uvicorn app:app --host 0.0.0.0 --port 8000      # FastAPI + raw endpoints
```
Both show transcript, grounded answer, per-stage timing and retrieved context.
Gradio is the deployable one (ZeroGPU is Gradio-only) and its `@spaces.GPU`
decorator is a no-op off-Spaces, so the same file runs locally. FastAPI exposes
`POST /ask` (audio) and `POST /ask_text` (JSON) for scripting.

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

⚠ **Caveat, and it matters given everything else on this page:** this ablation
ran at 500 rows, while the quality and latency numbers above are at 2,000 and
10,000. Three other tuned values flipped when the corpus grew, so the ranking of
the top three could too. What is unlikely to flip is `semantic` losing — it
trails on every metric at every scale tested, and costs ~10x more to build.
Re-run `eval.py --rows 2000 --chunk <s>` before treating the top-three order as
settled.

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

### Generation: why extractive, and a bug that hid for weeks
`GENERATION_MODE=llm` routes through Sarvam chat. It works — and its answers are
genuinely better, synthesizing a definition instead of quoting a sentence — but
it is **not** the default:

| mode | latency | quality |
|---|---|---|
| **extractive** (default) | ~30 ms | best grounded sentences, grounded by construction |
| `llm` (sarvam-105b) | 21 s / 40 s / 79 s (median **40 s**) | better prose |

40 seconds is unusable interactively, so extractive stays.

Testing that path turned up two things worth repeating:

**1. Graceful degradation hides breakage.** `sarvam-m` had been *deprecated* —
the API returns 400 pointing at `sarvam-105b`. Nobody noticed, because
`generate()` catches every exception and silently falls back to extractive. The
feature was dead and the system looked fine. **Test fallback paths directly,
bypassing the try/except**, or you are testing the fallback rather than the
feature.

**2. `STAGE_TIMEOUT_S` was declared and wired to nothing.** The harness
advertised a 15 s stage timeout that no code applied — hence a 79 s call running
to completion. A timeout you advertise but never enforce is worse than none: it
buys false confidence. It is now passed on both Sarvam calls. And because
tenacity was retrying *timeouts*, degrading took 3 × 15 s = 46 s; timeouts now
fail fast, so fallback lands in 15 s as configured.

## Experiments worth running
- Compare chunking strategies with `python eval.py --rows 2000 --chunk <s>`
  (already done for all four — see above; `sentence` won).
- Sweep the fusion weight with `python eval.py --rows 2000 --sweep`, and compare
  fusion methods with `FUSION=rrf` vs `FUSION=minmax` (see above).
- Re-run `scripts/faithfulness.py` after **any** corpus size change — abstention
  thresholds do not survive scaling.
- Show the guardrails firing: ask something off-topic and something unsafe; the
  system abstains instead of hallucinating.

## Deploying — live on ZeroGPU
**▶ [huggingface.co/spaces/lokeshkumar79/voice-rag-hindi](https://huggingface.co/spaces/lokeshkumar79/voice-rag-hindi)**
· one command: `./deploy/push_to_zerogpu.sh <hf-username>`

Getting there took five failed attempts, and the traps are undocumented enough
to be worth writing down.

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

### Five things that broke, in order
Each looked like a different problem and reported success while failing:

1. **`create_repo` returns 402 even for ZeroGPU** — hardware must be passed *at
   creation* (`space_hardware="zero-a10g"`). You cannot create a Gradio Space on
   free cpu-basic and switch afterwards; there is no first step to take.
2. **`short_description` > 60 chars** fails server-side *after* the Space is
   created and the index staged. `push_to_zerogpu.sh` now validates the
   frontmatter locally first.
3. **The 430 MB `dense.faiss` was silently dropped** — `Found 31 files …
   Committing 30/30`, exit code 0. The cause was **our own `.gitignore`**:
   `huggingface_hub` feeds it to the preupload API, and it carries `*.faiss` to
   keep build artifacts out of git. Correct for the repo, catastrophic for the
   Space, where the index *is* the payload.
4. **Removing it from the upload wasn't enough** — the resolution order ends
   with "the `.gitignore` already hosted on the Hub", and an earlier run had
   published one. It has to be deleted from the Space.
5. **The demo burned its whole daily GPU quota in ~8 queries** — `@spaces.GPU`
   wrapped the entire pipeline, so the Space held an accelerator through the
   1–2.5 s *network* call to Sarvam. STT now runs outside the GPU scope, and
   `duration` is 30 s rather than an over-declared 90 s.

The through-line: **three of these exited 0 while failing.** The deploy script
now verifies every staged file actually landed, because on this platform a green
exit code is not evidence.

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
