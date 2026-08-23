# Résumé bullets — Voice-Enabled Hindi RAG

Copy-paste ready. Every number is measured and reproducible from this repo; the
section links point at the evidence in [`README.md`](../README.md).

- Built and deployed a voice-enabled Hindi RAG pipeline (Sarvam STT -> BGE-M3 dense
  retrieval over 103K FAISS-HNSW chunks -> grounded generation) with query-time
  retrieval at 11.5 ms P50 / 53.6 ms P100, holding a sub-200 ms budget on a corpus
  19x the initial build.
- Ran controlled ablations on gold-labeled MS MARCO-XI (chunking; dense/BM25/hybrid;
  RRF vs min-max fusion) and chose dense-only over hybrid on a negative result
  validated against its strongest counter-explanation.
- Profiled a retrieval regression to rank_bm25 consuming 99% of latency (superlinear
  degradation); swapped to a sparse-matrix BM25 for a 652x speedup at 0.998 rank
  correlation.
- Cut ungrounded answers from 100% to 15% with a cross-encoder abstention gate
  calibrated on realistic negatives; documented that abstention thresholds must be
  re-tuned per corpus size.
- Diagnosed five silent Hugging Face deploy failures (three exited 0 while failing),
  including a .gitignore rule that dropped the 430 MB index from the upload; shipped
  a one-command ZeroGPU deploy script.

---

## Where each claim is evidenced

| Bullet | Evidence |
|---|---|
| 11.5 ms P50 / 53.6 ms P100, 103K chunks | [Latency](../README.md#latency--python-benchmarkpy---n-200) |
| Ablations; dense-only over hybrid | [Does hybrid retrieval actually help?](../README.md#does-hybrid-retrieval-actually-help--a-measured-no) |
| rank_bm25 at 99% of latency; 652x | [Why it got faster while getting 19x bigger](../README.md#why-it-got-faster-while-getting-19x-bigger) |
| 100% -> 15% ungrounded; thresholds per corpus size | [What the guardrails buy](../README.md#what-the-guardrails-buy--python-scriptsfaithfulnesspy---n-150) |
| Five silent deploy failures | [Deploying — live on ZeroGPU](../README.md#deploying--live-on-zerogpu) |

## Talking points for interviews

The bullets lead with outcomes; these are what to say when someone digs. The
strongest material is the **negative results and the measurement mistakes
caught** — assembling a RAG pipeline is common, being able to say which of your
defaults are wrong and how you found out is not.

- **"Why dense-only? Isn't hybrid standard?"** It is, and it lost here. BM25
  helped at 5.4K chunks and stopped helping at 21.7K. Before accepting that, I
  tested the likelier explanation — that min-max score fusion was at fault, since
  it pins the best candidate at 1.0 however weak it is. RRF (the standard remedy)
  helped mid-range but still peaked at pure dense. Both things were true: the
  fusion was mildly suboptimal *and* BM25 genuinely does not help this corpus.
- **"How do you know the guardrail works?"** Because I measured it with the gate
  on and off. Its original threshold scored 0.93 balanced accuracy against
  gibberish and blocked only 26% of *realistic* unanswerable queries — gibberish
  sits far from everything in embedding space and validates almost any threshold.
  Recalibrating on real out-of-index questions, then switching to a cross-encoder
  signal (6.5x wider class separation), took it to 85%.
- **"Sub-200 ms on a voice pipeline?"** No — and the README says so in the first
  section. The STT hop alone is ~100x the local retrieval segment. The budget
  applies to query-time retrieval, and both numbers are always reported together.
- **"What surprised you?"** That every tuned value was corpus-size dependent.
  Chunking strategy, abstention threshold and fusion weight all flipped between
  5K and 100K chunks. Re-running each sweep after a scale change became standard
  practice rather than an afterthought.
