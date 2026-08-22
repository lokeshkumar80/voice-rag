"""
The harness (requirement #5): structured orchestration around the pipeline.

It runs the stages in order, times each one, applies guardrails between stages,
and recovers from failures instead of crashing:

  audio ──(STT)──► transcript ──(input guardrail)──► [embed+retrieve]
        ──(off-topic guardrail)──► generate ──(grounding guardrail)──► answer

Every stage's latency is captured in StageTiming, which is what the benchmark
turns into P50/P70/P100 numbers. A text query can skip STT (skip_stt=True) so
you can benchmark the retrieval budget in isolation.
"""
from __future__ import annotations
import time
from typing import BinaryIO, Optional

import config
from src import guardrails, generator
from src.retriever import Retriever
from src.schemas import PipelineResult, StageTiming
from src import stt as stt_mod


class Pipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def run(self,
            *,
            audio: Optional[BinaryIO | bytes] = None,
            audio_filename: str = "audio.webm",
            text: Optional[str] = None,
            use_rerank: bool = config.USE_RERANK,
            generation_mode: Optional[str] = None) -> PipelineResult:
        timing = StageTiming()
        res = PipelineResult(ok=False, timing=timing)

        # ---- Stage 1: STT (skipped when a text query is supplied) ----
        if text is not None:
            transcript, lang = text.strip(), config.SARVAM_LANG
        else:
            if audio is None:
                res.reason = "no_input"
                return res
            t0 = time.perf_counter()
            try:
                # Pass the real filename through: its extension selects the
                # codec. Sarvam tolerates a wrong label, but don't rely on that.
                transcript, lang = stt_mod.transcribe(audio, filename=audio_filename)
            except Exception as e:
                res.reason = f"stt_error:{e}"
                res.timing.stt_ms = (time.perf_counter() - t0) * 1000
                return res
            timing.stt_ms = (time.perf_counter() - t0) * 1000
        res.transcript, res.lang = transcript, lang

        # ---- Guardrail: input ----
        g0 = time.perf_counter()
        ok, reason = guardrails.check_input(transcript)
        timing.guardrail_ms += (time.perf_counter() - g0) * 1000
        if not ok:
            res.reason, res.abstained = reason, True
            res.answer = ("I couldn't understand that." if reason == "empty_or_too_short"
                          else "I can't help with that request.")
            return res

        # ---- Stage 2: retrieve (embed + hybrid search + optional rerank) ----
        # retrieve() fills timing.embed_ms / retrieve_ms / rerank_ms in place.
        contexts = self.retriever.retrieve(transcript, use_rerank=use_rerank, timing=timing)
        res.contexts = contexts

        # ---- Guardrail: off-topic ----
        g0 = time.perf_counter()
        ok, reason = guardrails.check_retrieval(contexts)
        timing.guardrail_ms += (time.perf_counter() - g0) * 1000
        if not ok:
            res.reason, res.abstained = reason, True
            res.answer = guardrails.abstain_text(lang)
            res.ok = True     # abstaining correctly is a success, not an error
            return res

        # ---- Stage 3: generate ----
        t0 = time.perf_counter()
        answer = generator.generate(transcript, contexts, mode=generation_mode)
        timing.generate_ms = (time.perf_counter() - t0) * 1000
        res.answer = answer

        # ---- Guardrail: grounding / hallucination ----
        g0 = time.perf_counter()
        grounded, greason = guardrails.check_grounding(answer, contexts)
        timing.guardrail_ms += (time.perf_counter() - g0) * 1000
        res.grounded = grounded
        if not grounded:
            res.abstained, res.reason = True, greason
            res.answer = guardrails.abstain_text(lang)

        res.ok = True
        return res
