"""Structured input/output contracts used everywhere in the harness."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: int
    text: str
    # metadata-aware retrieval: carried through and usable as filters
    query_id: Optional[int] = None
    query_type: Optional[str] = None
    is_selected: Optional[int] = None
    source_row: Optional[int] = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: Optional[float] = None


class StageTiming(BaseModel):
    """Per-stage latency in milliseconds; the basis for P50/P70/P100 analytics."""
    stt_ms: float = 0.0
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    guardrail_ms: float = 0.0

    @property
    def retrieval_total_ms(self) -> float:
        # The segment the <200ms budget applies to (query-time, no STT, no LLM).
        return self.embed_ms + self.retrieve_ms + self.rerank_ms

    @property
    def total_ms(self) -> float:
        return (self.stt_ms + self.embed_ms + self.retrieve_ms
                + self.rerank_ms + self.generate_ms + self.guardrail_ms)


class PipelineResult(BaseModel):
    ok: bool
    transcript: str = ""
    answer: str = ""
    abstained: bool = False
    reason: str = ""                     # why we abstained / errored
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    timing: StageTiming = Field(default_factory=StageTiming)
    grounded: bool = True
    lang: str = ""
