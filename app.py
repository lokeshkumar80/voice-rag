"""
FastAPI server = the live working link + demo surface.

Endpoints:
  GET  /            -> mic-record web UI
  POST /ask         -> multipart audio -> full voice pipeline
  POST /ask_text    -> {"query": "..."} -> retrieval+generation (no mic needed)
  GET  /health      -> readiness

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from src.harness import Pipeline
from src.indexer import HybridIndex
from src.retriever import Retriever

app = FastAPI(title="Voice RAG — MSMARCO-XI")

_pipe: Pipeline | None = None


@app.on_event("startup")
def _load():
    global _pipe
    if not HybridIndex.exists():
        raise RuntimeError("No index found. Run `python ingest.py` first.")
    _pipe = Pipeline(Retriever(HybridIndex.load()))


def _serialize(res) -> dict:
    return {
        "ok": res.ok,
        "transcript": res.transcript,
        "answer": res.answer,
        "abstained": res.abstained,
        "grounded": res.grounded,
        "reason": res.reason,
        "lang": res.lang,
        "timing_ms": {
            "stt": round(res.timing.stt_ms, 1),
            "embed": round(res.timing.embed_ms, 1),
            "retrieve": round(res.timing.retrieve_ms, 1),
            "rerank": round(res.timing.rerank_ms, 1),
            "generate": round(res.timing.generate_ms, 1),
            "guardrail": round(res.timing.guardrail_ms, 1),
            "retrieval_total": round(res.timing.retrieval_total_ms, 1),
            "total": round(res.timing.total_ms, 1),
        },
        "contexts": [
            {"text": c.chunk.text[:400], "score": round(c.score, 3),
             "query_type": c.chunk.query_type}
            for c in res.contexts
        ],
    }


class TextQuery(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ready" if _pipe else "loading",
            "generation_mode": config.GENERATION_MODE, "lang": config.LANG}


@app.post("/ask")
async def ask(file: UploadFile = File(...)):
    audio = await file.read()
    # Forward the client's filename -- its extension is what picks the Sarvam
    # codec. Previously dropped, which mislabelled browser WebM as wav.
    res = _pipe.run(audio=audio, audio_filename=file.filename or "audio.webm")
    return JSONResponse(_serialize(res))


@app.post("/ask_text")
def ask_text(q: TextQuery):
    res = _pipe.run(text=q.query)
    return JSONResponse(_serialize(res))


# static UI
_STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))
