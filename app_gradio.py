"""
Gradio front end -- the deployable demo (HF Spaces + ZeroGPU).

Why this exists alongside app.py: free HF Spaces only run Gradio on ZeroGPU
(Docker/Gradio-on-CPU need PRO), and ZeroGPU is Gradio-SDK-only. app.py remains
the FastAPI service with the raw /ask and /ask_text endpoints; this is the same
pipeline behind a UI that Spaces can host. Gradio's built-in mic recorder also
replaces the hand-rolled MediaRecorder JS in static/index.html.

ZeroGPU contract (see https://huggingface.co/docs/hub/en/spaces-zerogpu):
  - the model is placed on CUDA at *module* level, not lazily inside the handler
    (CUDA transfers are optimised for startup placement)
  - GPU work is wrapped in @spaces.GPU, which acquires a GPU per call and
    releases it after
  - the decorator is a no-op off-Spaces, so this file also just runs locally

Run locally:  python app_gradio.py
"""
from __future__ import annotations
import os
import time

import gradio as gr

import config
from src.harness import Pipeline
from src.indexer import HybridIndex, get_embedder
from src.retriever import Retriever

# --- ZeroGPU decorator, with a local no-op fallback -------------------------
try:
    import spaces
    GPU = spaces.GPU
    ON_ZEROGPU = True
except ImportError:                       # running locally, or a plain Space
    ON_ZEROGPU = False

    def GPU(fn=None, **_kw):
        """No-op stand-in so the same file runs off-Spaces."""
        if fn is None:
            return lambda f: f
        return fn

# --- module-level load (ZeroGPU wants CUDA placement at startup) ------------
if not HybridIndex.exists():
    raise SystemExit(
        f"No index at {config.INDEX_DIR}. Run `python ingest.py` first -- the "
        "Space ships a prebuilt index rather than building one at startup."
    )

print(f"Loading index from {config.INDEX_DIR} ...")
_INDEX = HybridIndex.load()
print(f"  {len(_INDEX.chunks):,} chunks")
get_embedder()                            # place BGE-M3 on CUDA now, not per-call
_PIPE = Pipeline(Retriever(_INDEX))
print("Ready.")

ABSTAIN_NOTE = (
    "The system abstained. That is the guardrail working, not a failure: nothing "
    "in the indexed corpus supports an answer, so it declines rather than "
    "inventing one."
)


def _timing_rows(t) -> list[list]:
    return [
        ["speech-to-text (Sarvam, network)", f"{t.stt_ms:.0f}"],
        ["embed query (BGE-M3)", f"{t.embed_ms:.1f}"],
        ["search (FAISS HNSW + BM25)", f"{t.retrieve_ms:.1f}"],
        ["rerank (cross-encoder)", f"{t.rerank_ms:.1f}"],
        ["generate (extractive)", f"{t.generate_ms:.1f}"],
        ["guardrails", f"{t.guardrail_ms:.1f}"],
        ["— retrieval budget (<200 ms target)", f"{t.retrieval_total_ms:.1f}"],
        ["— total", f"{t.total_ms:.0f}"],
    ]


@GPU(duration=90)
def answer(audio_path: str | None, text: str, use_rerank: bool):
    """One turn of the pipeline. Audio wins if both are supplied."""
    if not audio_path and not (text or "").strip():
        return "", "Record a question or type one.", [], "", ""

    t0 = time.perf_counter()
    if audio_path:
        with open(audio_path, "rb") as f:
            data = f.read()
        res = _PIPE.run(audio=data, audio_filename=os.path.basename(audio_path),
                        use_rerank=use_rerank)
    else:
        res = _PIPE.run(text=text, use_rerank=use_rerank)
    wall = (time.perf_counter() - t0) * 1000

    status = res.reason or ("abstained" if res.abstained else "answered")
    if res.abstained and res.reason not in ("unsafe_content", "empty_or_too_short"):
        status = f"{status} — {ABSTAIN_NOTE}"

    ctx = "\n\n".join(
        f"**[{i+1}]** (score {c.score:.3f})\n{c.chunk.text[:400]}"
        for i, c in enumerate(res.contexts[:5])
    ) or "_no context retrieved_"

    return (res.transcript or text or "", res.answer or "", _timing_rows(res.timing),
            f"`{status}`  ·  wall clock {wall:.0f} ms", ctx)


with gr.Blocks(title="Voice RAG · Hindi · MSMARCO-XI") as demo:
    gr.Markdown(
        "# 🎙️ Voice RAG — Hindi\n"
        "Ask a question in Hindi by voice or text. The pipeline transcribes it "
        "(Sarvam), retrieves from a chunked MS MARCO-XI corpus (BGE-M3 + FAISS "
        "HNSW), and answers **only** from what it retrieved.\n\n"
        f"**Corpus:** {len(_INDEX.chunks):,} chunks · "
        f"**chunking:** {config.CHUNK_STRATEGY} · "
        f"**fusion:** alpha={config.HYBRID_ALPHA} ({config.FUSION})"
    )
    with gr.Row():
        with gr.Column(scale=1):
            mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak (Hindi)")
            txt = gr.Textbox(label="…or type", placeholder="कॉर्पोरेशन क्या है?", lines=2)
            rr = gr.Checkbox(
                value=False,
                label="Cross-encoder rerank (better answers + far better abstention, ~330 ms slower)",
            )
            go = gr.Button("Ask", variant="primary")
            gr.Examples(
                examples=[["कॉर्पोरेशन क्या है?"], ["मधुमेह के लक्षण क्या हैं?"],
                          ["What is the capital of Mars?"], ["asdf qwerty zxcv"]],
                inputs=txt,
                label="Try these — the last two should make it abstain",
            )
        with gr.Column(scale=1):
            out_tr = gr.Textbox(label="Transcript", interactive=False)
            out_ans = gr.Textbox(label="Answer", lines=5, interactive=False)
            out_status = gr.Markdown()
    with gr.Row():
        out_time = gr.Dataframe(headers=["stage", "ms"], label="Per-stage latency",
                                interactive=False, wrap=True)
    with gr.Accordion("Retrieved context (what the answer is grounded in)", open=False):
        out_ctx = gr.Markdown()

    gr.Markdown(
        "---\n"
        "*Retrieval runs in ~11 ms P50 on an RTX 4060; speech-to-text is a network "
        "call to Sarvam and dominates end-to-end time at 1–2.5 s. Off-topic "
        "questions are meant to be refused — that behaviour is measured, not "
        "decorative: ungrounded answers drop from 100% to 14.7% with the "
        "cross-encoder gate enabled.*"
    )

    for trigger in (go.click, txt.submit):
        trigger(answer, inputs=[mic, txt, rr],
                outputs=[out_tr, out_ans, out_time, out_status, out_ctx])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.getenv("PORT", "7860")))
