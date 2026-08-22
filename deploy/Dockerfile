# HF Spaces (Docker SDK) image for the voice-RAG demo.
# Free Spaces are CPU-only, so we install the CPU torch wheel explicitly --
# the default pip index serves the CUDA build (~2.5GB) which is dead weight here
# and slow enough to time the build out.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    DEVICE=cpu \
    INDEX_DIR=/app/data/index \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# CPU-only torch first, so the resolver never pulls the CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the embedder into the image. Without this the Space downloads 2.3GB on
# every cold start and the first request times out.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-m3', device='cpu')"

# HF Spaces runs as a non-root user; the cache must be writable by it.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
