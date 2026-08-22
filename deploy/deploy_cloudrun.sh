#!/usr/bin/env bash
# Deploy the voice-RAG demo to Google Cloud Run.
#
# One-time setup (interactive, you must do these):
#   1. Create a project at https://console.cloud.google.com and ENABLE BILLING.
#      Cloud Run's always-free tier covers a demo, but the project still needs
#      a billing account attached or deploys are rejected.
#   2. gcloud auth login
#   3. gcloud config set project <your-project-id>
#
# Then:  ./deploy/deploy_cloudrun.sh [service-name] [region]
#
# The Sarvam key is passed as a runtime env var, never baked into the image.
set -euo pipefail

SERVICE="${1:-voice-rag-hindi}"
REGION="${2:-asia-south1}"          # Mumbai: closest to Sarvam's API, lowest STT RTT
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

command -v gcloud >/dev/null || { echo "!! gcloud not on PATH." >&2; exit 1; }

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || {
  echo "!! No project set. Run: gcloud config set project <your-project-id>" >&2
  exit 1; }

[ -f "$ROOT/data/index/dense.faiss" ] || {
  echo "!! No index at data/index/. Run 'python ingest.py' first." >&2
  echo "   The image ships a prebuilt index; building it at image-build time" >&2
  echo "   would download the dataset and blow the Cloud Build timeout." >&2
  exit 1; }

# Read the key from .env (gitignored). Never hardcode it, never bake it in.
SARVAM_KEY="$(grep -E '^SARVAM_API_KEY=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- || true)"
[ -n "$SARVAM_KEY" ] || { echo "!! SARVAM_API_KEY not found in .env" >&2; exit 1; }

echo "==> Project: $PROJECT   Service: $SERVICE   Region: $REGION"

echo "==> Enabling required APIs (idempotent) ..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --quiet

echo "==> Staging build context ..."
cd "$ROOT"
git ls-files -z | xargs -0 -I{} cp --parents {} "$STAGE/"
cp "$ROOT/deploy/Dockerfile" "$STAGE/Dockerfile"     # Cloud Build wants it at root
mkdir -p "$STAGE/data/index"
cp "$ROOT"/data/index/* "$STAGE/data/index/"
rm -f "$STAGE/.env"                                  # belt and braces
printf 'data/raw/\n.venv/\n__pycache__/\n' > "$STAGE/.gcloudignore"

echo "==> Deploying (first build is slow: CPU torch + BGE-M3 bake in) ..."
# Memory: BGE-M3 is ~2.3GB in fp32, so 4Gi with headroom for FAISS + the index.
# startup-cpu-boost shortens the cold start where the model loads.
# min-instances=0 keeps it inside the always-free tier; the tradeoff is a slow
# first request after idle. Set it to 1 if you want the demo always warm (costs).
gcloud run deploy "$SERVICE" \
  --source "$STAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --startup-cpu-boost \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 300 \
  --port 8080 \
  --set-env-vars "DEVICE=cpu,INDEX_DIR=/app/data/index,SARVAM_API_KEY=${SARVAM_KEY}" \
  --quiet

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" \
        --format='value(status.url)')"
cat <<MSG

==> Live: $URL

    Health:  curl $URL/health
    Ask:     curl -X POST $URL/ask_text -H 'Content-Type: application/json' \\
               -d '{"query":"कॉर्पोरेशन क्या है?"}'

    NOTE: min-instances=0 means the first request after idle pays a cold start
    (the 2.3GB model loads). That is the price of staying in the free tier.
MSG
