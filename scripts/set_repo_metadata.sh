#!/usr/bin/env bash
# Set the GitHub repo's description, homepage and topics.
#
# These are what a recruiter sees before they open a single file: the repo card
# in search results and on your profile. An unset description shows blank.
#
# Requires `gh auth login` first (the agent shell has no GitHub credentials).
# Values are also in docs/REPO_METADATA.md if you would rather paste them into
# the web UI: Settings -> General for description/homepage, and the gear icon
# beside "About" on the repo landing page for topics.
set -euo pipefail

REPO="${1:-lokeshkumar80/voice-rag}"

gh auth status >/dev/null 2>&1 || {
  echo "!! Not authenticated. Run: gh auth login" >&2
  exit 1; }

echo "==> Updating $REPO ..."
gh repo edit "$REPO" \
  --description "Voice-enabled Hindi RAG pipeline (Sarvam STT -> BGE-M3 dense retrieval over FAISS-HNSW -> grounded answers) with measured ablations, guardrails, and sub-200ms retrieval. Deployed on HF ZeroGPU." \
  --homepage "https://lokeshkumar79-voice-rag-hindi.hf.space" \
  --add-topic rag \
  --add-topic retrieval \
  --add-topic information-retrieval \
  --add-topic faiss \
  --add-topic bge-m3 \
  --add-topic sarvam \
  --add-topic hindi-nlp \
  --add-topic hugging-face-spaces \
  --add-topic semantic-search \
  --add-topic vector-search

echo "==> Verifying ..."
gh repo view "$REPO" --json description,homepageUrl,repositoryTopics
