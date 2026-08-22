#!/usr/bin/env bash
# Deploy the Gradio demo to a Hugging Face ZeroGPU Space.
#
# Free personal accounts can host 2 ZeroGPU Spaces, provided the account is in
# "good standing": verified email AND older than 30 days. ZeroGPU is also
# Gradio-SDK-only -- Docker Spaces need PRO, which is why this exists separately
# from push_to_space.sh.
#
# One-time setup (you must do these):
#   1. Create a WRITE token: https://huggingface.co/settings/tokens
#   2. export HF_TOKEN=hf_xxx        (or: huggingface-cli login)
#
# Then:  ./deploy/push_to_zerogpu.sh <your-hf-username> [space-name]
#
# AFTER the push, two manual steps in the Space UI (neither can be scripted):
#   a) Settings -> Hardware -> select **ZeroGPU**
#   b) Settings -> Variables and secrets -> add secret SARVAM_API_KEY
set -euo pipefail

USER="${1:?usage: push_to_zerogpu.sh <hf-username> [space-name]}"
SPACE="${2:-voice-rag-hindi}"
REPO_ID="$USER/$SPACE"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> Target: https://huggingface.co/spaces/$REPO_ID"

[ -f "$ROOT/data/index/dense.faiss" ] || {
  echo "!! No index at data/index/. Run 'python ingest.py' first." >&2
  echo "   The Space ships a prebuilt index; building one at startup would" >&2
  echo "   exceed the startup timeout." >&2
  exit 1; }

IDX_MB=$(du -sm "$ROOT/data/index" | cut -f1)
echo "==> Index size: ${IDX_MB} MB (uploaded via LFS; large indexes take a while)"

echo "==> Creating Space (idempotent) ..."
python - "$REPO_ID" <<'PY'
import sys
from huggingface_hub import HfApi
repo_id = sys.argv[1]
api = HfApi()
who = api.whoami()["name"]
owner = repo_id.split("/")[0]
if who.lower() != owner.lower():
    # Guard against publishing to someone else's account by accident.
    print(f"!! Token authenticates as '{who}' but target owner is '{owner}'.")
    sys.exit(1)
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio",
                exist_ok=True)
print(f"   ok: {repo_id}")
PY

echo "==> Staging ..."
cd "$ROOT"
git ls-files -z | xargs -0 -I{} cp --parents {} "$STAGE/"
cp "$ROOT/deploy/README_zerogpu.md"      "$STAGE/README.md"        # HF reads frontmatter here
cp "$ROOT/deploy/requirements_space.txt" "$STAGE/requirements.txt" # slimmer, torch pinned
mkdir -p "$STAGE/data/index"
cp -r "$ROOT"/data/index/* "$STAGE/data/index/"
rm -f "$STAGE/.env"                        # never ship the key
rm -rf "$STAGE/deploy"                     # deploy scripts aren't part of the app

echo "==> Uploading ..."
python - "$REPO_ID" "$STAGE" <<'PY'
import sys
from huggingface_hub import HfApi
repo_id, folder = sys.argv[1], sys.argv[2]
HfApi().upload_folder(repo_id=repo_id, repo_type="space", folder_path=folder,
                      commit_message="Deploy voice-RAG Gradio demo")
print("   upload complete")
PY

cat <<MSG

==> Pushed: https://huggingface.co/spaces/$REPO_ID

    TWO MANUAL STEPS REMAIN (the API cannot set these):
      1. Settings -> Hardware        -> select ZeroGPU
      2. Settings -> Variables and secrets -> new SECRET
           SARVAM_API_KEY = <your key from .env>
         Without it the mic path fails; the text box still works.

    The first build installs torch + downloads BGE-M3 (~2.3GB). Expect a wait.
    If GPU allocation misbehaves, check torch is still within ZeroGPU's
    supported range (see deploy/requirements_space.txt).
MSG
