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
# Hardware and the SARVAM_API_KEY secret are both set at creation time, so there
# are no manual UI steps. Note the hardware CANNOT be set afterwards: creating a
# Gradio Space on free cpu-basic returns 402, so there is no Space to switch.
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

# The Sarvam key is set as a Space secret at creation time (below), so the mic
# path works on first boot. Read from .env, never hardcoded, never uploaded.
SARVAM_KEY="$(grep -E '^SARVAM_API_KEY=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- || true)"
[ -n "$SARVAM_KEY" ] || echo "!! SARVAM_API_KEY not in .env -- mic path will fail until you add the secret manually."

echo "==> Creating Space as ZeroGPU (idempotent) ..."
SARVAM_KEY="$SARVAM_KEY" python - "$REPO_ID" <<'PY'
import os
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

# space_hardware MUST be set at creation. A Space cannot be created on
# free cpu-basic at all (HTTP 402: Gradio/Docker Spaces need PRO) -- the
# ZeroGPU exemption applies to the Space's hardware, so creating it first and
# switching afterwards is impossible: there is no first step.
secrets = []
key = os.environ.get("SARVAM_KEY", "")
if key:
    secrets.append({"key": "SARVAM_API_KEY", "value": key,
                    "description": "Sarvam STT + chat API key"})
try:
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio",
                    space_hardware="zero-a10g",
                    space_secrets=secrets or None, exist_ok=True)
except Exception as e:
    if "402" in str(e):
        print("!! 402 even with ZeroGPU hardware requested. ZeroGPU needs an")
        print("   account in good standing: verified email AND older than 30")
        print("   days. Check https://huggingface.co/settings/account")
    raise
print(f"   ok: {repo_id} (hardware: zero-a10g)")
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

    Hardware (ZeroGPU) and the SARVAM_API_KEY secret were set at creation.
    Verify under Settings if the mic path misbehaves.

    The first build installs torch + downloads BGE-M3 (~2.3GB). Expect a wait.
    If GPU allocation misbehaves, check torch is still within ZeroGPU's
    supported range (see deploy/requirements_space.txt).
MSG
