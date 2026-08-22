#!/usr/bin/env bash
# Deploy this project to a Hugging Face Space.
#
#   1. Create a WRITE token: https://huggingface.co/settings/tokens
#   2. huggingface-cli login          (or: export HF_TOKEN=hf_xxx)
#   3. ./deploy/push_to_space.sh <your-hf-username> [space-name]
#
# Then add SARVAM_API_KEY as a Space *secret* in the Space's Settings tab.
# Never commit the key -- .env is gitignored for exactly this reason.
set -euo pipefail

USER="${1:?usage: push_to_space.sh <hf-username> [space-name]}"
SPACE="${2:-voice-rag-hindi}"
REPO_ID="$USER/$SPACE"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"

echo "==> Target Space: https://huggingface.co/spaces/$REPO_ID"

if [ ! -f "$ROOT/data/index/dense.faiss" ]; then
  echo "!! No index at data/index/. Run 'python ingest.py' first -- the Space" >&2
  echo "   ships a prebuilt index; building it at image-build time would" >&2
  echo "   download the dataset and time the build out." >&2
  exit 1
fi

echo "==> Creating Space (idempotent) ..."
python - "$REPO_ID" <<'PY'
import sys
from huggingface_hub import HfApi
repo_id = sys.argv[1]
api = HfApi()
who = api.whoami()["name"]
owner = repo_id.split("/")[0]
if who.lower() != owner.lower():
    print(f"!! Token authenticates as '{who}' but target owner is '{owner}'.")
    print("   Refusing to publish to an account you are not logged in as.")
    sys.exit(1)
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                exist_ok=True)
print(f"   ok: {repo_id}")
PY

echo "==> Staging files ..."
cd "$ROOT"
git ls-files -z | xargs -0 -I{} cp --parents {} "$STAGE/"
cp "$ROOT/deploy/Dockerfile" "$STAGE/Dockerfile"
cp "$ROOT/deploy/README_space.md" "$STAGE/README.md"   # HF reads frontmatter here
mkdir -p "$STAGE/data/index"
cp "$ROOT"/data/index/* "$STAGE/data/index/"
rm -f "$STAGE/.env"                                    # belt and braces

echo "==> Files to upload:"
(cd "$STAGE" && find . -type f | sed 's|^\./|   |' | sort)

echo "==> Uploading (large index files go via LFS automatically) ..."
python - "$REPO_ID" "$STAGE" <<'PY'
import sys
from huggingface_hub import HfApi
repo_id, folder = sys.argv[1], sys.argv[2]
HfApi().upload_folder(
    repo_id=repo_id, repo_type="space", folder_path=folder,
    commit_message="Deploy voice-RAG demo",
)
print("   upload complete")
PY

rm -rf "$STAGE"
cat <<MSG

==> Done: https://huggingface.co/spaces/$REPO_ID

    FINAL STEP (required, or STT will fail):
      Space -> Settings -> Variables and secrets -> New secret
      Name:  SARVAM_API_KEY
      Value: <your key from .env>

    The build takes a while: it installs CPU torch and bakes in BGE-M3 (~2.3GB).
MSG
