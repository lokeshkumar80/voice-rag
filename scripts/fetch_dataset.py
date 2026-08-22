"""
Download the language's parquet once, so runs stop depending on a live stream.

Streaming from hf:// re-downloads the split on every run, and the reader has no
read timeout -- when the HF edge drops the connection the process parks in
CLOSE-WAIT and hangs forever rather than failing. That cost us a stalled
ablation. Ablations re-read the same split 4+ times, so fetching once is both
faster and deterministic.

`config.data_files()` picks the local copy up automatically once it exists.

Run:  python scripts/fetch_dataset.py            # current LANG_CODE, validation
      python scripts/fetch_dataset.py --split train
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Known sizes, so a truncated transfer is caught rather than silently producing
# a corrupt parquet that fails much later inside the eval run.
EXPECTED_BYTES = {"hinval.parquet": 461888616}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=config.SPLIT, choices=["train", "validation"])
    args = ap.parse_args()

    prefix = config.LANG_FILE[config.LANG]
    fname = f"{prefix}train.parquet" if args.split == "train" else f"{prefix}val.parquet"
    remote = f"{args.split}/{fname}"
    dest_dir = config.DATA_CACHE_DIR
    dest = os.path.join(dest_dir, fname)

    if os.path.exists(dest):
        mb = os.path.getsize(dest) / 1e6
        print(f"Already cached: {dest} ({mb:.1f} MB)")
        return

    os.makedirs(dest_dir, exist_ok=True)
    url = (f"https://huggingface.co/datasets/{config.DATASET_ID}"
           f"/resolve/main/{remote}")
    print(f"Downloading {config.DATASET_ID}:{remote}")

    # curl, not hf_hub_download: these files are ~460MB and this transfer drops
    # often enough that a client without a stall timeout just hangs. Observed
    # twice -- the process parked at 0% CPU with zero open sockets, waiting on a
    # connection the far end had already closed. --speed-time/--speed-limit abort
    # a stalled transfer, --retry resumes it, -C - continues a partial file.
    cmd = [
        "curl", "-L", "--fail", "--retry", "10", "--retry-delay", "3",
        "--retry-all-errors", "--connect-timeout", "20",
        "--speed-time", "60", "--speed-limit", "1024",
        "-C", "-", "-o", dest, url,
    ]
    if subprocess.call(cmd) != 0:
        print(f"!! Download failed. Partial file kept at {dest}; rerun to resume.")
        sys.exit(1)

    size = os.path.getsize(dest)
    if size != EXPECTED_BYTES.get(fname, size):
        print(f"!! Size mismatch: got {size}, expected {EXPECTED_BYTES[fname]}.")
        print("   Rerun to resume the transfer.")
        sys.exit(1)
    print(f"Cached -> {dest} ({size / 1e6:.1f} MB)")
    print("config.data_files() will now prefer this local copy.")


if __name__ == "__main__":
    main()
