#!/usr/bin/env bash
# Block until the GPU has enough free memory to load BGE-M3 (~2.3GB + headroom).
#
# Recording the demo GIF on CPU would put ~400ms embed latency on screen, which
# contradicts the 11.5ms P50 the README reports. Wait for real hardware instead.
#
# Usage:  ./scripts/wait_for_gpu.sh && python app_gradio.py
set -euo pipefail
NEED_MB="${1:-3000}"

while :; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$FREE" -ge "$NEED_MB" ]; then
    echo "GPU free: ${FREE} MiB — enough for BGE-M3. Go."
    exit 0
  fi
  HOLDER=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -1)
  printf '\r%s  free=%s MiB (need %s)  holder: %s' "$(date +%H:%M:%S)" "$FREE" "$NEED_MB" "${HOLDER:-none}"
  sleep 30
done
