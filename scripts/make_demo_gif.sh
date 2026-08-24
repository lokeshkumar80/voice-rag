#!/usr/bin/env bash
# assets/demo.mp4 -> optimized assets/demo.gif for the README.
#
# Two-pass palette (palettegen + paletteuse). A single-pass GIF is limited to a
# generic 256-colour palette and looks washed out; generating a palette from the
# actual frames costs one extra pass and is the difference between a demo that
# looks deliberate and one that looks like a screen capture.
#
# Target: ~800px wide, under 10MB. GitHub stops animating GIFs past roughly
# 10MB -- it shows a still frame instead, with no warning.
#
# Usage:  ./scripts/make_demo_gif.sh [input.mp4] [output.gif]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Accept whatever the screen recorder produced. GNOME's built-in recorder emits
# WebM regardless of what you name the file, so don't assume .mp4 -- an extension
# that lies about the container is a small thing that wastes a real debugging
# session later. ffmpeg sniffs content, so any of these work.
default_input() {
  for f in "$ROOT"/assets/demo.webm "$ROOT"/assets/demo.mp4 "$ROOT"/assets/demo.mov "$ROOT"/assets/demo.mkv; do
    [ -f "$f" ] && { echo "$f"; return; }
  done
  echo "$ROOT/assets/demo.webm"     # for the error message below
}
IN="${1:-$(default_input)}"
OUT="${2:-$ROOT/assets/demo.gif}"
WIDTH="${WIDTH:-800}"
FPS="${FPS:-12}"                 # 12 is plenty for a UI screencast; 24 doubles size
MAX_MB=10

command -v ffmpeg >/dev/null || {
  echo "!! ffmpeg not found.  sudo apt install ffmpeg" >&2; exit 1; }

[ -f "$IN" ] || {
  echo "!! No input at: $IN" >&2
  echo "   Record a 20-30s clip as assets/demo.webm (or .mp4/.mov/.mkv)" >&2
  echo "   -- see assets/README.md." >&2
  exit 1; }

PALETTE="$(mktemp --suffix=.png)"
trap 'rm -f "$PALETTE"' EXIT

echo "==> Input : $IN ($(du -h "$IN" | cut -f1))"
echo "==> Target: ${WIDTH}px wide, ${FPS} fps"

echo "==> Pass 1/2: generating palette from the actual frames ..."
ffmpeg -hide_banner -loglevel error -y -i "$IN" \
  -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,palettegen=stats_mode=diff" \
  "$PALETTE"

echo "==> Pass 2/2: encoding ..."
ffmpeg -hide_banner -loglevel error -y -i "$IN" -i "$PALETTE" \
  -lavfi "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  "$OUT"

SIZE_MB=$(( $(stat -c%s "$OUT") / 1000000 ))
echo "==> Wrote $OUT (${SIZE_MB} MB)"

if [ "$SIZE_MB" -ge "$MAX_MB" ]; then
  cat >&2 <<MSG

!! ${SIZE_MB}MB is at or over GitHub's ~${MAX_MB}MB animation limit -- it will
   display as a still frame with no warning. Shrink it by any of:

     FPS=10 ./scripts/make_demo_gif.sh          # fewer frames
     WIDTH=640 ./scripts/make_demo_gif.sh       # smaller
     # or trim the clip first, which helps most:
     ffmpeg -i assets/demo.mp4 -t 20 -c copy assets/demo_short.mp4
     ./scripts/make_demo_gif.sh assets/demo_short.mp4
MSG
  exit 1
fi

echo "==> Under the ${MAX_MB}MB limit. Commit it:"
echo "     git add assets/demo.gif && git commit -m 'Add README demo GIF'"

if git -C "$ROOT" check-ignore -q "$OUT" 2>/dev/null; then
  echo "!! WARNING: $OUT is gitignored and would NOT be committed." >&2
  exit 1
fi
echo "==> Confirmed: not gitignored, will be tracked."
