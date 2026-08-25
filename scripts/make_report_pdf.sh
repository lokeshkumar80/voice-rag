#!/usr/bin/env bash
# docs/report.html -> docs/Voice-RAG-Field-Report.pdf
#
# The PDF duplicates numbers that live in README.md. A stale PDF contradicting a
# live README is worse than shipping no PDF at all, so this script regenerates
# from source and tests/test_report_consistency.py fails CI if they diverge.
#
# Usage:  ./scripts/make_report_pdf.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/docs/report.html"
CSS="$ROOT/docs/report-print.css"
OUT="$ROOT/docs/Voice-RAG-Field-Report.pdf"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

CHROME=""
for c in google-chrome google-chrome-stable chromium chromium-browser; do
  command -v "$c" >/dev/null 2>&1 && { CHROME="$c"; break; }
done
[ -n "$CHROME" ] || { echo "!! No Chrome/Chromium found (needed to render)." >&2; exit 1; }
[ -f "$SRC" ] || { echo "!! Missing $SRC" >&2; exit 1; }
[ -f "$CSS" ] || { echo "!! Missing $CSS" >&2; exit 1; }

echo "==> Assembling print variant ..."
# The sticky nav rail is hidden in print, so the PDF would otherwise have no way
# to navigate 23 pages. Inject a print-only contents block derived from the
# section headings themselves, so it cannot fall out of step with the document.
python - "$SRC" "$STAGE/body.html" <<'PY'
import html
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()

items = re.findall(r'<h2><span class="num">(\d+)</span>\s*(.*?)</h2>', s, re.S)
if not items:
    sys.exit("!! No numbered <h2> sections found; cannot build contents.")

lis = "\n".join(
    f"    <li><b>{n}</b> {html.escape(re.sub(r'<[^>]+>', '', t).strip())}</li>"
    for n, t in items
)
toc = f'<nav class="print-toc" aria-label="Contents">\n  <h2>Contents</h2>\n  <ol>\n{lis}\n  </ol>\n</nav>\n'

if "</header>" not in s:
    sys.exit("!! No </header> anchor to insert contents after.")
open(dst, "w", encoding="utf-8").write(s.replace("</header>", "</header>" + toc, 1))
print(f"   contents built from {len(items)} sections")
PY

cat "$STAGE/body.html" "$CSS" > "$STAGE/print.html"

echo "==> Rendering with $CHROME ..."
# --no-pdf-header-footer matters: without it Chrome stamps the source file:// URL
# onto every page, which leaks a local path into a shared document.
"$CHROME" --headless=new --disable-gpu --no-sandbox --no-first-run \
  --hide-scrollbars --force-color-profile=srgb \
  --virtual-time-budget=25000 --run-all-compositor-stages-before-draw \
  --no-pdf-header-footer --print-to-pdf="$OUT" \
  "file://$STAGE/print.html" 2>&1 | grep -iE "written|error" || true

[ -f "$OUT" ] || { echo "!! Chrome produced no PDF." >&2; exit 1; }

echo "==> Verifying ..."
python - "$OUT" <<'PY'
import sys
try:
    from pypdf import PdfReader
except ImportError:
    print("   (pypdf not installed — skipping content check; pip install pypdf)")
    sys.exit(0)

r = PdfReader(sys.argv[1])
text = "\n".join((p.extract_text() or "") for p in r.pages)
if "/tmp/" in text or "file://" in text:
    sys.exit("!! A local path leaked into the PDF — check --no-pdf-header-footer.")
print(f"   {len(r.pages)} pages, {len(text):,} chars, no path leakage")
PY

echo "==> $OUT ($(du -h "$OUT" | cut -f1))"
echo "    Numbers are guarded by tests/test_report_consistency.py — run pytest"
echo "    after editing README.md or the report."
