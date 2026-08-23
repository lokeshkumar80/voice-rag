# Demo recording — how to make `assets/demo.gif`

## Why this exists
The live Space runs on free ZeroGPU: **5 minutes of GPU per day, shared across
every visitor.** A recruiter clicking through on a busy day gets a quota error,
not a demo. A GIF in the README always works — it is the hedge, not a
replacement for the live link.

## What to record (20–30 seconds)
Keep it short and show the two things that make this project interesting:

1. **A Hindi question that gets answered.** Use the mic if you can — it is a
   *voice* RAG project and the transcript appearing is the payoff. Suggested:
   `कॉर्पोरेशन क्या है?`
2. **An off-topic question that gets refused.** This is the part most RAG demos
   cannot show. Suggested: `What is the capital of Mars?` — it should abstain
   with `off_topic_low_score_...`.

Let the per-stage latency table stay visible in frame. The STT-vs-retrieval split
(~1–2.5 s vs ~11 ms) is the whole latency argument, visible in one glance.

**Do not** record the API key, `.env`, or your browser's other tabs.

## How to record

⚠ **Record on a GPU, not CPU.** `DEVICE=cpu` runs fine but puts ~400 ms embed
latency on screen, contradicting the 11.5 ms P50 the README reports. A demo that
disagrees with your own headline number is worse than no demo.

Either the live Space or a local run works — local does not spend ZeroGPU quota:

```bash
./scripts/wait_for_gpu.sh && python app_gradio.py    # http://localhost:7860
```

`wait_for_gpu.sh` blocks until ~3 GB is free, since another job on this machine
(or a stale process) will otherwise OOM the model load. Check what is holding it
with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.

| OS | Tool |
|---|---|
| Ubuntu (GNOME) | `Ctrl+Alt+Shift+R` starts/stops the built-in recorder → `~/Videos` |
| Any | [OBS Studio](https://obsproject.com/), or `peek` (`sudo apt install peek`) |

Record the **browser window only**, not the full desktop — a 1920×1080 desktop
capture scaled to 800px makes the text unreadable.

## Then
1. Save or rename the clip to **`assets/demo.mp4`**
2. Run:
   ```bash
   ./scripts/make_demo_gif.sh
   ```
3. Commit `assets/demo.gif`. The README embed activates automatically — remove
   the `<!-- TODO -->` marker above it once the file exists.

The script targets ~800px wide and under 10MB (GitHub renders inline up to
~10MB; beyond that it silently refuses to animate). It will tell you if the
result is too large and how to shrink it.
