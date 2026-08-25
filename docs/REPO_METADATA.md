# GitHub repo metadata

The description, homepage and topics are what a recruiter sees *before* opening
any file — the repo card in search, on your profile, and under "About". Leaving
them unset is a wasted first impression on an otherwise finished project.

**One command** (after `gh auth login`):

```bash
./scripts/set_repo_metadata.sh lokeshkumar80/voice-rag
```

Or paste manually in the web UI — description and homepage under
**Settings → General**, topics via the **gear icon beside "About"** on the repo
landing page.

## Description
```
Voice-enabled Hindi RAG pipeline (Sarvam STT -> BGE-M3 dense retrieval over FAISS-HNSW -> grounded answers) with measured ablations, guardrails, and sub-200ms retrieval. Deployed on HF ZeroGPU.
```

## Homepage
The **`.hf.space` origin**, not `huggingface.co/spaces/...`. The latter embeds
the app in a cross-origin iframe where browsers refuse microphone access, so a
visitor arriving from the repo card would find the mic broken.
```
https://lokeshkumar79-voice-rag-hindi.hf.space
```

## Topics
```
rag retrieval information-retrieval faiss bge-m3 sarvam hindi-nlp hugging-face-spaces semantic-search vector-search
```

## Verify
```bash
gh repo view lokeshkumar80/voice-rag --json description,homepageUrl,repositoryTopics
```
