"""
Speech-to-text via Sarvam (requirement #1). Wrapped with retries + timeout so a
transient network blip doesn't kill the request (part of the harness contract).

Sarvam REST STT accepts audio <= 30s. SDK:
    client.speech_to_text.transcribe(file=..., model="saaras:v3", language_code="hi-IN")
    -> object with .transcript and .language_code
"""
from __future__ import annotations
import io
import os
from typing import BinaryIO, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config


class STTError(RuntimeError):
    pass


# The SDK takes an explicit `input_audio_codec`. We used to hardcode the filename
# to "audio.wav" while the browser's MediaRecorder actually uploads WebM/Opus.
# Verified 2026-08-22: Sarvam sniffs the real content and transcribes correctly
# even when mislabelled, so this was never an outage -- but it depended on
# undocumented provider leniency. Send the real codec instead of relying on it.
_CODEC_BY_EXT = {
    ".wav": "wav", ".wave": "wav", ".mp3": "mp3", ".webm": "webm",
    ".ogg": "ogg", ".opus": "opus", ".flac": "flac", ".m4a": "x-m4a",
    ".mp4": "mp4", ".aac": "aac", ".aiff": "aiff", ".amr": "amr",
}


def codec_for(filename: str) -> Optional[str]:
    """Best-effort codec for Sarvam's `input_audio_codec`; None lets it sniff."""
    return _CODEC_BY_EXT.get(os.path.splitext(filename or "")[1].lower())


def _client():
    if not config.SARVAM_API_KEY:
        raise STTError("SARVAM_API_KEY is not set. Add it to your .env file.")
    from sarvamai import SarvamAI
    return SarvamAI(api_subscription_key=config.SARVAM_API_KEY)


@retry(reraise=True,
       stop=stop_after_attempt(config.STAGE_RETRIES + 1),
       wait=wait_exponential(multiplier=0.3, max=3),
       retry=retry_if_exception_type(STTError))
def transcribe(audio: BinaryIO | bytes, filename: str = "audio.webm",
               codec: Optional[str] = None) -> Tuple[str, str]:
    """Return (transcript, detected_language_code).

    `filename` should be the *real* name the client uploaded -- the extension is
    what determines the codec. Browser MediaRecorder produces WebM/Opus, so that
    is the default here rather than the wav the SDK examples use.
    """
    try:
        client = _client()
        buf = io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio
        # SDK expects a file-like; give it a name so the codec is detected.
        buf.name = filename          # always set: a stale .name would mislabel it
        kwargs = {}
        codec = codec or codec_for(filename)
        if codec:
            kwargs["input_audio_codec"] = codec
        # STAGE_TIMEOUT_S applies here too -- see the note in generator.py.
        resp = client.speech_to_text.transcribe(
            file=buf,
            model=config.SARVAM_STT_MODEL,
            language_code=config.SARVAM_LANG,
            request_options={"timeout_in_seconds": int(config.STAGE_TIMEOUT_S)},
            **kwargs,
        )
        transcript = getattr(resp, "transcript", "") or ""
        lang = getattr(resp, "language_code", "") or config.SARVAM_LANG
        return transcript.strip(), lang
    except STTError:
        raise
    except Exception as e:               # network / SDK errors -> retryable
        raise STTError(f"Sarvam STT failed: {e}") from e
