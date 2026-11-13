"""Local speech-to-text with faster-whisper (CPU or CUDA; no cloud calls).

The model loads on first use and stays in memory. Install with `pip install faster-whisper`;
without it, `available()` is False and the UI hides the microphone button.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import threading

from src.logger import get_logger

_log = get_logger(__name__)

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "base")
_model = None
_model_name: str | None = None
_lock = threading.Lock()


def available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def _get(model_name: str):
    global _model, _model_name
    with _lock:
        if _model is None or _model_name != model_name:
            from faster_whisper import WhisperModel
            _log.info("Loading Whisper model %r", model_name)
            _model = WhisperModel(model_name, device="auto", compute_type="int8")
            _model_name = model_name
        return _model


def transcribe(audio: bytes, suffix: str = ".webm", model_name: str | None = None) -> str:
    """Transcribe an audio clip (any format PyAV can decode: webm, ogg, wav, mp4…)."""
    if not available():
        raise RuntimeError("Voice input needs faster-whisper: pip install faster-whisper")
    model = _get(model_name or DEFAULT_MODEL)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        segments, _info = model.transcribe(path, vad_filter=True, beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(path)
