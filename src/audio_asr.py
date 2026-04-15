"""
Audio transcription + keyword detection.

Wraps faster-whisper with a thin interface:
  - load the tiny model once
  - transcribe a raw 16 kHz mono PCM chunk
  - scan the transcript for instructor-correction keywords

If faster-whisper is unavailable at import time, the loader returns None and
the Pipeline degrades to video-only (no crash).
"""

from __future__ import annotations

import re
import wave
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


CORRECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bstop\b", re.IGNORECASE),
    re.compile(r"\bwait\b", re.IGNORECASE),
    re.compile(r"\bwrong\b", re.IGNORECASE),
    re.compile(r"\bdon'?t\b", re.IGNORECASE),
    re.compile(r"\bnot (?:that|like|the)\b", re.IGNORECASE),
    re.compile(r"\bother one\b", re.IGNORECASE),
    re.compile(r"\bdifferent\b", re.IGNORECASE),
    re.compile(r"\bhold on\b", re.IGNORECASE),
    re.compile(r"\bnope\b", re.IGNORECASE),
]


@dataclass
class KeywordHit:
    keyword: str
    segment_start: float
    segment_end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str = ""
    segments: List[Tuple[float, float, str]] = None  # (start, end, text)
    hits: List[KeywordHit] = None


def load_transcriber(model_size: str = "tiny") -> Optional["Transcriber"]:
    """Load faster-whisper. Returns None if the dep is missing."""
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except Exception as e:
        print(f"  [audio] faster-whisper unavailable ({e!s}); audio disabled.")
        return None
    try:
        return Transcriber(model_size=model_size)
    except Exception as e:
        print(f"  [audio] failed to initialise Whisper model: {e!s}; audio disabled.")
        return None


class Transcriber:
    """Thin wrapper around faster-whisper for short PCM chunks."""

    def __init__(self, model_size: str = "tiny"):
        from faster_whisper import WhisperModel

        # int8 on CPU is the default install target and keeps latency low.
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    @staticmethod
    def _pcm_to_wav_path(pcm: bytes, sample_rate: int = 16000) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)
        return tmp.name

    def transcribe(self, pcm: bytes, chunk_start_sec: float) -> TranscriptionResult:
        """Transcribe one PCM chunk and scan for correction keywords."""
        if not pcm:
            return TranscriptionResult(text="", segments=[], hits=[])
        wav_path = self._pcm_to_wav_path(pcm)
        try:
            segments_iter, _info = self._model.transcribe(
                wav_path,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                beam_size=1,
                no_speech_threshold=0.5,
            )
            segments: List[Tuple[float, float, str]] = []
            pieces: List[str] = []
            hits: List[KeywordHit] = []
            for seg in segments_iter:
                seg_start = chunk_start_sec + float(seg.start)
                seg_end = chunk_start_sec + float(seg.end)
                text = (seg.text or "").strip()
                if not text:
                    continue
                segments.append((seg_start, seg_end, text))
                pieces.append(text)
                for pat in CORRECTION_PATTERNS:
                    m = pat.search(text)
                    if m:
                        hits.append(
                            KeywordHit(
                                keyword=m.group(0),
                                segment_start=seg_start,
                                segment_end=seg_end,
                                text=text,
                            )
                        )
                        break
            return TranscriptionResult(
                text=" ".join(pieces),
                segments=segments,
                hits=hits,
            )
        finally:
            Path(wav_path).unlink(missing_ok=True)
