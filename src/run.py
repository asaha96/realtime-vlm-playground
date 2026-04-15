"""
VLM Orchestrator — real-time pipeline.

Receives frames and audio from the streaming harness and emits step_completion
and error_detected events through harness.emit_event().

Design principles:
  - Callbacks return immediately. VLM round-trips run on a thread pool so
    harness delivery timing is not inflated by API latency.
  - Frames are sampled, not streamed 1:1. A motion gate and rate limit cap
    API spend at ~1 call per 2.5 s of video by default.
  - The procedure state machine only advances on confident VLM completions;
    each step id is emitted at most once.
  - Audio is an independent error signal: instructor corrections ("no",
    "stop", "wait", ...) map straight to error_detected events.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import StreamingHarness
from src.data_loader import load_procedure_json, validate_procedure_format
from src.prompts import VLMResult, build_prompt, parse_response
from src.audio_asr import load_transcriber


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------


def call_vlm(
    api_key: str,
    frame_base64: str,
    prompt: str,
    model: str = "google/gemini-2.5-flash",
    stream: bool = False,
    max_tokens: int = 220,
    temperature: float = 0.1,
    timeout: float = 30.0,
) -> str:
    """Call a VLM on OpenRouter and return the response text."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Evaluation",
    }
    payload = {
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}"
                        },
                    },
                ],
            }
        ],
    }

    if stream:
        resp = requests.post(
            url, json=payload, headers=headers, stream=True, timeout=timeout
        )
        resp.raise_for_status()
        full_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    full_text += delta["content"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        return full_text

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Frame sampler
# ---------------------------------------------------------------------------


class FrameSampler:
    """
    Decide whether a given frame is worth sending to the VLM.

    A frame passes the gate when either:
      1. enough wall-clock video-time has elapsed since the last pick AND
         the scene has changed noticeably, OR
      2. we have not sent anything in a long while (heartbeat).

    Boosted mode temporarily relaxes the minimum interval after an audio
    correction keyword, so a tight video confirmation can follow the
    instructor's words.
    """

    def __init__(
        self,
        min_interval_sec: float = 2.5,
        heartbeat_sec: float = 8.0,
        motion_threshold: float = 6.0,
        boost_interval_sec: float = 1.0,
        boost_duration_sec: float = 6.0,
    ):
        self.min_interval_sec = min_interval_sec
        self.heartbeat_sec = heartbeat_sec
        self.motion_threshold = motion_threshold
        self.boost_interval_sec = boost_interval_sec
        self.boost_duration_sec = boost_duration_sec

        self._last_sent_t: float = -1e9
        self._last_small: Optional[np.ndarray] = None
        self._boost_until_t: float = -1e9

    def trigger_boost(self, video_t: float) -> None:
        self._boost_until_t = max(self._boost_until_t, video_t + self.boost_duration_sec)

    def _motion_score(self, frame: np.ndarray) -> float:
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._last_small is None:
            self._last_small = gray
            return float("inf")
        diff = cv2.absdiff(gray, self._last_small)
        score = float(diff.mean())
        self._last_small = gray
        return score

    def should_sample(self, frame: np.ndarray, video_t: float) -> bool:
        elapsed = video_t - self._last_sent_t
        min_interval = (
            self.boost_interval_sec
            if video_t < self._boost_until_t
            else self.min_interval_sec
        )
        motion = self._motion_score(frame)
        take = False
        if elapsed >= self.heartbeat_sec:
            take = True
        elif elapsed >= min_interval and motion >= self.motion_threshold:
            take = True
        if take:
            self._last_sent_t = video_t
        return take


# ---------------------------------------------------------------------------
# State tracker
# ---------------------------------------------------------------------------


@dataclass
class StepView:
    step_id: int
    description: str


class StateTracker:
    """
    Procedure state: completed-step set, current expected step, last VLM
    observation (fed back into the next prompt as context), and per-kind
    dedup windows.
    """

    def __init__(self, steps: List[Dict[str, Any]]):
        self._lock = threading.Lock()
        self._steps: List[StepView] = [
            StepView(step_id=int(s["step_id"]), description=str(s["description"]))
            for s in steps
        ]
        self._by_id: Dict[int, StepView] = {s.step_id: s for s in self._steps}
        self._completed: List[int] = []
        self._completed_set: set[int] = set()
        self._last_observation: str = ""
        self._last_error_t: float = -1e9

    @property
    def step_ids(self) -> List[int]:
        return [s.step_id for s in self._steps]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            completed_ids = list(self._completed)
            remaining = [
                {"step_id": s.step_id, "description": s.description}
                for s in self._steps
                if s.step_id not in self._completed_set
            ]
            return {
                "completed": completed_ids,
                "upcoming": remaining[:2],
                "last_observation": self._last_observation,
            }

    def step_description(self, step_id: int) -> str:
        s = self._by_id.get(step_id)
        return s.description if s else ""

    def record_observation(self, obs: str) -> None:
        if not obs:
            return
        with self._lock:
            self._last_observation = obs[:240]

    def accept_step_completion(self, step_id: int, confidence: float) -> bool:
        """
        Accept a VLM-reported completion if:
          - it's a valid step id,
          - not already completed,
          - and it's the next-expected step or one ahead (small reorderings).
        """
        if confidence < 0.55:
            return False
        with self._lock:
            if step_id not in self._by_id or step_id in self._completed_set:
                return False
            remaining_ids = [
                s.step_id for s in self._steps if s.step_id not in self._completed_set
            ]
            if not remaining_ids:
                return False
            allowed = set(remaining_ids[:2])
            if step_id not in allowed:
                return False
            self._completed.append(step_id)
            self._completed_set.add(step_id)
            return True

    def accept_error(self, video_t: float, dedup_sec: float = 4.0) -> bool:
        with self._lock:
            if video_t - self._last_error_t < dedup_sec:
                return False
            self._last_error_t = video_t
            return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """VLM orchestration pipeline. One instance per run."""

    # Gemini 2.5 Flash pricing (OpenRouter, Apr 2026 — approximate):
    # $0.075 / M input tokens, $0.30 / M output tokens. A JPEG image at 320x240
    # costs ~260 tokens; our prompt is ~320 tokens of text; output capped at
    # ~220 tokens. Rough per-call cost ≈ $0.0001. Treat as an upper-bound
    # sanity estimate rather than a billing oracle.
    COST_PER_CALL_USD = 0.0001

    def __init__(
        self,
        harness: StreamingHarness,
        api_key: str,
        procedure: Dict[str, Any],
        model: str = "google/gemini-2.5-flash",
        max_workers: int = 4,
        audio_model_size: str = "tiny",
    ):
        self.harness = harness
        self.api_key = api_key
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps: List[Dict[str, Any]] = procedure["steps"]
        self.model = model

        self.state = StateTracker(self.steps)
        self.sampler = FrameSampler()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.transcriber = load_transcriber(model_size=audio_model_size)

        # Stats
        self._stats_lock = threading.Lock()
        self.api_calls = 0
        self.api_errors = 0
        self.audio_chunks = 0

    # -- frames ------------------------------------------------------------

    def on_frame(self, frame: np.ndarray, timestamp_sec: float, frame_base64: str):
        if not self.sampler.should_sample(frame, timestamp_sec):
            return
        self.executor.submit(self._process_frame, frame_base64, timestamp_sec)

    def _process_frame(self, frame_base64: str, timestamp_sec: float) -> None:
        snapshot = self.state.snapshot()
        prompt = build_prompt(
            task_name=self.task_name,
            completed_ids=snapshot["completed"],
            upcoming=snapshot["upcoming"],
            last_observation=snapshot["last_observation"],
        )
        try:
            text = call_vlm(
                self.api_key,
                frame_base64,
                prompt,
                model=self.model,
                stream=True,
            )
            with self._stats_lock:
                self.api_calls += 1
        except Exception as e:
            with self._stats_lock:
                self.api_errors += 1
            print(f"  [vlm] call failed at {timestamp_sec:.1f}s: {e!s}")
            return

        result = parse_response(text)
        self.state.record_observation(result.observation)

        if result.step_completed is not None:
            if self.state.accept_step_completion(result.step_completed, result.confidence):
                self._emit_step(result, timestamp_sec)

        if result.error_detected:
            if self.state.accept_error(timestamp_sec):
                self._emit_error(result, timestamp_sec, source="video")

    def _emit_step(self, result: VLMResult, timestamp_sec: float) -> None:
        step_id = int(result.step_completed)
        desc = self.state.step_description(step_id) or result.observation
        event = {
            "timestamp_sec": round(float(timestamp_sec), 3),
            "type": "step_completion",
            "step_id": step_id,
            "confidence": round(float(result.confidence), 3),
            "description": desc,
            "source": "video",
            "vlm_observation": result.observation[:240],
        }
        try:
            self.harness.emit_event(event)
        except ValueError as e:
            print(f"  [emit] rejected step_completion: {e!s}")

    def _emit_error(
        self,
        result: VLMResult,
        timestamp_sec: float,
        source: str,
    ) -> None:
        confidence = max(0.4, float(result.confidence))
        event = {
            "timestamp_sec": round(float(timestamp_sec), 3),
            "type": "error_detected",
            "error_type": result.error_type,
            "severity": "warning",
            "confidence": round(min(1.0, confidence), 3),
            "description": result.error_description or result.observation or "unspecified error",
            "source": source,
            "vlm_observation": result.observation[:240],
            "spoken_response": result.spoken_response
            or "Hold on — check your last action against the procedure.",
        }
        try:
            self.harness.emit_event(event)
        except ValueError as e:
            print(f"  [emit] rejected error_detected: {e!s}")

    # -- audio -------------------------------------------------------------

    def on_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        if self.transcriber is None:
            return
        with self._stats_lock:
            self.audio_chunks += 1
        # Capture by value into the closure.
        self.executor.submit(self._process_audio, audio_bytes, start_sec, end_sec)

    def _process_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float) -> None:
        try:
            tx = self.transcriber.transcribe(audio_bytes, chunk_start_sec=start_sec)
        except Exception as e:
            print(f"  [audio] transcription failed at {start_sec:.1f}s: {e!s}")
            return

        if not tx.hits:
            return

        # One error per chunk is enough — take the earliest hit.
        hit = min(tx.hits, key=lambda h: h.segment_start)

        # Ground truth errors are timestamped at the START of the wrong
        # action, which precedes the instructor's correction by ~2 s. Bias
        # a bit earlier but never before the chunk's start.
        est_t = max(start_sec, hit.segment_start - 2.0)

        # Video side should take a closer look while the student is still
        # mid-error.
        self.sampler.trigger_boost(est_t)

        if not self.state.accept_error(est_t):
            return

        snapshot = self.state.snapshot()
        stub = VLMResult(
            observation=snapshot["last_observation"],
            error_detected=True,
            error_type="wrong_action",
            error_description=(
                f"Instructor correction heard: '{hit.text[:120]}'"
                if hit.text
                else "Instructor correction heard."
            ),
            spoken_response="",
            confidence=0.6,
        )
        self._emit_error(stub, est_t, source="audio")

    # -- lifecycle ---------------------------------------------------------

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)

    def stats_summary(self) -> Dict[str, Any]:
        with self._stats_lock:
            calls = self.api_calls
            errs = self.api_errors
            audio = self.audio_chunks
        return {
            "model_used": self.model,
            "api_calls": calls,
            "api_errors": errs,
            "audio_chunks_processed": audio,
            "total_api_cost_usd_estimate": round(calls * self.COST_PER_CALL_USD, 4),
            "audio_transcription_enabled": self.transcriber is not None,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Orchestrator Pipeline")
    parser.add_argument("--procedure", required=True, help="Path to procedure JSON")
    parser.add_argument("--video", required=True, help="Path to video MP4 (with audio)")
    parser.add_argument("--output", default="output/events.json", help="Output JSON path")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (1.0 = real-time)")
    parser.add_argument("--frame-fps", type=float, default=2.0,
                        help="Frames per second delivered to pipeline (default: 2)")
    parser.add_argument("--audio-chunk-sec", type=float, default=5.0,
                        help="Audio chunk duration in seconds (default: 5)")
    parser.add_argument("--model", default="google/gemini-2.5-flash",
                        help="OpenRouter model string for frame VLM calls")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Thread pool size for VLM + audio workers")
    parser.add_argument("--audio-model", default="tiny",
                        help="faster-whisper model size (tiny|base|small)")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only")
    args = parser.parse_args()

    print("=" * 60)
    print("  VLM ORCHESTRATOR")
    print("=" * 60)
    print()

    procedure = load_procedure_json(args.procedure)
    validate_procedure_format(procedure)
    task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
    print(f"  Procedure: {task_name} ({len(procedure['steps'])} steps)")
    print(f"  Video:     {args.video}")
    print(f"  Speed:     {args.speed}x")
    print(f"  Model:     {args.model}")
    print()

    if args.dry_run:
        if not Path(args.video).exists():
            print(f"  WARNING: Video not found: {args.video}")
            print("  [DRY RUN] Procedure validated. Video not checked (file missing).")
        else:
            print("  [DRY RUN] Inputs validated. Skipping pipeline.")
        return

    if not Path(args.video).exists():
        print(f"  ERROR: Video not found: {args.video}")
        sys.exit(1)

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("  ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    harness = StreamingHarness(
        video_path=args.video,
        procedure_path=args.procedure,
        speed=args.speed,
        frame_fps=args.frame_fps,
        audio_chunk_sec=args.audio_chunk_sec,
    )

    pipeline = Pipeline(
        harness,
        api_key,
        procedure,
        model=args.model,
        max_workers=args.max_workers,
        audio_model_size=args.audio_model,
    )

    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)

    t0 = time.monotonic()
    try:
        results = harness.run()
    finally:
        pipeline.shutdown()
    wall = time.monotonic() - t0

    harness.save_results(results, args.output)

    stats = pipeline.stats_summary()
    stats_path = Path(args.output).with_suffix(".stats.json")
    stats_payload = {
        **stats,
        "video_source": args.video,
        "procedure_path": args.procedure,
        "speed": args.speed,
        "wall_duration_sec": round(wall, 2),
        "events_emitted": len(results.events),
        "mean_detection_delay_sec": results.mean_detection_delay_sec,
        "max_detection_delay_sec": results.max_detection_delay_sec,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats_payload, f, indent=2)

    print()
    print(f"  Output:       {args.output}")
    print(f"  Stats:        {stats_path}")
    print(f"  Events:       {len(results.events)}")
    print(f"  API calls:    {stats['api_calls']} "
          f"(errors: {stats['api_errors']})")
    print(f"  Est. cost:    ${stats['total_api_cost_usd_estimate']:.4f}")
    print(f"  Audio ASR:    {'on' if stats['audio_transcription_enabled'] else 'off'}")
    print()

    if not results.events:
        print("  WARNING: No events detected.")


if __name__ == "__main__":
    main()
