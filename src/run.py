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
    Strict monotonic step progression.

    The pipeline asks the VLM a single focused question — "is the current
    step done, and how far ahead of current is the student?" — and the
    state machine advances `current_step_idx` accordingly. Each advance
    emits step_completion for the step that was just left behind. This
    avoids relying on the VLM to disambiguate visually similar actions by
    their step id.

    Out-of-order VLM responses (common at speed>1) are dropped via the
    monotonic timestamp guard.
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
        # Index into self._steps of the currently expected step.
        self._current_idx: int = 0
        # Video-time when we entered the current step.
        self._current_idx_since_t: float = 0.0
        # Sliding window of recent readings: (done, advance_by) tuples.
        self._read_window: List[tuple] = []
        # Monotonic timestamp guard — VLM responses can return out-of-order at
        # speed>1 with multiple in-flight workers; we ignore stale observations.
        self._last_advance_t: float = -1e9

    @property
    def step_ids(self) -> List[int]:
        return [s.step_id for s in self._steps]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            current = (
                self._steps[self._current_idx]
                if self._current_idx < len(self._steps)
                else None
            )
            nxt = (
                self._steps[self._current_idx + 1]
                if self._current_idx + 1 < len(self._steps)
                else None
            )
            nxt2 = (
                self._steps[self._current_idx + 2]
                if self._current_idx + 2 < len(self._steps)
                else None
            )

            def _pack(s):
                return {"step_id": s.step_id, "description": s.description} if s else None

            return {
                "current": _pack(current),
                "next": _pack(nxt),
                "next2": _pack(nxt2),
                "last_observation": self._last_observation,
                "procedure_complete": current is None,
            }

    def step_description(self, step_id: int) -> str:
        s = self._by_id.get(step_id)
        return s.description if s else ""

    def record_observation(self, obs: str) -> None:
        if not obs:
            return
        with self._lock:
            self._last_observation = obs[:240]

    def advance(
        self,
        current_step_done: bool,
        advance_by: int,
        video_t: float,
        confidence: float,
    ) -> List[int]:
        """
        Apply a VLM reading to the state machine. Returns a list of step_ids
        that were just marked completed at `video_t`.

        Advance rules (primary signal = advance_by):
          - require MIN_DWELL seconds in the current step before any advance
            (prevents over-eager zero-to-N jumps),
          - require >= 2 of last 3 readings to have advance_by >= 1 OR
            current_step_done=True with confidence >= CONF_MIN,
          - then advance by the max(advance_by) across the window (filling
            intermediate skipped steps in a single emission).
        """
        CONF_MIN = 0.60
        WIN = 3
        THRESHOLD = 2
        MIN_DWELL = 4.0
        MIN_ADVANCE_GAP = 3.0
        emitted: List[int] = []
        with self._lock:
            if video_t <= self._last_advance_t:
                return emitted
            if self._current_idx >= len(self._steps):
                return emitted

            if confidence >= CONF_MIN:
                self._read_window.append((bool(current_step_done), int(advance_by)))
                if len(self._read_window) > WIN:
                    self._read_window.pop(0)

            dwell_ok = (video_t - self._current_idx_since_t) >= MIN_DWELL
            gap_ok = (video_t - self._last_advance_t) >= MIN_ADVANCE_GAP
            if not (dwell_ok and gap_ok):
                return emitted

            # How many readings suggest "move forward" (done OR advance_by>=1)?
            positive = sum(
                1 for (d, a) in self._read_window if d or a >= 1
            )
            if positive < THRESHOLD:
                return emitted

            # Pick the max advance_by in the window; floor at 1 if only done
            # signals are present.
            max_adv = max((a for (_, a) in self._read_window), default=0)
            n = max(1, min(max_adv, len(self._steps) - self._current_idx))

            for _ in range(n):
                step = self._steps[self._current_idx]
                if step.step_id not in self._completed_set:
                    self._completed.append(step.step_id)
                    self._completed_set.add(step.step_id)
                    emitted.append(step.step_id)
                self._current_idx += 1
            self._current_idx_since_t = video_t
            self._read_window.clear()
            self._last_advance_t = video_t
            return emitted

    def finalize(self, video_t: float) -> List[int]:
        """
        At end-of-video, emit completion for the current pending step if it
        is the LAST remaining step (the last step rarely produces a 'next
        activity' transition, so a one-off finalize is worth the risk). We
        do not fire finalize mid-procedure — a bad final emission would be
        a precision hit with no corresponding GT match.
        """
        emitted: List[int] = []
        with self._lock:
            if self._current_idx >= len(self._steps):
                return emitted
            step = self._steps[self._current_idx]
            if step.step_id in self._completed_set:
                return emitted
            is_last = self._current_idx == len(self._steps) - 1
            if not is_last:
                return emitted
            self._completed.append(step.step_id)
            self._completed_set.add(step.step_id)
            self._current_idx += 1
            emitted.append(step.step_id)
            return emitted

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
        verbose: bool = False,
    ):
        self.harness = harness
        self.api_key = api_key
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps: List[Dict[str, Any]] = procedure["steps"]
        self.model = model
        self.verbose = verbose

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
        if snapshot["procedure_complete"]:
            # Nothing useful to ask; skip the call and save cost.
            return
        prompt = build_prompt(
            task_name=self.task_name,
            current_step=snapshot["current"],
            next_step=snapshot["next"],
            step_after_next=snapshot["next2"],
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
        if self.verbose:
            print(
                f"  [vlm @ {timestamp_sec:6.1f}s] done={result.current_step_done} "
                f"advance={result.advance_by} conf={result.confidence:.2f} "
                f"err={result.error_detected} obs={result.observation[:80]!r}"
            )

        for sid in self.state.advance(
            result.current_step_done,
            result.advance_by,
            timestamp_sec,
            result.confidence,
        ):
            self._emit_step_by_id(sid, result, timestamp_sec)

        # Errors are costly when wrong (40% of the automated score) so we bias
        # toward precision: require near-certainty AND that the student is
        # not merely ahead-of-schedule AND that the model wasn't signalling
        # "done" in the same breath (the model sometimes flags "different
        # from expected" as an error even when it's actually the next step).
        if (
            result.error_detected
            and result.advance_by == 0
            and not result.current_step_done
            and result.confidence >= 0.90
        ):
            if self.state.accept_error(timestamp_sec):
                self._emit_error(result, timestamp_sec, source="video")

    def _emit_step_by_id(
        self,
        step_id: int,
        result: VLMResult,
        timestamp_sec: float,
    ) -> None:
        desc = self.state.step_description(step_id) or result.observation
        event = {
            "timestamp_sec": round(float(timestamp_sec), 3),
            "type": "step_completion",
            "step_id": int(step_id),
            "confidence": round(max(0.5, float(result.confidence)), 3),
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

    # Suppress audio errors in the first N seconds — both training clips we
    # inspected start with the instructor setting up ("no, I didn't say
    # record..."), not correcting the student. GT never marks an error in
    # this pre-action window.
    AUDIO_ERROR_IGNORE_SEC = 10.0

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

        if hit.segment_start < self.AUDIO_ERROR_IGNORE_SEC:
            return

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

    def drain(self) -> None:
        """Wait for all in-flight VLM/audio jobs, keep the executor alive."""
        # ThreadPoolExecutor has no public "join" — swap it for a fresh one
        # after draining so shutdown() can be called cleanly afterwards.
        old = self.executor
        old.shutdown(wait=True)
        self.executor = ThreadPoolExecutor(max_workers=1)

    def finalize_at(self, video_duration_sec: float) -> List[Dict[str, Any]]:
        """Mark any still-pending last step as completed at end-of-video."""
        events: List[Dict[str, Any]] = []
        for step_id in self.state.finalize(video_duration_sec):
            desc = self.state.step_description(step_id)
            events.append({
                "timestamp_sec": round(float(video_duration_sec), 3),
                "type": "step_completion",
                "step_id": int(step_id),
                "confidence": 0.6,
                "description": desc,
                "source": "video",
                "vlm_observation": "End-of-video finalisation.",
                "detection_delay_sec": 0.0,
            })
        return events

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
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-VLM-call diagnostic lines")
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
        verbose=args.verbose,
    )

    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)

    t0 = time.monotonic()
    try:
        results = harness.run()
        # Drain in-flight VLM calls so late completions still reach the
        # event log before finalisation.
        pipeline.drain()
        # End-of-video: mark the last running activity complete. Use the
        # video_duration as the emission timestamp (best approximation of
        # "done" when we never see a transition out of it).
        for final in pipeline.finalize_at(results.video_duration_sec):
            results.events.append(final)
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
