# VLM Orchestrator — Technical Report

## Architecture

The pipeline is a single `Pipeline` class in `src/run.py` with four
collaborators kept in the same file for easy reading:

```
on_frame ─▶ FrameSampler (motion + rate limit) ─▶ ThreadPoolExecutor(4)
                                                      │
                                                      ▼
                                          call_vlm (Gemini 2.5 Flash, SSE)
                                                      │
                                                      ▼
                              parse_response ─▶ StateTracker.advance ─▶ emit
                                                      │
on_audio ─▶ faster-whisper (tiny) ─▶ keyword scan ────┘ ─▶ emit (source=audio)
```

Frame and audio callbacks are non-blocking: both enqueue work onto a
4-worker thread pool so the harness delivery loop never waits on a VLM
round-trip. All state mutations go through a lock.

## Frame sampling

Gemini Flash round-trip is ~1.5–3 s; sending every frame would blow the
budget and the worker pool. `FrameSampler` sends a frame only when
either:

1. the wall-clock-in-video-time since the last send is >= `min_interval`
   (2.5 s) AND a downsampled-grayscale frame-diff exceeds `motion_threshold`
   (mean absolute diff >= 6 / 255), or
2. `heartbeat_sec` (8 s) has elapsed without a send.

On an audio correction keyword we temporarily drop `min_interval` to 1 s
for 6 s of video time so a tight video confirmation can follow. This cuts
API calls from ~860 per clip (every frame) to ~80 per clip (-90%) while
keeping resolution around step transitions.

## State machine

A single-frame VLM cannot reliably pick "this is step 4" vs "step 5" when
the actions are visually similar (insert vs remove battery). Instead of
asking the model to name the step, the prompt anchors on a concrete
current step and asks two simpler questions:

* `current_step_done` — is the current step visibly finished, OR has the
  student clearly moved on?
* `advance_by` — how many steps ahead of `current` is the student doing
  right now (0–3)?

The state machine then advances strictly monotonically. An advance fires
when **≥2 of the last 3 readings** agree that movement forward is
happening, after a minimum dwell of 4 s in the current step. If
`advance_by >= 1`, we fill in skipped intermediate steps in a single emit.
At end-of-video we finalise only the last remaining step (never
mid-procedure, to preserve precision).

VLM responses can return out-of-order at speed>1 with 4 parallel workers;
a monotonic timestamp guard ignores stale observations.

## Audio path

`faster-whisper` (`tiny`, int8 CPU) transcribes each 5 s PCM chunk in
~500 ms. Chunks are processed on the same thread pool. The transcript is
scanned for short imperative phrases (`^no[ ,.]`, `stop`, `wait`, `wrong`,
`don't`, `not that`, `hold on`, ...). A match within a ≤10-word utterance
after `t >= 10 s` (ignores pre-action instructor setup) fires an
`error_detected` event with `source="audio"`, timestamped 2 s before the
keyword (biased toward the GT convention of marking errors at the *start*
of the wrong action). Keyword hits also trigger a 6-second frame-sampler
"boost" so the video side gets a denser look.

On R066 (circuit-breaker) audio alone drove the entire error F1 — the
video channel is deliberately biased toward precision.

## Model selection and cost

A single model tier: **`google/gemini-2.5-flash`** via OpenRouter, streamed
with `stream=true` for lower TTFT. Prompt ≈ 320 tokens, image ≈ 260
tokens, output capped at 220 tokens, temperature 0.1. Measured per-run
cost on 3-minute clips: 70–90 API calls, **$0.008–0.009 per clip**.
Expected budget for 15 training clips: ~$0.14.

I considered escalating to `gemini-2.5-pro` or `claude-sonnet-4` on
ambiguous frames but the gain in accuracy was not worth the 5-10x cost
hit, and the state machine's sliding-window advance rule already absorbs
noise from the cheaper model.

## Measured results (speed=1.0, tolerance ±5 s)

| Clip | step F1 | err F1 | mean latency |
|---|---|---|---|
| R066 circuit-breaker | 0.45 | 0.18 | 1.9 s |
| R073 GoPro | 0.25 | — | 1.9 s |
| R142 RAM | 0.08 | — | 2.0 s |

The combined score on R066 is ≈ 0.42 (0.40·0.45 + 0.40·0.18 + 0.20·0.81).
Latency scores are consistently >0.8 because VLM calls complete in ~2 s
at 1x and we timestamp emissions at the **frame timestamp** (not
response-arrival wall time), so the harness' `detection_delay` measures
only the API round-trip.

### Where accuracy is lost

Single-frame disambiguation on visually similar actions (insert vs remove)
is the main miss — the state machine requires two confirming frames, and
by then the student can have drifted 5–15 s past the GT end time. R142
(RAM install) is particularly affected because four of the 13 steps are
"insert first RAM card", "insert second RAM card", ... — identical visual
actions with only positional differences the frame crop can't resolve.

## Bonus: bidirectional streaming redesign

With a bidirectional streaming API (Gemini Live / an equivalent with
continuous frame+audio input and streamed event output), the pipeline
simplifies substantially:

1. Remove the `ThreadPoolExecutor` and frame-sampling gate. Feed
   downsampled frames (1–2 fps) and audio chunks continuously; the model
   holds state across frames and can emit `step_completion` /
   `error_detected` tokens the instant it is confident.
2. Replace the per-frame prompt with a short system prompt that carries
   the procedure once at session start. State persists on the server
   side, so the client stops re-sending "already completed: …" context
   with every call (saves ~60% of per-call tokens).
3. Audio becomes a native modality rather than a separate whisper pass;
   instructor corrections and user actions are correlated by the model
   inside its attention rather than stitched together in our
   orchestrator.
4. Detection delay drops to the streaming equivalent of time-to-first-
   token (typically 300–800 ms) — a latency score > 0.95 becomes
   achievable without any frame-sampling trickery.
5. Error recall goes up because cross-modal fusion is native: the model
   can link the visual "grabbing red toolbox" frame to the audio
   "no, that's the wrong one" ~3 s later without us writing keyword
   regexes.

The client-side work shrinks to a thin adapter (open the session, push
frames and audio, surface emitted events) plus prompt engineering on the
procedure wrapper. Most of the complexity in `src/run.py` (sampler,
state machine, dedup, thread pool) evaporates.

## Repository layout (what I wrote)

* `src/run.py` — Pipeline, FrameSampler, StateTracker, CLI.
* `src/prompts.py` — prompt template + `VLMResult` parser.
* `src/audio_asr.py` — faster-whisper wrapper with keyword scan.
* `requirements.txt` — added `faster-whisper>=1.0`.
* `Makefile` — updated default `VIDEO=` path and added a `SPEED=` knob.

Not modified: `src/harness.py`, `src/evaluator.py`, `src/dashboard.py`,
`src/data_loader.py`.
