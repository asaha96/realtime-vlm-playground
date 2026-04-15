"""
Prompt construction and VLM-response parsing.

The VLM is asked to return a single compact JSON object. Keeping the prompt
small and the output schema tight is load-bearing: smaller prompt -> lower
time-to-first-token -> lower detection latency; structured output -> trivial
parsing with no regex heuristics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


VALID_ERROR_TYPES = {
    "wrong_action",
    "wrong_sequence",
    "safety_violation",
    "improper_technique",
    "other",
}


@dataclass
class VLMResult:
    observation: str = ""
    current_step_done: bool = False
    advance_by: int = 0  # how many steps ahead the student appears to be
    error_detected: bool = False
    error_type: str = "other"
    error_description: str = ""
    spoken_response: str = ""
    confidence: float = 0.0
    raw: str = ""


def build_prompt(
    task_name: str,
    current_step: Optional[Dict[str, Any]],
    next_step: Optional[Dict[str, Any]],
    step_after_next: Optional[Dict[str, Any]],
    last_observation: str,
) -> str:
    """
    Ask the VLM a targeted question anchored at a single current step.

    Instead of "identify which of N steps is happening" (unreliable from
    single frames), we ask the more tractable pair:
      - Is the current step finished?
      - Is the student already working on a later step (skipped ahead)?
    The orchestrator advances the state machine based on the answers.
    """
    cur = (
        f"[{current_step['step_id']}] {current_step['description']}"
        if current_step
        else "(procedure already complete — no current step)"
    )
    nxt = (
        f"[{next_step['step_id']}] {next_step['description']}"
        if next_step
        else "(none — no step after current)"
    )
    nxt2 = (
        f"[{step_after_next['step_id']}] {step_after_next['description']}"
        if step_after_next
        else "(none)"
    )
    last_obs_str = last_observation.strip() or "none"

    return (
        f"You are a real-time assistant watching a technician perform: {task_name}.\n"
        "\n"
        f"CURRENT expected step: {cur}\n"
        f"NEXT step (after current): {nxt}\n"
        f"STEP AFTER NEXT: {nxt2}\n"
        f"Previous observation: {last_obs_str}\n"
        "\n"
        "Respond with ONE JSON object and nothing else:\n"
        "{\n"
        '  "observation": "<one short sentence describing what the person is doing now>",\n'
        '  "current_step_done": <true if the CURRENT step has been completed (its '
        "terminal state is visible, or the student is now clearly moving on to a later "
        'action), else false>,\n'
        '  "advance_by": <integer 0-3: how many steps ahead of CURRENT the student '
        "appears to be doing right now. 0 = still on CURRENT (or idle/transitioning), "
        "1 = doing NEXT, 2 = doing STEP AFTER NEXT, 3 = skipped even further. If unsure, "
        'return 0>,\n'
        '  "error": {\n'
        '    "detected": <true|false>,\n'
        '    "type": "wrong_action|wrong_sequence|safety_violation|improper_technique|other",\n'
        '    "description": "<short>",\n'
        '    "spoken_response": "<what you would say to correct them, or empty>"\n'
        "  },\n"
        '  "confidence": <0.0..1.0>\n'
        "}\n"
        "\n"
        "Guidance:\n"
        "- current_step_done: set true when the physical action of the CURRENT step is "
        "visibly finished (terminal state present in the frame) OR when the student has "
        "clearly moved on to the NEXT / later action. Otherwise false. Do NOT set true on "
        "purely mid-action frames.\n"
        "- advance_by: your reading of how far ahead of CURRENT the student has moved. "
        "If current_step_done=false and the student is on CURRENT, return 0. If they're "
        "now on NEXT, return 1. Skipping ahead without completing CURRENT is allowed; "
        "the orchestrator will fill in the gaps.\n"
        "- error.detected: true only for clear mistakes (wrong tool, wrong object, unsafe "
        "action, damage). Being between two steps is NOT an error.\n"
        "- confidence: your certainty the JSON above is correct."
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> VLMResult:
    """
    Parse the model's JSON response. Tolerant of extra prose before/after the
    JSON object (some models wrap it in code fences or commentary).
    """
    result = VLMResult(raw=text or "")
    if not text:
        return result

    match = _JSON_RE.search(text)
    if not match:
        return result
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return result

    result.observation = str(data.get("observation", "")).strip()
    result.current_step_done = bool(data.get("current_step_done", False))
    ab = data.get("advance_by", 0)
    try:
        result.advance_by = max(0, min(3, int(ab)))
    except (TypeError, ValueError):
        result.advance_by = 0

    err = data.get("error") or {}
    if isinstance(err, dict):
        result.error_detected = bool(err.get("detected", False))
        etype = str(err.get("type", "other")).strip()
        result.error_type = etype if etype in VALID_ERROR_TYPES else "other"
        result.error_description = str(err.get("description", "")).strip()
        result.spoken_response = str(err.get("spoken_response", "")).strip()

    conf = data.get("confidence", 0.0)
    try:
        result.confidence = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        result.confidence = 0.0

    return result
