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
    step_completed: Optional[int] = None
    error_detected: bool = False
    error_type: str = "other"
    error_description: str = ""
    spoken_response: str = ""
    confidence: float = 0.0
    raw: str = ""


def build_prompt(
    task_name: str,
    completed_ids: List[int],
    upcoming: List[Dict[str, Any]],
    last_observation: str,
) -> str:
    """
    Build the per-frame prompt. `upcoming` is the next 1-3 expected steps
    (already trimmed by the caller) — we show the next two at most to keep
    tokens low.
    """
    next1 = upcoming[0] if upcoming else None
    next2 = upcoming[1] if len(upcoming) > 1 else None

    completed_str = (
        ", ".join(str(i) for i in completed_ids[-4:]) if completed_ids else "none"
    )
    next1_str = (
        f"[{next1['step_id']}] {next1['description']}"
        if next1
        else "none (procedure complete)"
    )
    next2_str = (
        f"[{next2['step_id']}] {next2['description']}" if next2 else "none"
    )
    last_obs_str = last_observation.strip() or "none"

    return (
        f"You are a real-time assistant watching a technician perform: {task_name}.\n"
        f"Recently completed step ids: {completed_str}.\n"
        f"Expected NEXT step: {next1_str}\n"
        f"The step AFTER that:  {next2_str}\n"
        f"Previous observation: {last_obs_str}\n"
        "\n"
        "Look at this frame and respond with ONE JSON object and nothing else. "
        "Schema:\n"
        "{\n"
        '  "observation": "<one short sentence describing what the person is doing now>",\n'
        '  "step_completed": <step_id that JUST finished in this frame, or null>,\n'
        '  "error": {\n'
        '    "detected": <true|false>,\n'
        '    "type": "wrong_action|wrong_sequence|safety_violation|improper_technique|other",\n'
        '    "description": "<short>",\n'
        '    "spoken_response": "<what you would say to correct them, or empty>"\n'
        "  },\n"
        '  "confidence": <0.0..1.0>\n'
        "}\n"
        "Rules: only set step_completed when the action for that step is clearly "
        "finished in THIS frame. Do not repeat a step id that is already in the "
        "completed list. Only flag an error if the person is clearly doing "
        "something wrong for the expected next step."
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
    sc = data.get("step_completed")
    if isinstance(sc, int):
        result.step_completed = sc
    elif isinstance(sc, str) and sc.strip().isdigit():
        result.step_completed = int(sc.strip())

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
