"""
Judge agent: converts NCU metrics + roofline analysis → structured JSON brief.

The Judge sees raw profiling data and produces a focused brief for the Coder.
It never sees kernel code — it only knows about hardware metrics.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import anthropic

from .ncu_parser import format_metrics_for_prompt
from .roofline import format_roofline_for_prompt

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "judge_system.txt").read_text()


def run_judge(
    metrics: dict[str, Any],
    roofline: dict[str, Any],
    failure_context: str = "",
    model: str = "claude-opus-4-6",
    temperature: float = 0.1,
) -> dict[str, Any]:
    """
    Run the Judge agent and return a parsed JSON brief.

    Args:
        metrics: Parsed NCU metrics dict (from ncu_parser.parse_ncu_output).
        roofline: Roofline analysis dict (from roofline.compute_roofline).
        failure_context: Formatted string of recent regression history.
        model: Claude model to use.
        temperature: Low temperature for reproducible analysis.

    Returns:
        Parsed JSON brief dict with keys: bound, primary_bottleneck, root_cause,
        top_3_changes, do_not_try, target_metric, metrics_summary.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_message = _build_user_message(metrics, roofline, failure_context)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    return _parse_brief(raw)


def _build_user_message(
    metrics: dict[str, Any],
    roofline: dict[str, Any],
    failure_context: str,
) -> str:
    parts = [
        "## Profiling Results\n",
        format_metrics_for_prompt(metrics),
        "\n",
        format_roofline_for_prompt(roofline),
    ]

    if failure_context:
        parts.extend([
            "\n\n## Previous Regressions (DO NOT repeat these approaches):",
            failure_context,
        ])

    parts.append(
        "\n\nBased on the above, produce your JSON brief for the Coder."
    )

    return "\n".join(parts)


def _parse_brief(raw: str) -> dict[str, Any]:
    """Extract and parse JSON from the model's response."""
    # Try extracting from code block first
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if json_match:
        raw = json_match.group(1).strip()

    # Try extracting bare JSON object
    obj_match = re.search(r"\{[\s\S]*\}", raw)
    if obj_match:
        raw = obj_match.group(0)

    try:
        brief = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge returned invalid JSON: {e}\n\nRaw response:\n{raw}")

    # Validate required keys
    required = ["bound", "primary_bottleneck", "root_cause", "top_3_changes", "do_not_try"]
    missing = [k for k in required if k not in brief]
    if missing:
        raise ValueError(f"Judge brief missing required keys: {missing}")

    # Ensure top_3_changes has at least 1 entry
    if not brief["top_3_changes"]:
        raise ValueError("Judge brief has empty top_3_changes")

    return brief


def format_brief_for_coder(brief: dict[str, Any]) -> str:
    """Format the Judge's brief as a readable block for the Coder prompt."""
    lines = [
        "## Performance Brief from Judge",
        f"  Bound: {brief.get('bound', 'UNKNOWN')}",
        f"  Primary bottleneck: {brief.get('primary_bottleneck', '')}",
        f"  Root cause: {brief.get('root_cause', '')}",
        "",
        "  Top 3 changes to make:",
    ]
    for i, change in enumerate(brief.get("top_3_changes", []), 1):
        lines.append(
            f"    {i}. [{change.get('confidence','?')}] {change.get('change', '')}"
            + (f"\n       Expected: {change.get('expected_impact', '')}" if change.get("expected_impact") else "")
        )

    if brief.get("do_not_try"):
        lines.append("\n  Do NOT try these (caused regressions):")
        for item in brief["do_not_try"]:
            lines.append(f"    - {item}")

    if brief.get("target_metric"):
        lines.append(f"\n  Target metric: {brief['target_metric']}")

    metrics_summary = brief.get("metrics_summary", {})
    if metrics_summary:
        lines.append("\n  Key metrics (for reference):")
        for k, v in metrics_summary.items():
            if v is not None:
                lines.append(f"    {k}: {v}")

    return "\n".join(lines)
