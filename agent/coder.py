"""
Coder agent: given a Judge brief + current kernel → produces improved kernel code.

The Coder sees only the Judge's brief and current kernel code.
It does NOT see raw NCU metrics — that separation ensures focused reasoning.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import anthropic

from .judge import format_brief_for_coder

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "coder_system.txt").read_text()

# Required kernel entry point signature (used to validate output)
_REQUIRED_SIGNATURE = "def kernel("
_REQUIRED_PARAMS = [
    "routing_logits",
    "routing_bias",
    "hidden_states",
    "hidden_states_scale",
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm2_weights",
    "gemm2_weights_scale",
    "output",
    "local_expert_offset",
    "routed_scaling_factor",
]


def run_coder(
    brief: dict[str, Any],
    current_kernel: str,
    variant: str = "main",
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.3,
) -> str:
    """
    Run the Coder agent and return the new kernel source code.

    Args:
        brief: Judge's JSON brief.
        current_kernel: Current kernel.py source code.
        variant: "main" | "conservative" | "aggressive" — adjusts instructions.
        model: Claude model to use.
        temperature: Slightly higher for code diversity.

    Returns:
        Complete kernel.py source code as a string.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_message = _build_user_message(brief, current_kernel, variant)

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=temperature,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    code = _extract_code(raw)
    _validate_signature(code)
    return code


def _build_user_message(
    brief: dict[str, Any],
    current_kernel: str,
    variant: str,
) -> str:
    brief_text = format_brief_for_coder(brief)

    variant_instructions = {
        "main": (
            "Implement ALL changes from the Judge's top_3_changes list. "
            "Follow the brief precisely."
        ),
        "conservative": (
            "Implement ONLY the first change from top_3_changes. "
            "Make no other modifications — one surgical change only."
        ),
        "aggressive": (
            "Implement ALL changes from top_3_changes AND additionally "
            "increase num_stages by 1 in the autotune configs. "
            "Also consider restructuring the inner K loop for better pipelining."
        ),
    }

    return f"""{brief_text}

## Variant instruction: {variant.upper()}
{variant_instructions.get(variant, variant_instructions['main'])}

## Current kernel code
```python
{current_kernel}
```

Write the improved kernel.py now."""


def _extract_code(raw: str) -> str:
    """Extract Python code from the model's response."""
    # Try fenced code block
    m = re.search(r"```python\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Try any code block
    m = re.search(r"```\s*([\s\S]*?)```", raw)
    if m:
        return m.group(1).strip()

    # Fallback: assume entire response is code if it contains def kernel
    if "def kernel(" in raw:
        return raw.strip()

    raise ValueError(
        "Coder response did not contain a code block with `def kernel(`.\n"
        f"First 200 chars: {raw[:200]}"
    )


def _validate_signature(code: str) -> None:
    """Check that the kernel() entry point has the right parameters."""
    if _REQUIRED_SIGNATURE not in code:
        raise ValueError(f"Generated code missing `{_REQUIRED_SIGNATURE}`")

    # Extract the signature
    sig_start = code.index(_REQUIRED_SIGNATURE)
    sig_end = code.index(":", sig_start)
    sig_text = code[sig_start:sig_end]

    missing = [p for p in _REQUIRED_PARAMS if p not in sig_text]
    if missing:
        raise ValueError(
            f"Generated kernel() signature missing parameters: {missing}\n"
            f"Found signature: {sig_text}"
        )
