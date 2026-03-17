"""
Parallel variant generation: runs 3 Coder calls concurrently via asyncio.

Variants:
  main         — follow the Judge brief exactly
  conservative — only the first change, nothing else
  aggressive   — all changes + increase num_stages
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import anthropic

from .coder import _build_user_message, _extract_code, _validate_signature, _SYSTEM_PROMPT

VARIANTS = ["main", "conservative", "aggressive"]


async def _run_coder_async(
    client: anthropic.AsyncAnthropic,
    brief: dict[str, Any],
    current_kernel: str,
    variant: str,
    model: str,
    temperature: float,
) -> tuple[str, str]:
    """Run a single Coder call asynchronously. Returns (variant, code)."""
    user_message = _build_user_message(brief, current_kernel, variant)
    response = await client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=temperature,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip()
    code = _extract_code(raw)
    _validate_signature(code)
    return variant, code


async def generate_variants_async(
    brief: dict[str, Any],
    current_kernel: str,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.3,
) -> dict[str, str]:
    """
    Launch 3 Coder API calls concurrently and return all variant codes.

    Returns:
        dict mapping variant name → kernel source code.
        On individual variant failure, logs the error and omits that variant.
    """
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    tasks = [
        _run_coder_async(client, brief, current_kernel, v, model, temperature)
        for v in VARIANTS
    ]

    results: dict[str, str] = {}
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    for variant, outcome in zip(VARIANTS, outcomes):
        if isinstance(outcome, Exception):
            print(f"  [parallel_explore] Variant '{variant}' failed: {outcome}")
        else:
            v_name, code = outcome
            results[v_name] = code

    if not results:
        raise RuntimeError("All 3 parallel coder variants failed.")

    return results


def generate_variants(
    brief: dict[str, Any],
    current_kernel: str,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.3,
) -> dict[str, str]:
    """Synchronous wrapper around generate_variants_async."""
    return asyncio.run(
        generate_variants_async(brief, current_kernel, model, temperature)
    )
