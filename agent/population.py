"""
OpenEvolve-style kernel population.

Maintains a pool of top-K performing kernels and supports crossbreeding
two parents via Claude to synthesize a hybrid kernel.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import anthropic

from .tree import KernelNode

_SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "crossbreed_system.txt").read_text()


class KernelPopulation:
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.members: list[KernelNode] = []

    def update(self, node: KernelNode) -> None:
        """Add a new node; evict the worst performer if over capacity."""
        if node.speedup_factor is None:
            return
        # Avoid duplicates
        if any(m.id == node.id for m in self.members):
            return
        self.members.append(node)
        # Keep only the best max_size
        self.members.sort(key=lambda n: n.speedup_factor or 0.0, reverse=True)
        self.members = self.members[: self.max_size]

    def crossbreed(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.5,
    ) -> str | None:
        """
        Pick 2 members from the population and ask Claude to synthesize a hybrid.

        Returns:
            New kernel source code, or None if fewer than 2 members exist.
        """
        if len(self.members) < 2:
            return None

        parent_a = self.members[0]  # best
        parent_b = random.choice(self.members[1:])  # random from remaining top-K

        user_message = _build_crossbreed_message(parent_a, parent_b)

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            temperature=temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        return _extract_code(raw)

    def summary(self) -> str:
        if not self.members:
            return "Population: empty"
        lines = [f"Population ({len(self.members)} members):"]
        for i, m in enumerate(self.members):
            lines.append(
                f"  {i+1}. {m.id} @ {m.speedup_factor:.3f}x "
                f"({m.latency_ms:.3f}ms, iter {m.iteration})"
            )
        return "\n".join(lines)


def _build_crossbreed_message(parent_a: KernelNode, parent_b: KernelNode) -> str:
    def _metrics_str(n: KernelNode) -> str:
        m = n.metrics
        parts = []
        for k in ["achieved_occupancy_pct", "tensor_active_pct", "l2_hit_rate_pct", "dominant_stall"]:
            if k in m:
                parts.append(f"{k}={m[k]}")
        return ", ".join(parts) if parts else "no metrics available"

    return f"""## Kernel A (score: {parent_a.speedup_factor:.3f}x speedup, {parent_a.latency_ms:.3f}ms)
Metrics: {_metrics_str(parent_a)}

```python
{parent_a.kernel_code}
```

## Kernel B (score: {parent_b.speedup_factor:.3f}x speedup, {parent_b.latency_ms:.3f}ms)
Metrics: {_metrics_str(parent_b)}

```python
{parent_b.kernel_code}
```

Kernel A is the better performer overall.
Analyze what specific optimizations in Kernel B are NOT present in Kernel A, and incorporate them.
Write a new kernel that combines the best of both."""


def _extract_code(raw: str) -> str:
    import re
    m = re.search(r"```python\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]*?)```", raw)
    if m:
        return m.group(1).strip()
    if "def kernel(" in raw:
        return raw.strip()
    raise ValueError("Crossbreed response did not contain valid kernel code.")
