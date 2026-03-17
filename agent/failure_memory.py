"""
Failure memory: logs regressions with unified diffs so the Judge never
recommends the same dead-end approach twice.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FailureRecord:
    iteration: int
    timestamp: float
    parent_hash: str         # SHA256[:8] of parent kernel code
    child_hash: str          # SHA256[:8] of child kernel code
    diff: str                # unified diff parent → child
    regression_ms: float     # latency increase (positive = slower)
    regression_pct: float    # % slowdown relative to parent
    brief_used: dict         # the Judge brief that led to this failure
    metric_changes: dict     # which NCU metrics got worse


def _sha8(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:8]


class FailureMemory:
    def __init__(self, path: str | Path = "agent_state/failures.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        iteration: int,
        parent_code: str,
        child_code: str,
        parent_latency_ms: float,
        child_latency_ms: float,
        brief_used: dict,
        metric_changes: dict | None = None,
    ) -> FailureRecord:
        """Record a regression and append to the log file."""
        diff = "\n".join(
            difflib.unified_diff(
                parent_code.splitlines(),
                child_code.splitlines(),
                lineterm="",
                n=2,
            )
        )
        regression_ms = child_latency_ms - parent_latency_ms
        regression_pct = 100.0 * regression_ms / max(parent_latency_ms, 1e-9)

        record = FailureRecord(
            iteration=iteration,
            timestamp=time.time(),
            parent_hash=_sha8(parent_code),
            child_hash=_sha8(child_code),
            diff=diff,
            regression_ms=regression_ms,
            regression_pct=regression_pct,
            brief_used=brief_used,
            metric_changes=metric_changes or {},
        )

        with self.path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        return record

    def load_all(self) -> list[FailureRecord]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(FailureRecord(**json.loads(line)))
        return records

    def get_context_for_prompt(self, n_recent: int = 5) -> str:
        """
        Return a formatted string of recent failures for injection into Judge prompts.
        Includes the diff summary and which metrics worsened.
        """
        records = self.load_all()
        if not records:
            return ""

        recent = records[-n_recent:]
        lines = []
        for r in recent:
            lines.append(
                f"\n--- FAILURE at iteration {r.iteration} "
                f"(+{r.regression_ms:.3f}ms, +{r.regression_pct:.1f}% slower) ---"
            )

            # Summarize what the Judge recommended
            brief = r.brief_used
            changes = brief.get("top_3_changes", [])
            if changes:
                lines.append("  Judge recommended:")
                for c in changes[:3]:
                    lines.append(f"    - {c.get('change', '?')}")

            # Diff summary (first 15 lines)
            diff_lines = r.diff.splitlines()
            meaningful = [l for l in diff_lines if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
            if meaningful:
                lines.append("  Code changes that caused regression:")
                for dl in meaningful[:8]:
                    lines.append(f"    {dl}")
                if len(meaningful) > 8:
                    lines.append(f"    ... ({len(meaningful)-8} more lines)")

            # Metric changes
            if r.metric_changes:
                worse = {k: v for k, v in r.metric_changes.items() if isinstance(v, (int, float)) and v < 0}
                if worse:
                    lines.append("  Metrics that got worse:")
                    for k, v in list(worse.items())[:5]:
                        lines.append(f"    {k}: {v:+.2f}")

        return "\n".join(lines) if lines else ""
