"""
Kernel search tree: tracks parent/child relationships between kernel versions.
Enables backtracking when optimization gets stuck in a local minimum.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _sha8(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:8]


@dataclass
class KernelNode:
    id: str                         # e.g., "iter_007_main"
    parent_id: str | None
    kernel_code: str
    iteration: int
    variant: str                    # "main" | "conservative" | "aggressive" | "autotune" | "crossbreed" | "seed"
    latency_ms: float | None = None
    speedup_factor: float | None = None
    metrics: dict = field(default_factory=dict)
    brief: dict = field(default_factory=dict)
    children: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    code_hash: str = ""

    def __post_init__(self):
        if not self.code_hash:
            self.code_hash = _sha8(self.kernel_code)


class KernelTree:
    def __init__(self):
        self.nodes: dict[str, KernelNode] = {}
        self.best_node_id: str | None = None
        self.current_node_id: str | None = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path = "agent_state/tree.json") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "best_node_id": self.best_node_id,
            "current_node_id": self.current_node_id,
            "nodes": {
                nid: {k: v for k, v in asdict(n).items() if k != "kernel_code"}
                for nid, n in self.nodes.items()
            },
        }
        # Save kernel codes separately (potentially large)
        codes_path = path.parent / "kernels"
        codes_path.mkdir(exist_ok=True)
        for nid, n in self.nodes.items():
            code_file = codes_path / f"{nid}.py"
            if not code_file.exists():
                code_file.write_text(n.kernel_code)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path = "agent_state/tree.json") -> "KernelTree":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        tree = cls()
        tree.best_node_id = data["best_node_id"]
        tree.current_node_id = data["current_node_id"]
        codes_path = path.parent / "kernels"
        for nid, node_data in data["nodes"].items():
            code_file = codes_path / f"{nid}.py"
            code = code_file.read_text() if code_file.exists() else ""
            node = KernelNode(kernel_code=code, **node_data)
            tree.nodes[nid] = node
        return tree

    @classmethod
    def load_or_create(cls, path: str | Path = "agent_state/tree.json") -> "KernelTree":
        return cls.load(path) if Path(path).exists() else cls()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def add_node(self, node: KernelNode) -> None:
        self.nodes[node.id] = node
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children.append(node.id)
        # Update best
        if node.speedup_factor is not None:
            if self.best_node_id is None:
                self.best_node_id = node.id
            else:
                best = self.nodes[self.best_node_id]
                if (best.speedup_factor is None or
                        node.speedup_factor > best.speedup_factor):
                    self.best_node_id = node.id

    def update_result(
        self,
        node_id: str,
        latency_ms: float,
        speedup_factor: float,
        metrics: dict,
    ) -> None:
        node = self.nodes[node_id]
        node.latency_ms = latency_ms
        node.speedup_factor = speedup_factor
        node.metrics = metrics
        # Re-check best
        if self.best_node_id is None or (
            self.nodes[self.best_node_id].speedup_factor is None
            or speedup_factor > self.nodes[self.best_node_id].speedup_factor
        ):
            self.best_node_id = node_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_best_node(self) -> KernelNode | None:
        if self.best_node_id and self.best_node_id in self.nodes:
            return self.nodes[self.best_node_id]
        # Fallback: find node with highest speedup
        scored = [n for n in self.nodes.values() if n.speedup_factor is not None]
        if not scored:
            return None
        return max(scored, key=lambda n: n.speedup_factor)

    def get_current_node(self) -> KernelNode | None:
        if self.current_node_id and self.current_node_id in self.nodes:
            return self.nodes[self.current_node_id]
        return self.get_best_node()

    def get_recent_nodes(self, window: int = 8) -> list[KernelNode]:
        """Return the last `window` nodes by iteration order."""
        all_nodes = sorted(self.nodes.values(), key=lambda n: n.iteration)
        return all_nodes[-window:]

    def should_backtrack(self, window: int = 8) -> bool:
        """Return True if no improvement in the last `window` nodes."""
        best = self.get_best_node()
        if best is None:
            return False
        recent = self.get_recent_nodes(window)
        if len(recent) < window:
            return False
        best_recent = max(
            (n.speedup_factor for n in recent if n.speedup_factor is not None),
            default=0.0,
        )
        return best_recent <= (best.speedup_factor or 0.0)

    def backtrack(self) -> KernelNode:
        """
        Find a promising alternative branch to explore.
        Strategy: pick the best node not in the current lineage.
        """
        current = self.get_current_node()
        if current is None:
            return self.get_best_node()

        # Get the lineage of the current node
        lineage: set[str] = set()
        nid = current.id
        while nid:
            lineage.add(nid)
            parent_id = self.nodes[nid].parent_id
            nid = parent_id if parent_id and parent_id in self.nodes else None

        # Find the best node NOT in the current lineage
        candidates = [
            n for n in self.nodes.values()
            if n.id not in lineage and n.speedup_factor is not None
        ]
        if not candidates:
            return self.get_best_node()

        return max(candidates, key=lambda n: n.speedup_factor)

    def get_path_to_best(self) -> list[KernelNode]:
        """Return the path from root to best node (for context)."""
        best = self.get_best_node()
        if best is None:
            return []
        path = []
        nid = best.id
        while nid and nid in self.nodes:
            path.append(self.nodes[nid])
            nid = self.nodes[nid].parent_id
        return list(reversed(path))

    def summary(self) -> str:
        total = len(self.nodes)
        scored = [n for n in self.nodes.values() if n.speedup_factor is not None]
        best = self.get_best_node()
        lines = [
            f"Tree: {total} nodes, {len(scored)} benchmarked",
        ]
        if best:
            lines.append(
                f"Best: {best.id} @ {best.speedup_factor:.3f}x speedup "
                f"({best.latency_ms:.3f}ms, iter {best.iteration})"
            )
        return " | ".join(lines)
