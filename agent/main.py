"""
Agent optimization loop CLI entry point.

Usage:
    python -m agent.main [options]

Examples:
    # Fresh run, 50 iterations
    python -m agent.main --iterations 50

    # Resume from last checkpoint
    python -m agent.main --iterations 100 --resume

    # Use specific models
    python -m agent.main --judge-model claude-opus-4-6 --coder-model claude-sonnet-4-6

    # Faster iteration (less thorough)
    python -m agent.main --autotune-every 10 --crossbreed-every 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _check_env() -> None:
    """Validate required environment variables."""
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Set them before running:")
        for k in missing:
            print(f"  export {k}=...")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic MoE kernel optimization loop (Judge + Coder + tree search)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Maximum number of Judge+Coder iterations",
    )
    parser.add_argument(
        "--autotune-every", type=int, default=5,
        help="Run tile-size grid search every N iterations",
    )
    parser.add_argument(
        "--crossbreed-every", type=int, default=10,
        help="Run population crossbreed every N iterations",
    )
    parser.add_argument(
        "--backtrack-window", type=int, default=8,
        help="Backtrack if no improvement in last N iterations",
    )
    parser.add_argument(
        "--judge-model", default="claude-opus-4-6",
        help="Claude model for the Judge agent",
    )
    parser.add_argument(
        "--coder-model", default="claude-sonnet-4-6",
        help="Claude model for the Coder agent",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from existing agent_state/ (default: True)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start fresh, ignoring existing agent_state/",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    _check_env()

    resume = args.resume and not args.fresh

    print(f"Starting optimization loop:")
    print(f"  Iterations:      {args.iterations}")
    print(f"  Autotune every:  {args.autotune_every}")
    print(f"  Crossbreed every:{args.crossbreed_every}")
    print(f"  Backtrack window:{args.backtrack_window}")
    print(f"  Judge model:     {args.judge_model}")
    print(f"  Coder model:     {args.coder_model}")
    print(f"  Resume:          {resume}")
    print()

    from agent.loop import run_optimization_loop

    run_optimization_loop(
        max_iterations=args.iterations,
        autotune_every=args.autotune_every,
        crossbreed_every=args.crossbreed_every,
        backtrack_window=args.backtrack_window,
        judge_model=args.judge_model,
        coder_model=args.coder_model,
        resume=resume,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
