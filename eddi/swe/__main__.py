"""Command line entry point for the engineering loop and its benchmark.

    python3 -m eddi.swe bench              # score the loop on the local suite
    python3 -m eddi.swe bench --ablate     # loop alone vs loop + proposer
    python3 -m eddi.swe solve <repo> --probe "pytest -q" --goal "..."

The benchmark is the point. A capability claim without a number is a hope, and
this module exists so that every improvement to the loop can be stated as a
score that can be reproduced on any machine in a couple of seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .bench import TASKS, format_outcomes, run_suite
from .loop import EngineeringLoop, Probe, TaskState
from .proposers import HeuristicProposer
from .tools import Toolbox


def _solver_factory(use_proposer: bool, verbose: bool = False):
    """Build the callable the benchmark invokes on each materialised task."""

    def solve(root: Path):
        task = next((t for t in TASKS if t.name == Path(root).name), None)
        if task is None:
            return (0, "unknown task")
        toolbox = Toolbox(str(root))
        state = TaskState(goal=task.goal, probes=[Probe(task.probe)])
        loop = EngineeringLoop(
            toolbox,
            propose=HeuristicProposer() if use_proposer else None,
            verbose=verbose,
        )
        loop.run(state)
        return (state.tried(), loop.report(state).splitlines()[-1])

    return solve


def cmd_bench(args: argparse.Namespace) -> int:
    """Score the loop. The number reported is tasks whose hidden test passes."""
    if args.ablate:
        print("=== loop only (deterministic phases) ===")
        only = run_suite(_solver_factory(False), hide_visible_tests=True)
        print(format_outcomes(only))
        print()
        print("=== loop + symbolic proposer ===")
        full = run_suite(_solver_factory(True), hide_visible_tests=True)
        print(format_outcomes(full))
        baseline = sum(1 for o in only if o.resolved)
        solved = sum(1 for o in full if o.resolved)
        print()
        print(
            f"loop alone {baseline}/{len(full)}  ->  with proposer {solved}/{len(full)}"
            f"  ({solved - baseline} tasks needed judgement)"
        )
        return 0 if solved == len(full) else 1

    outcomes = run_suite(_solver_factory(True), hide_visible_tests=not args.show_tests)
    print(format_outcomes(outcomes))
    return 0 if all(o.resolved for o in outcomes) else 1


def cmd_solve(args: argparse.Namespace) -> int:
    """Run the loop against a real repository."""
    root = Path(args.repo).expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    toolbox = Toolbox(str(root))
    probes = [Probe(cmd) for cmd in (args.probe or [])]
    state = TaskState(goal=args.goal or "", probes=probes)
    loop = EngineeringLoop(
        toolbox,
        propose=HeuristicProposer() if not args.no_proposer else None,
        verbose=True,
        max_iterations=args.iterations,
    )
    loop.run(state)
    print()
    print(loop.report(state))
    return 0 if state.solved else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eddi.swe", description="Engineering loop: solve, verify, and measure."
    )
    sub = parser.add_subparsers(dest="command")

    bench = sub.add_parser("bench", help="score the loop on the local task suite")
    bench.add_argument("--ablate", action="store_true",
                       help="compare the loop alone against the loop with a proposer")
    bench.add_argument("--show-tests", action="store_true",
                       help="leave the visible tests in place (easier mode)")
    bench.set_defaults(func=cmd_bench)

    solve = sub.add_parser("solve", help="run the loop on a real repository")
    solve.add_argument("repo")
    solve.add_argument("--probe", action="append",
                       help="command that should pass when fixed (repeatable)")
    solve.add_argument("--goal", help="what is wrong, in the issue's words")
    solve.add_argument("--iterations", type=int, default=8)
    solve.add_argument("--no-proposer", action="store_true",
                       help="localise and report without proposing edits")
    solve.set_defaults(func=cmd_solve)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        return cmd_bench(argparse.Namespace(ablate=True, show_tests=False))
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
