"""How large is the decision the proposer actually has to make?

The claim under test: an agent's engineering competence lives mostly in the
loop, and the model is only consulted at one point -- choosing which structural
hypothesis explains a failure. If that choice is small, a small model suffices,
and "frontier quality on low-end hardware" stops being a contradiction.

This measures, for each benchmark task, the size of the space the proposer
searches: how many functions are in play, how many editable lines are inside
them, and how many distinct edit *shapes* the five strategies can produce.
"""

import tempfile
from pathlib import Path

from eddi.swe.bench import TASKS, materialise, score_task
from eddi.swe.loop import EngineeringLoop, Probe, TaskState
from eddi.swe.proposers import HeuristicProposer, view_functions
from eddi.swe.tools import Toolbox


def _solver_for(task):
    """A solver that is just the loop, as bench uses it."""
    from eddi.swe.__main__ import _solver_factory

    return _solver_factory(use_proposer=True)


def measure(task):
    root = Path(tempfile.mkdtemp())
    for relative, body in task.files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    tools = Toolbox(str(root))
    loop = EngineeringLoop(tools, propose=HeuristicProposer(), verbose=False)

    # Localise exactly as the real run does, then count what it is choosing among.
    state = TaskState(goal=task.goal, probes=[Probe(task.probe)])
    state = loop.run(state)

    # ``state.solved`` is the loop's own belief. What matters is the graded
    # hidden test, so it is run too -- the two disagree, and only one of them
    # is an honest measure of the task being fixed.

    modules = sorted(
        {frame.path for frame in state.frames if frame.path}
        | set(state.files_read)
    )
    functions = 0
    editable_lines = 0
    for module in modules:
        for view in view_functions(tools, module):
            functions += 1
            editable_lines += sum(
                1
                for line in view.source.splitlines()[1:]
                if line.strip() and not line.strip().startswith(("#", '"""', "'''"))
            )

    strategies = [
        name
        for name in dir(HeuristicProposer)
        if name.startswith("_") and not name.startswith("__") and name != "_try"
    ]
    outcome = score_task(task, _solver_for(task))
    return {
        "task": task.name,
        "modules": len(modules),
        "functions": functions,
        "editable_lines": editable_lines,
        "strategies": len(strategies),
        "solved": outcome.hidden_passes,
        "edits": state.tried(),
    }


def main():
    rows = [measure(task) for task in TASKS]
    header = f"{'task':26} {'mod':>4} {'fn':>4} {'lines':>6} {'strat':>6} {'edits':>6} {'ok':>3}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['task']:26} {r['modules']:4d} {r['functions']:4d} "
            f"{r['editable_lines']:6d} {r['strategies']:6d} {r['edits']:6d} "
            f"{'yes' if r['solved'] else 'NO':>3}"
        )
    print()
    total_fn = sum(r["functions"] for r in rows)
    total_lines = sum(r["editable_lines"] for r in rows)
    print(f"functions touched across all tasks: {total_fn}")
    print(f"editable lines inside them:         {total_lines}")
    print(f"strategies available:               {rows[0]['strategies']}")
    print()
    print("decision space per task, worst case:")
    print(f"  lines x strategies = {max(r['editable_lines'] for r in rows)}"
          f" x {rows[0]['strategies']} = "
          f"{max(r['editable_lines'] for r in rows) * rows[0]['strategies']} candidates")


if __name__ == "__main__":
    main()
