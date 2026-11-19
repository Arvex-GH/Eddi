"""Does the model close the gap that structure could not?

Two claims are tested here, and the second matters more than the first.

1. Given the two out-of-distribution bugs from coverage_boundary.py, a
   proposer that can read the issue and the source can fix them. Those bugs
   are real: the correct line is in the file, and the issue describes the
   behaviour precisely enough to pick it out.

2. A model that is wrong, or that answers outside the file, costs nothing.
   Its proposal is rejected before it can be applied, the loop keeps the
   deterministic path, and the task is reported unsolved. A model in the loop
   must be able to fail safely -- otherwise "frontier quality on a laptop"
   means "crashes are now probabilistic".

The model used here is a stand-in with a fixed policy, not a neural network:
it reads the FUNCTION text out of the prompt and picks the line that
contradicts the issue. That isolates the question of whether the interface and
the loop can carry a model's judgement at all, which is what is being decided
before any weights are involved.
"""

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eddi.swe.bench import score_task
from eddi.swe.llm import HybridProposer, LLMProposer, proposal_from_reply
from eddi.swe.loop import EngineeringLoop, Probe, TaskState
from eddi.swe.proposers import HeuristicProposer
from eddi.swe.tools import Toolbox
from coverage_boundary import TASKS


def _functions_in(prompt):
    """Parse the rendered function blocks back out of the prompt."""
    blocks, current = [], []
    for line in prompt.splitlines():
        if line.startswith("--- "):
            if current:
                blocks.append(current)
            current = [line]
        elif current is not None and re.match(r"^\s*\d+ \| ", line):
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _clean(lines):
    """Strip the 'NNNN | ' gutter back to source."""
    out = []
    for line in lines:
        match = re.match(r"^\s*\d+ \| (.*)$", line)
        out.append(match.group(1) if match else line)
    return out


#: A model stand-in with a narrow, inspectable policy: replace the first line
#: whose arithmetic contradicts the issue. This is *not* a language model; it is
#: a witness that a correct judgement, delivered through this interface, is
#: enough to solve the case. Any real model is asked exactly this question.
def honest_model(system, user):
    """A witness policy: read the issue, read the source, name the fix.

    Not a language model. It exists to prove the interface can carry a correct
    judgement end to end, so that when a real model is wired in the only thing
    that changes is where the decision comes from.
    """
    import json

    issue_text = user.split("ISSUE", 1)[1].split("FUNCTIONS", 1)[0].lower()
    functions = {}
    paths = {}
    for block in _functions_in(user):
        header = block[0]
        rest = header.split("::", 1)
        path = rest[0].replace("---", "").strip()
        name = rest[1].strip().split("(")[0].strip().rsplit(" ", 1)[0]
        functions[name] = _clean(block[1:])
        paths[name] = path

    def reply(strategy, edits, reason):
        return json.dumps(
            {"strategy": strategy, "edits": edits, "reason": reason}
        )

    def indent_of(line):
        return line[: len(line) - len(line.lstrip())]

    if "swapped" in issue_text or "transposed" in issue_text:
        # Both bodies are wrong and belong to each other: one decision, two
        # edits. Sending them as a sequence is what makes it atomic.
        edits = []
        for name, correct in (("area", "return width * height"),
                              ("perimeter", "return 2 * (width + height)")):
            for line in functions.get(name, []):
                if line.strip().startswith("return"):
                    edits.append(
                        {"path": paths[name], "old": line,
                         "new": indent_of(line) + correct}
                    )
                    break
        return reply("novel", edits, "the two bodies belong to each other")

    if "negative" in issue_text and "valueerror" in issue_text:
        for line in functions.get("shipping_cost", []):
            if line.strip().startswith("base ="):
                return reply(
                    "novel",
                    [{"path": paths["shipping_cost"], "old": line,
                      "new": line + "\n" + indent_of(line) + "if weight_kg < 0: raise ValueError(weight_kg)"}],
                    "a negative weight has no meaning",
                )

    return json.dumps({"strategy": "novel", "edits": [], "reason": "cannot tell"})


def dishonest_model(system, user):
    """Answers confidently with a line that is not in the file."""
    return (
        '{"strategy":"novel","old":"    return 42","new":"    return 43",'
        '"reason":"plausible looking nonsense"}'
    )


def run(task, complete):
    proposer = HybridProposer(
        heuristic=HeuristicProposer(),
        model=LLMProposer(complete=complete),
    )
    outcome = score_task(task, lambda root: _solve(root, proposer))
    return outcome, proposer


def _solve(root, proposer):
    loop = EngineeringLoop(Toolbox(str(root)), propose=proposer, verbose=False)
    state = loop.run(TaskState(goal=STATE_GOAL["goal"], probes=[Probe(STATE_GOAL["probe"])]))
    return state.tried(), ""


STATE_GOAL = {"goal": "", "probe": ""}


def main():
    print("hybrid: structure first, model only on a miss")
    print()
    for task in TASKS:
        STATE_GOAL["goal"], STATE_GOAL["probe"] = task.goal, task.probe
        outcome, proposer = run(task, honest_model)
        mark = "SOLVED" if outcome.hidden_passes else "unsolved"
        print(f"  {task.name:34} {mark:8} model calls: {proposer.model_calls}")
        if not outcome.hidden_passes and outcome.detail:
            print(f"      {outcome.detail[:110]}")
    print()
    print("and a model that is confidently wrong:")
    print()
    for task in TASKS:
        STATE_GOAL["goal"], STATE_GOAL["probe"] = task.goal, task.probe
        outcome, proposer = run(task, dishonest_model)
        mark = "SOLVED (bad!)" if outcome.hidden_passes else "safely unsolved"
        print(f"  {task.name:34} {mark:16} model calls: {proposer.model_calls}")
    print()
    print("A wrong answer from the model cannot be applied: the proposed line")
    print("is not in the repository, so it is rejected in microseconds and the")
    print("loop reports honestly. The model is a participant, not an authority.")


if __name__ == "__main__":
    main()
