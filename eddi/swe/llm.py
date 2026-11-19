"""The one place where judgement has to come from somewhere other than structure.

The loop is deterministic and the heuristic proposer covers the bug families
that announce themselves in code structure. Measured, that is seven of nine --
the two it misses are real bugs whose fix has no shape: two bodies transposed,
a precondition that does not exist yet. Those need something that can read the
issue, look at the function and decide what the code should have said.

That is a small job, and this module is built so it stays small. The model is
not asked to plan, explore, run tests, or write a patch. It sees a narrowly
scoped question -- these functions, this failure, these five known patterns --
and answers with one line to replace. Everything else stays deterministic,
which is what makes the quality per parameter so high: the model spends its
capacity on the decision instead of on the plumbing.

The interface is a callable for a reason. An adapter that lives on a class
hierarchy invites the model to creep into more of the loop; a callable fits in
one place and is easy to point at whatever runs locally.
"""

from __future__ import annotations

import ast
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .loop import TaskState
from .proposers import FunctionView, Proposal, view_functions
from .tools import Toolbox


# --------------------------------------------------------------------------
# What the model is shown
# --------------------------------------------------------------------------

#: The answer alphabet. Constraining the output to these names turns an open
#: "write a patch" problem into a classification over six options, which is a
#: far smaller thing for a small model to get right. It can only choose a
#: strategy that the code can actually support.
KNOWN_STRATEGIES = (
    "trailing-drop",
    "shared-mutable",
    "suffix-slice",
    "inverted-guard",
    "search-boundary",
    "swallowed-error",
    "novel",
)

_SYSTEM = """\
You are repairing one function in a Python repository.

You are given: the issue text, the failing assertion, and the source of the
functions that could be involved. You do not explore, you do not run anything.
Decide which single line is wrong and what it should say instead.

Rules that matter more than being helpful:

- Answer with the smallest edit that changes behaviour. One line in, one line
  out. Do not reformat, rename, or restructure anything else.
- `old` must be copied character for character from the source you were given,
  including its indentation. If it is not present verbatim, the edit cannot be
  applied and your answer is wasted.
- `new` must keep the same indentation as `old`.
- If no single-line edit can fix this, say so with strategy "novel" and an
  empty `old` and `new`. A refusal is useful. A guess is not: it will be
  applied and then fail the tests, and the loop will have lost a turn.

Reply with JSON only:
{"strategy": "<one of the names>",
 "edits": [{"path": "<file>", "old": "<exact line>", "new": "<replacement>"}],
 "reason": "<one sentence>"}

Usually there is exactly one edit. Give several only when the correct change
is genuinely plural -- two bodies that belong to each other, exchanged. Each
edit in the list must independently satisfy the rules above.
"""


def build_prompt(state: TaskState, toolbox: Toolbox, max_functions: int = 8) -> str:
    """Render the narrow question the model is asked.

    Deliberately small: the goal, the failing assertion if there is one, and
    the source of the candidate functions. No repository listing, no history.
    Everything the model does not need is context it could be distracted by.
    """
    lines: List[str] = []
    lines.append("ISSUE")
    lines.append(state.goal.strip())
    lines.append("")

    if state.frames:
        lines.append("FAILING ASSERTION")
        frame = state.frames[0]
        lines.append(f"{frame.path}:{frame.line} in {frame.function or '?'}")
        if frame.source:
            lines.append(f"    {frame.source.strip()}")
        if state.last_output:
            lines.append("")
            lines.append("OUTPUT")
            lines.append(state.last_output.strip()[-1200:])
        lines.append("")

    lines.append("FUNCTIONS")
    shown = 0
    for path in _candidate_paths(state):
        if shown >= max_functions:
            break
        for view in view_functions(toolbox, path):
            if shown >= max_functions:
                break
            lines.append("")
            lines.append(f"--- {view.path} :: {view.name} (lines {view.start}-{view.end})")
            for offset, row in enumerate(view.source.splitlines()):
                lines.append(f"{view.start + offset:5d} | {row}")
            shown += 1

    lines.append("")
    lines.append("STRATEGIES")
    for name in KNOWN_STRATEGIES:
        lines.append(f"- {name}")
    return "\n".join(lines)


def _candidate_paths(state: TaskState) -> List[str]:
    ordered: List[str] = []
    for frame in state.frames:
        if frame.path and frame.path not in ordered:
            ordered.append(frame.path)
    for path in state.files_read:
        if path not in ordered:
            ordered.append(path)
    return [path for path in ordered if path.endswith(".py")]


# --------------------------------------------------------------------------
# Reading the answer
# --------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_reply(text: str) -> Optional[dict]:
    """Pull the JSON object out of a reply, tolerating fences and chatter."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for chunk in candidates:
        match = _JSON_RE.search(chunk)
        if not match:
            continue
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def proposal_from_reply(payload: dict, toolbox: Toolbox, state: TaskState):
    """Turn a validated reply into the tuple the loop consumes.

    Validation is the important half. A model that answers confidently with a
    line that is not in the file would otherwise produce an edit that silently
    matches nothing, and the loop would burn an iteration learning what this
    function can check in a millisecond.
    """
    if not isinstance(payload, dict):
        return None
    # "novel" means *unclassified*, not *refused*: it is the honest label for a
    # fix that matches none of the five known patterns, and those are precisely
    # the out-of-distribution bugs this adapter exists to solve. A refusal is
    # signalled by returning no edits at all.
    raw_edits = payload.get("edits")
    if raw_edits is None:
        raw_edits = [{"old": payload.get("old"), "new": payload.get("new")}]
    if not isinstance(raw_edits, list) or not raw_edits:
        return None

    available = {}
    for path in _candidate_paths(state):
        text = _raw_text(toolbox, path)
        if text is not None:
            available[path] = text

    edits = []
    for item in raw_edits:
        if not isinstance(item, dict):
            return None
        old, new = item.get("old"), item.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        if not old.strip() or old == new:
            return None
        path = item.get("path")
        # Without a path, the file is whichever candidate actually contains the
        # line. That is unambiguous in the cases this is asked about, and it
        # saves the model an output token it would sometimes get wrong.
        if isinstance(path, str) and path in available:
            if old in available[path]:
                edits.append((path, old, new))
                continue
            return None
        matches = [candidate for candidate, text in available.items() if old in text]
        if not matches:
            return None
        edits.append((matches[0], old, new))

    if len(edits) == 1:
        return edits[0]
    return edits


def _raw_text(toolbox: Toolbox, path: str) -> Optional[str]:
    """The file exactly as it is on disk.

    ``Toolbox.read`` deliberately returns line-numbered text, because that is
    what a model needs to reason about positions. Validation needs the
    opposite: the bytes, so that a proposal copied verbatim from the prompt
    can be checked for presence. Using the numbered view here rejected every
    correct answer -- the line was right and could never be matched.
    """
    root = Path(toolbox.root).resolve()
    try:
        candidate = (root / path).resolve()
        candidate.relative_to(root)
    except (ValueError, OSError):
        return None
    if not candidate.is_file():
        return None
    try:
        return candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------

@dataclass
class LLMProposer:
    """Ask a model for the one edit structure cannot name.

    ``complete`` is any callable taking a prompt and returning text. Keeping it
    that narrow means the same object drives a local llama.cpp server, an
    Ollama instance, a hosted endpoint, or a test double, without a single
    branch in the loop.
    """

    complete: object
    max_functions: int = 8
    name: str = "llm"
    last_prompt: str = ""
    last_reply: str = ""

    def __call__(self, state: TaskState, toolbox: Toolbox) -> Optional[Tuple[str, str, str]]:
        prompt = build_prompt(state, toolbox, max_functions=self.max_functions)
        self.last_prompt = prompt
        try:
            reply = self.complete(_SYSTEM, prompt)  # type: ignore[misc]
        except Exception:
            return None
        self.last_reply = reply or ""
        payload = parse_reply(self.last_reply)
        if payload is None:
            return None
        return proposal_from_reply(payload, toolbox, state)

    def propose(self, view: FunctionView) -> Optional[Proposal]:
        """Strategy-shaped entry point, for symmetry with the heuristics."""
        payload = parse_reply(self.last_reply or "")
        if payload is None:
            return None
        old, new = payload.get("old"), payload.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        return Proposal(view.path, old, new, str(payload.get("reason", "")))


# --------------------------------------------------------------------------
# A concrete transport, so this is not a stub
# --------------------------------------------------------------------------

def http_complete(
    model: str,
    base_url: str = "http://127.0.0.1:8080/v1/chat/completions",
    timeout: float = 120.0,
    temperature: float = 0.0,
):
    """A chat-completions transport, for llama.cpp, Ollama, vLLM or a proxy.

    The default points at a local llama.cpp server because that is the target
    this architecture is aimed at: a quantised model on the user's own machine,
    with no network in the loop and no per-token cost.
    """

    def complete(system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            base_url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        api_key = os.environ.get("EDDI_LLM_KEY")
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]

    return complete


def is_available(base_url: str = "http://127.0.0.1:8080/v1/models") -> bool:
    """Whether a local server is listening. Checked before it is relied on."""
    try:
        with urllib.request.urlopen(base_url, timeout=1.0):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


# --------------------------------------------------------------------------
# Hybrid: structure first, model only for what structure cannot name
# --------------------------------------------------------------------------

@dataclass
class HybridProposer:
    """Run the cheap strategies first and call the model only on a miss.

    This is the cost model of the whole design in eight lines. Seven of nine
    measured bug families are answered without inference; the model is woken
    for the remainder. On a laptop that is the difference between a tool that
    spins the fan for every edit and one that usually does not.
    """

    heuristic: object
    model: object
    name: str = "hybrid"
    model_calls: int = 0

    def __call__(self, state: TaskState, toolbox: Toolbox) -> Optional[Tuple[str, str, str]]:
        cheap = self.heuristic(state, toolbox)  # type: ignore[misc]
        if cheap is not None:
            return cheap
        self.model_calls += 1
        return self.model(state, toolbox)  # type: ignore[misc]
