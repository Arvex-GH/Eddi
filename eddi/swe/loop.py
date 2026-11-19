"""The loop: how a software engineering task actually gets solved.

Why this exists
---------------
mini-swe-agent resolves 65% of SWE-bench Verified with about 100 lines of
Python and no fine-tuning. The lesson is not that the loop is trivial -- it is
that *most of an agent's competence lives in the loop, not in the weights*.
The published system prompt hands the model a fixed workflow:

    1. find and read the relevant files
    2. write a script that reproduces the issue
    3. edit the source
    4. re-run the script
    5. check edge cases
    6. submit

Five of those six steps are deterministic. Only step 3 -- knowing exactly what
to change -- needs judgement. A frontier model brings judgement and also does
the other five by hand, expensively. A small system can invert that: perform
the deterministic five mechanically, and spend its scarce reasoning budget
only where reasoning is actually required.

That inversion is the whole design here:

* **Localise deterministically.** The traceback names a file and a line. The
  issue text names symbols. Neither requires a model to interpret.
* **Reproduce deterministically.** A failing test or a traceback is already a
  reproduction; writing a new script to re-derive it wastes turns.
* **Verify deterministically.** Run the tests. The exit code is truth.
* **Iterate deterministically.** When a run fails, the new traceback updates
  the localisation and the loop continues.
* **Use the model only to propose an edit** at a location already narrowed
  down to a few lines.

The loop is therefore useful even with a weak proposer, and it degrades
gracefully: with no proposer at all it still localises, reproduces and reports
exactly where the failure is.
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .tools import ToolResult, Toolbox


class Step(str, Enum):
    LOCATE = "locate"
    READ = "read"
    EDIT = "edit"
    VERIFY = "verify"
    DONE = "done"
    STUCK = "stuck"


@dataclass
class Frame:
    """One entry of a Python traceback, as a place a bug can live."""

    path: str
    line: int
    function: str = ""
    source: str = ""

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}"
        return f"{where} in {self.function}" if self.function else where


@dataclass
class Probe:
    """A fast, deterministic way to know whether the task is solved."""

    command: str
    description: str = ""

    def __str__(self) -> str:
        return self.command


@dataclass
class Attempt:
    step: Step
    detail: str
    ok: bool = True






def _bottom_up(toolbox, edits: list) -> list:
    """Order edits so that earlier ones cannot disturb later ones.

    Only edits to the same file can interfere, and only when one edit's `new`
    text contains another's `old`. Ordering by the position of `old` in the
    current file, descending, is what makes a set of edits behave like one
    atomic decision.
    """
    def position(edit):
        path, old, _new = edit
        result = toolbox.read(path, budget=200000)
        if not result.ok:
            return -1
        return result.text.find(old)

    indexed = list(enumerate(edits))
    indexed.sort(key=lambda pair: (pair[1][0], -position(pair[1])))
    return [edit for _index, edit in indexed]

@dataclass
class _EditOutcome:
    """The minimum of ToolResult that the loop uses after editing."""

    ok: bool
    text: str


def _as_edits(proposal) -> list:
    """Normalise a proposal to a list of (path, old, new) triples.

    The single-edit tuple is the common case and stays the documented shape;
    the sequence form exists for decisions that are inherently plural.
    """
    if proposal is None:
        return []
    if isinstance(proposal, tuple) and len(proposal) == 3 and all(
        isinstance(part, str) for part in proposal
    ):
        return [proposal]
    edits = []
    for item in proposal:
        if isinstance(item, tuple) and len(item) == 3 and all(
            isinstance(part, str) for part in item
        ):
            edits.append(item)
    return edits

@dataclass
class TaskState:
    """Everything the loop has learned about one task so far."""

    goal: str
    probes: List[Probe] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    attempts: List[Attempt] = field(default_factory=list)
    solved: bool = False
    verified: bool = False
    last_output: str = ""
    files_read: List[str] = field(default_factory=list)

    def note(self, step: Step, detail: str, ok: bool = True) -> None:
        self.attempts.append(Attempt(step, detail, ok))

    def explored(self) -> bool:
        return bool(self.files_read)

    def tried(self) -> int:
        return sum(1 for a in self.attempts if a.step is Step.EDIT)


# --------------------------------------------------------------------- parsing

#: A CPython traceback frame ends with a `File "...", line N, in name` line.
_FRAME_RE = re.compile(
    r'File "(?P<path>[^"]+)",\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>\S+))?'
)
_SOURCE_RE = re.compile(r"^\s{4}(?P<src>\S.*)$")


def parse_traceback(text: str, root_prefix: str = "") -> List[Frame]:
    """Extract the stack from a Python traceback, innermost frame last.

    The last frame is where the error was raised; the frames before it are the
    path that led there. Both matter -- the bug is often one frame up, in the
    caller that passed a bad value -- so all of them are returned, ordered.
    """
    frames: List[Frame] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = _FRAME_RE.search(line)
        if not match:
            continue
        source = ""
        for follow in lines[index + 1: index + 4]:
            src = _SOURCE_RE.match(follow)
            if src:
                source = src.group("src").strip()
                break
        path = match.group("path")
        if root_prefix and path.startswith(root_prefix):
            path = path[len(root_prefix):].lstrip("/\\")
        frames.append(
            Frame(
                path=path.replace("\\", "/"),
                line=int(match.group("line")),
                function=match.group("func") or "",
                source=source,
            )
        )
    return frames


#: Words that name a thing in a codebase rather than describe it.
_CODE_TOKEN_RE = re.compile(r"`([^`]{2,60})`")
_SNAKE_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
_CAMEL_RE = re.compile(r"\b([a-z]+(?:[A-Z][a-z0-9]+)+)\b")
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(")


def symbols_in(text: str) -> List[str]:
    """Pull likely code identifiers out of an issue description.

    Issue text is written by people, so it mixes prose with snippets. Backticks
    are the strongest signal; snake_case and camelCase names are next; a bare
    `name(` is a call. Ordinary words are filtered by requiring that a
    candidate look like code rather than English.
    """
    found: List[str] = []
    seen = set()

    def add(value: str) -> None:
        value = value.strip()
        if not value or value in seen:
            return
        # A symbol has no spaces and is not a sentence fragment.
        if " " in value or value.lower() in _ENGLISH:
            return
        seen.add(value)
        found.append(value)

    for match in _CODE_TOKEN_RE.finditer(text):
        add(match.group(1))
    for regex in (_SNAKE_RE, _CAMEL_RE, _CALL_RE):
        for match in regex.finditer(text):
            add(match.group(1))
    return found


#: Common English words that the identifier patterns would otherwise accept.
_ENGLISH = frozenset(
    """the and for with that this from have been will would could should when
    then than there their they them what which while where about into over
    under after before also just only more most some such other same very
    make makes made used uses using work works working test tests testing
    file files line lines code error errors issue issues bug bugs problem
    expected actual result results value values case cases return returns
    called call calls should would must need needs want wants like does
    done does going gets given list lists dict string int float none true
    false self args kwargs import from class def lambda raise catch try
    except finally print input output""".split()
)


# ----------------------------------------------------------------- the loop

#: A proposer receives (goal, frames, file view) and returns (path, old, new)
#: or None. It is the ONLY part that needs judgement, and it is injectable so
#: the loop can run with a stub, a heuristic, or a language model behind it.
Proposer = Callable[[TaskState, Toolbox], Optional[Tuple[str, str, str]]]


class EngineeringLoop:
    """A deterministic solve loop with one pluggable judgement point."""

    def __init__(
        self,
        toolbox: Toolbox,
        propose: Optional[Proposer] = None,
        max_iterations: int = 8,
        verbose: bool = False,
    ):
        self.tools = toolbox
        self.propose = propose
        self.max_iterations = max_iterations
        self.verbose = verbose

    # -- plumbing ---------------------------------------------------------
    def _say(self, step: Step, message: str) -> None:
        if self.verbose:
            print(f"  [{step.value:6}] {message}")

    # -- deterministic phases ---------------------------------------------
    def reproduce(self, state: TaskState) -> Optional[bool]:
        """Run every probe; a failing probe is the reproduction we need.

        Writing a fresh reproduction script is a turn a small system cannot
        afford, and it is usually redundant: the failing test, or the command
        in the issue, already demonstrates the bug.
        """
        if not state.probes:
            self._say(Step.VERIFY, "no probe supplied")
            return None
        any_failed = False
        for probe in state.probes:
            result = self.tools.run(probe.command)
            state.last_output = result.text
            failed = not result.ok or "Traceback (most recent call last)" in result.text
            if failed:
                any_failed = True
                frames = self._in_repo(
                    parse_traceback(result.text, root_prefix=str(self.tools.root))
                )
                if frames:
                    state.frames = frames
                    self._say(Step.LOCATE, f"{len(frames)} frame(s) from {probe.command}")
            state.note(
                Step.VERIFY,
                f"probe {probe.command!r} -> exit {result.meta.get('returncode')}",
                ok=not failed,
            )
            if failed:
                return False
        # True means "everything already passes", which is different from
        # "nothing was reproduced" -- conflating the two made the loop declare
        # success on tasks it had never touched.
        return not any_failed

    def _in_repo(self, frames: List[Frame]) -> List[Frame]:
        """Keep only frames that point at files inside the repository.

        A Python traceback ends with frames inside the standard library --
        unittest's loader, importlib, and so on. Those frames are always the
        outermost ones and they never contain the bug: the failure is in the
        user's code, and the library frames are just the scaffolding that
        called it. Following them costs a guarded read attempt per failure,
        which is how an otherwise correct loop died with "path escapes the
        repository" on every single task.
        """
        root = self.tools.root
        kept: List[Frame] = []
        for frame in frames:
            candidate = (root / frame.path).resolve()
            if root == candidate or root in candidate.parents:
                if candidate.is_file():
                    kept.append(frame)
        return kept

    def localise(self, state: TaskState) -> List[str]:
        """Decide which files are worth reading, cheapest signal first.

        1. the traceback, because it is exact
        2. symbols named in the goal, because an issue names what it breaks
        3. nothing else -- guessing at file names wastes the whole budget
        """
        candidates: List[str] = []
        for frame in state.frames:
            if frame.path not in candidates:
                candidates.append(frame.path)
        for symbol in symbols_in(state.goal)[:6]:
            found = self.tools.definitions(symbol)
            if not found.ok:
                continue
            for line in found.text.splitlines()[:3]:
                path = line.split(":", 1)[0]
                if path.endswith(".py") and path not in candidates:
                    candidates.append(path)

        # A failing assertion in a test file points at the *test*, not at the
        # bug: the bug is in the code the test called, and that code never
        # appears in the traceback because it did not raise. When every frame
        # lands in a test, the real candidates are the package modules the
        # issue is about.
        if candidates and all(Path(c).name.startswith("test_") for c in candidates):
            for path in self._package_modules():
                if path not in candidates:
                    candidates.append(path)
        self._say(Step.LOCATE, f"{len(candidates)} candidate file(s): {candidates[:5]}")
        return candidates[:6]

    def _package_modules(self) -> List[str]:
        """Source modules in the repository, tests and setup excluded."""
        found = self.tools.ls("**/*.py")
        if not found.ok:
            return []
        out: List[str] = []
        for line in found.text.splitlines():
            name = Path(line).name
            if name.startswith("test_") or name in {"setup.py", "conftest.py"}:
                continue
            out.append(line.strip())
        return out

    def inspect(self, state: TaskState, candidates: Sequence[str]) -> str:
        """Read the interesting part of each candidate file.

        A traceback frame points at one line, so the read is centred there --
        roughly a screen above and a screen below. Reading a 900-line module
        to find line 312 is how a context window gets spent on nothing.
        """
        view: List[str] = []
        by_path = {f.path: f for f in state.frames}
        for path in candidates[:3]:
            frame = by_path.get(path)
            start = max(1, (frame.line - 40)) if frame else 1
            end = (frame.line + 40) if frame else 160
            result = self.tools.read(path, start=start, end=end, budget=4000)
            if not result.ok:
                continue
            view.append(result.text)
            state.files_read.append(path)
            self._say(Step.READ, f"{path} lines {start}-{end}")
        return "\n\n".join(view)

    def verify(self, state: TaskState) -> bool:
        """Re-run the probes. Exit codes decide, not opinions."""
        if not state.probes:
            return False
        ok = True
        for probe in state.probes:
            result = self.tools.run(probe.command)
            state.last_output = result.text
            passed = result.ok and "Traceback (most recent call last)" not in result.text
            state.note(Step.VERIFY, f"verify {probe.command!r}", ok=passed)
            self._say(Step.VERIFY, f"{probe.command} -> {'pass' if passed else 'FAIL'}")
            if not passed:
                ok = False
                frames = self._in_repo(
                    parse_traceback(result.text, root_prefix=str(self.tools.root))
                )
                if frames:
                    state.frames = frames  # the new failure is the new evidence
        return ok

    # -- the loop ----------------------------------------------------------
    def run(self, state: TaskState) -> TaskState:
        """Solve, or report precisely why it could not be solved."""
        # 1. reproduce. A probe that already passes is not a solved task: it
        # means the bug is invisible to the probe, so the loop must fall back
        # to the issue text for localisation rather than claim success.
        if state.probes:
            reproduced = self.reproduce(state)
            if reproduced is True:
                state.note(
                    Step.LOCATE,
                    "probe already passes: the bug is not visible to it, "
                    "localising from the goal instead",
                    ok=False,
                )
                self._say(Step.VERIFY, "probe already passes; using the goal text")

        for iteration in range(1, self.max_iterations + 1):
            candidates = self.localise(state)
            if not candidates:
                state.note(Step.LOCATE, "could not localise: no frames, no symbols", ok=False)
                self._say(Step.STUCK, "nothing to read")
                return state

            view = self.inspect(state, candidates)
            if not view:
                state.note(Step.READ, "candidates could not be read", ok=False)
                return state

            if self.propose is None:
                # No proposer: an honest stop with the map already drawn.
                state.note(
                    Step.STUCK,
                    "localised but no edit proposed: " + ", ".join(candidates[:3]),
                    ok=False,
                )
                return state

            proposal = self.propose(state, self.tools)
            if not proposal:
                state.note(Step.EDIT, f"iteration {iteration}: no edit proposed", ok=False)
                return state

            # A decision may need more than one edit. Swapping two function
            # bodies is a single thought and two line replacements, and forcing
            # it through a one-edit interface loses half of it: the loop
            # re-localises after the first and never learns the second, leaving
            # the file in a state no single line can describe. Accepting a
            # sequence keeps one decision atomic without weakening the rule
            # that each edit inside it is small and exact.
            edits = _as_edits(proposal)
            if not edits:
                state.note(Step.EDIT, f"iteration {iteration}: unusable proposal", ok=False)
                return state
            failed = False
            # Apply in reverse document order, per file. A set of edits decided
            # against one state of the file can invalidate itself: swapping two
            # bodies means the first replacement makes the second `old` string
            # ambiguous or absent. Working bottom-up keeps every `old` valid
            # relative to the text it was chosen from.
            for path, old, new in _bottom_up(self.tools, edits):
                result = self.tools.edit(path, old, new)
                state.note(Step.EDIT, f"{path}: {result.text}", ok=result.ok)
                self._say(Step.EDIT, f"{path}: {result.text}")
                if not result.ok:
                    failed = True
                    break
            result = _EditOutcome(not failed, "")
            if not result.ok:
                # A failed edit is a failed prediction about the file; the loop
                # keeps the information and tries again with fresh evidence.
                if result.text.startswith("snippet not found"):
                    fresh = self.tools.read(path, budget=3000)
                    if fresh.ok:
                        state.last_output = fresh.text
                continue

            if self.verify(state):
                state.solved = True
                state.verified = True
                state.note(Step.DONE, f"solved in {iteration} edit(s)")
                self._say(Step.DONE, f"solved after {iteration} edit(s)")
                return state
            self._say(Step.EDIT, "verified: no; re-localising from the new failure")

        state.note(Step.STUCK, f"gave up after {self.max_iterations} iterations", ok=False)
        return state

    # -- reporting ---------------------------------------------------------
    def report(self, state: TaskState) -> str:
        """A human-readable account of what happened and where it stands."""
        lines = [f"goal: {state.goal}"]
        if state.frames:
            lines.append("stack:")
            for frame in state.frames[-4:]:
                src = f"  -> {frame.source}" if frame.source else ""
                lines.append(f"  {frame}{src}")
        if state.files_read:
            lines.append("read: " + ", ".join(dict.fromkeys(state.files_read)))
        for attempt in state.attempts[-6:]:
            mark = "ok" if attempt.ok else "!!"
            lines.append(f"  [{mark}] {attempt.step.value}: {attempt.detail}")
        verdict = (
            "SOLVED and verified"
            if state.solved and state.verified
            else "solved, unverified"
            if state.solved
            else "NOT solved"
        )
        lines.append(verdict)
        return "\n".join(lines)
