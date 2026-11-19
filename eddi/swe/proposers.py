"""Proposers: the one place in the loop that needs judgement.

The loop localises, reads, runs and verifies deterministically. What remains
is deciding *what edit to make*, and that is the only step that genuinely
requires a model. Keeping it behind a small interface has two benefits:

* the loop can be measured with no proposer at all, which tells us how much of
  the task the deterministic machinery already solves;
* several proposers can be compared on identical tasks, so a claim like "this
  is better" is a measurement rather than an opinion.

Two proposers live here. The heuristic one is deliberately not a language
model: it reasons over the code's structure and the failure evidence. If a
symbolic method can resolve a class of bugs, it should, because it costs
microseconds and never hallucinates. The language-model adapters in
`eddi.swe.model_adapter` handle the cases the heuristics cannot.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .loop import Frame, TaskState, symbols_in
from .tools import Toolbox, ToolResult


@dataclass
class Proposal:
    path: str
    old: str
    new: str
    reason: str = ""


# ---------------------------------------------------------------------------
# Structural inspection: read code as a program, not as text
# ---------------------------------------------------------------------------

@dataclass
class FunctionView:
    """A function's source, its line span, and its AST when it parses."""

    name: str
    path: str
    start: int
    end: int
    source: str
    tree: Optional[ast.AST] = None
    error: str = ""


def view_functions(toolbox: Toolbox, path: str) -> List[FunctionView]:
    """Parse a file and return every function with its exact source span.

    Exact spans are what make a precise edit possible: the whole body can be
    replaced by a compact snippet without guessing at whitespace.
    """
    read = toolbox.read(path, budget=200_000)
    if not read.ok:
        return []
    raw = (toolbox.root / path).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return []
    lines = raw.splitlines()
    views: List[FunctionView] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        views.append(
            FunctionView(
                name=node.name,
                path=path,
                start=start,
                end=end,
                source="\n".join(lines[start - 1: end]),
                tree=node,
            )
        )
    return views


def _raise_names(node: ast.AST) -> List[str]:
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            target = child.exc.func if isinstance(child.exc, ast.Call) else child.exc
            if isinstance(target, ast.Name):
                names.append(target.id)
    return names


def _duplicate_guards(view: FunctionView):
    """Find distinct `if` statements in one function with identical tests.

    A guard written twice is almost always the same condition pasted and then
    half-edited -- the classic inverted-comparison bug. Pairing each node with
    itself is not a duplicate, so identity is checked before the trees.
    """
    if view.tree is None:
        return []
    guards = [n for n in ast.walk(view.tree) if isinstance(n, ast.If)]
    pairs = []
    for index, node in enumerate(guards):
        marker = ast.dump(node.test)
        for other in guards[index + 1:]:
            if ast.dump(other.test) == marker:
                pairs.append((node, other))
    return pairs


# ---------------------------------------------------------------------------
# Heuristic proposer
# ---------------------------------------------------------------------------

_COMPARE_FLIP = {
    ast.Lt: ast.GtE, ast.Gt: ast.LtE, ast.LtE: ast.Gt, ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}

_CMP_TEXT = {
    ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not in",
    ast.Is: "is", ast.IsNot: "is not",
}

_OP_TEXT = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//",
    ast.Div: "/", ast.Mod: "%", ast.Pow: "**",
}


class HeuristicProposer:
    """Propose precise edits for bug families that can be recognised statically.

    Each strategy is a fix pattern with a *necessary* signature in the code.
    That is the discipline that keeps this honest: a strategy fires only when
    the structure of the function proves the bug is possible. Guessing produces
    diffs that look plausible and fail the tests, which is worse than stopping.

    Families handled:

    ``trailing-drop``   the last computed value is thrown away (``return x[:-1]``
                        where the loop already emitted every element)
    ``inverted-guard``  two guards in one function test the same way, so one of
                        them cannot mean what it says
    ``shared-mutable``  a default argument is a mutable literal, so state leaks
                        between calls
    ``search-boundary`` a binary search loop excludes the final element
    ``swallowed-error`` an ``except`` block returns when it should retry
    """

    name = "heuristic"

    def __init__(self) -> None:
        self.strategies = (
            self._trailing_drop,
            self._shared_mutable,
            self._suffix_slice_mismatch,
            self._inverted_guard,
            self._search_boundary,
            self._swallowed_error,
        )

    # -- entry point -------------------------------------------------------
    def __call__(self, state: TaskState, toolbox: Toolbox) -> Optional[Tuple[str, str, str]]:
        candidates: List[str] = []
        for frame in state.frames:
            if frame.path not in candidates:
                candidates.append(frame.path)
        for path in state.files_read:
            if path not in candidates:
                candidates.append(path)
        for path in candidates:
            if not path.endswith(".py"):
                continue
            for view in view_functions(toolbox, path):
                for strategy in self.strategies:
                    try:
                        proposal = strategy(view)
                    except Exception:  # a strategy must never break the loop
                        continue
                    if proposal is not None:
                        return (proposal.path, proposal.old, proposal.new)
        return None

    # -- strategies --------------------------------------------------------
    def _trailing_drop(self, view: FunctionView) -> Optional[Proposal]:
        """`return out[:-1]` after a loop that appended every element."""
        for node in ast.walk(view.tree) if view.tree else []:
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            value = node.value
            if not isinstance(value, ast.Subscript):
                continue
            sl = value.slice
            if not isinstance(sl, ast.Slice) or sl.lower is not None or sl.step is not None:
                continue
            if not (isinstance(sl.upper, ast.UnaryOp) and isinstance(sl.upper.op, ast.USub)
                    and isinstance(sl.upper.operand, ast.Constant)
                    and sl.upper.operand.value == 1):
                continue
            # The loop must genuinely cover every element, or the slice is right.
            covered = _loop_covers_input(view, value.value)
            if not covered:
                continue
            # Edit the whole `return ...` line so the replacement is unique in
            # the file; `out[:-1]` alone may also appear elsewhere.
            for row in view.source.splitlines():
                if "[:-1]" in row and row.lstrip().startswith("return"):
                    return Proposal(
                        view.path, row, row.replace("[:-1]", ""),
                        "loop already emits every chunk, so [:-1] discards the last",
                    )
        return None

    def _shared_mutable(self, view: FunctionView) -> Optional[Proposal]:
        """A mutable default argument accumulates state between calls."""
        node = view.tree
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        for arg, default in zip(node.args.args[-len(node.args.defaults):],
                                node.args.defaults):
            if not isinstance(default, (ast.List, ast.Dict, ast.Set)):
                continue
            literal = ast.get_source_segment(view.source, default) or "[]"
            replacement = {ast.List: "None", ast.Dict: "None", ast.Set: "None"}[
                type(default)
            ]
            # The signature must be rebuilt *with* the defaults, or the
            # substitution has nothing to replace and the strategy silently
            # never fires. Build each parameter as name=default first.
            parts = []
            defaults = dict(
                zip(
                    [a.arg for a in node.args.args[-len(node.args.defaults):]],
                    node.args.defaults,
                )
            )
            for parameter in node.args.args:
                if parameter.arg in defaults:
                    # get_source_segment needs the module the node came from;
                    # here the node belongs to the whole file while `view.source`
                    # is only the function, so it silently returns None and the
                    # signature never matches. unparse is exact and always works.
                    text = ast.unparse(defaults[parameter.arg])
                    parts.append(f"{parameter.arg}={text}")
                else:
                    parts.append(parameter.arg)
            old_sig = f"def {node.name}({', '.join(parts)})"
            new_parts = list(parts)
            for index, piece in enumerate(new_parts):
                if piece.startswith(f"{arg.arg}="):
                    new_parts[index] = f"{arg.arg}={replacement}"
            new_sig = f"def {node.name}({', '.join(new_parts)})"
            if new_sig == old_sig:
                continue
            # The body must actually mutate it, else the default is harmless.
            if not _mutates(node, arg.arg):
                continue
            guard = (
                f"    if {arg.arg} is None:\n"
                f"        {arg.arg} = {literal}\n"
            )
            return Proposal.with_body_guard(view, old_sig, new_sig, arg.arg, literal, guard)
        return None

    def _suffix_slice_mismatch(self, view: FunctionView) -> Optional[Proposal]:
        """A suffix rule whose slice cannot produce its own stated example.

        Inflection handling is written "if the word ends with S, strip len(S)
        and maybe append a letter". The classic defect is a slice that does not
        match the suffix it tested:

            if w.endswith("ies"):
                w = w[:-2]          # studies -> study

        `w[:-2]` leaves "studi", which is neither the word nor a prefix of it,
        and nothing raises -- so no traceback ever points here. Singular and
        plural simply stop matching, and retrieval loses half the vocabulary.

        The example in the trailing comment is the author's own statement of
        intent, which makes it the cheapest possible oracle: apply the rule's
        slice to the example's input and see whether the output is reproduced.
        """
        rows = view.source.splitlines()
        for node in view.tree.body if view.tree else []:
            if not isinstance(node, ast.If):
                continue
            tested = _endswith_argument(node.test)
            if not tested:
                continue
            example = _comment_example(rows, node, view.start)
            if example is None:
                continue
            before, after = example
            if not before.endswith(tested):
                continue
            for statement in node.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if not statement.targets or not isinstance(statement.targets[0], ast.Name):
                    continue
                var = statement.targets[0].id
                slice_len, appended = _slice_shape(statement.value)
                if slice_len is None:
                    continue
                produced = before[: len(before) - slice_len] + appended
                if produced == after:
                    continue  # the rule does what its example says
                # Find the slice that *would* produce the example.
                correct = _minimal_slice(before, after, tested)
                if correct is None:
                    continue
                slice_len, appended = correct
                for row in rows:
                    code = row.split("#", 1)[0]
                    if not code.strip().startswith(f"{var} ="):
                        continue
                    if f"[:-{slice_len}]" in code:
                        continue  # already correct
                    updated_code = re.sub(
                        r"\[:-\d+\](\s*\+\s*['\"][^'\"]*['\"])?",
                        f"[:-{slice_len}]" + (f" + {appended!r}" if appended else ""),
                        code,
                        count=1,
                    )
                    if updated_code == code:
                        continue
                    return Proposal(
                        view.path,
                        row,
                        updated_code + row[len(code):],
                        f"{tested!r} rule produces {produced!r} but the "
                        f"example says {after!r}",
                    )
        return None

    def _inverted_guard(self, view: FunctionView) -> Optional[Proposal]:
        """Two guards in one function test identically: one is inverted."""
        for first, second in _duplicate_guards(view):
            test_src = ast.get_source_segment(view.source, first.test)
            if not test_src:
                continue
            compare = _flip_compare_in_text(test_src)
            if compare is None:
                continue
            new_test = compare
            old_line = f"if {test_src}:"
            new_line = f"if {new_test}:"
            rows = view.source.splitlines()
            hits = [i for i, row in enumerate(rows) if old_line.strip() == row.strip()]
            if len(hits) < 2:
                continue
            # Flip the SECOND occurrence: the first guard usually states the
            # primary invariant, and the copy that was edited is the later one.
            target = hits[1]
            original_line = rows[target]
            rows[target] = original_line.replace(test_src, new_test, 1)
            return Proposal(
                view.path,
                original_line,
                rows[target],
                f"guard {test_src!r} appears twice; the second is inverted",
            )
        return None

    def _search_boundary(self, view: FunctionView) -> Optional[Proposal]:
        """`while low < high` never examines the final element."""
        src = view.source
        if "while low < high" not in src:
            return None
        if "high = mid - 1" not in src:
            return None
        if "mid + 1" not in src or "mid - 1" not in src:
            return None
        # Return the *line* that changes, not the whole function. A whole-file
        # replacement risks touching the wrong occurrence and produces a diff
        # too large to review; a single unique line is exactly the edit this
        # bug needs.
        line = next(
            (row for row in src.splitlines() if "while low < high" in row), None
        )
        if line is None:
            return None
        return Proposal(view.path, line, line.replace("while low < high", "while low <= high"),
                        "while low < high excludes the last element")

    def _swallowed_error(self, view: FunctionView) -> Optional[Proposal]:
        """An `except` that re-raises instead of retrying, inside a retry loop."""
        node = view.tree
        if node is None:
            return None
        has_loop = any(isinstance(child, ast.For) for child in ast.walk(node))
        if not has_loop:
            return None
        for child in ast.walk(node):
            if not isinstance(child, ast.ExceptHandler):
                continue
            body = child.body
            raising = any(isinstance(stmt, ast.Raise) for stmt in body)
            if not raising:
                continue
            # If the handler raises, the loop above it is decoration.
            old = "            raise\n"
            if old not in view.source:
                continue
            updated = view.source.replace(old, "            continue\n", 1)
            if updated == view.source:
                continue
            return Proposal(view.path, view.source, updated,
                            "handler re-raises inside a retry loop, so it never retries")
        return None


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    return "out"


def _loop_covers_input(view: FunctionView, target: ast.AST) -> bool:
    """True when a `for ... in range(0, len(x), n)` loop appended `x` slices."""
    if view.tree is None:
        return False
    for node in ast.walk(view.tree):
        if not isinstance(node, ast.For):
            continue
        call = node.iter
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
            and len(call.args) == 3
        ):
            continue
        start, stop, _step = call.args
        if not (isinstance(start, ast.Constant) and start.value == 0):
            continue
        if not (isinstance(stop, ast.Call) and isinstance(stop.func, ast.Name)
                and stop.func.id == "len"):
            continue
        return True
    return False


def _mutates(func: ast.AST, arg_name: str) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == arg_name:
                return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    base = target.value
                    if isinstance(base, ast.Name) and base.id == arg_name:
                        return True
    return False


def _flip_compare_in_text(text: str) -> Optional[str]:
    """Invert the single comparison operator in a guard expression."""
    for old, new in ((" >= ", " < "), (" <= ", " > "), (" > ", " <= "),
                     (" < ", " >= "), (" != ", " == "), (" == ", " != ")):
        if old in text:
            return text.replace(old, new, 1)
    return None


class Proposal:
    path: str
    old: str
    new: str
    reason: str = ""

    def __init__(self, path: str, old: str, new: str, reason: str = ""):
        self.path, self.old, self.new, self.reason = path, old, new, reason

    @classmethod
    def with_body_guard(cls, view, old_sig, new_sig, arg, literal, guard):
        """Replace the signature and inject a `None` guard as the first statement.

        The guard must land at the *top* of the body, after any docstring, and
        at exactly four spaces of indentation. Replacing the signature first and
        then inserting by line index keeps the edit small and reviewable, which
        is the same discipline the rest of this module follows.
        """
        source = view.source
        if old_sig not in source:
            return None
        mutated = source.replace(old_sig, new_sig, 1)
        rows = mutated.splitlines()
        signature_at = next(
            (i for i, row in enumerate(rows) if row.startswith("def ")), None
        )
        if signature_at is None:
            return None
        insert_at = signature_at + 1
        if insert_at < len(rows) and rows[insert_at].lstrip().startswith('"""'):
            # A one-line docstring closes on the same line; a multi-line one
            # closes later. Advancing unconditionally skipped the first body
            # statement and produced an IndentationError, so the guard has to
            # stop on whichever line actually closes the docstring.
            if rows[insert_at].lstrip().count('"""') >= 2:
                insert_at += 1
            else:
                insert_at += 1
                while insert_at < len(rows) and '"""' not in rows[insert_at]:
                    insert_at += 1
                insert_at += 1
        # Each guard line keeps its own relative indentation: the `if` sits at
        # four spaces and its body at eight. Stripping every row to four spaces
        # collapsed the nesting and produced an IndentationError.
        base = min(
            (len(row) - len(row.lstrip()) for row in guard.splitlines() if row.strip()),
            default=0,
        )
        guard_rows = [
            "    " + row[base:] for row in guard.rstrip("\n").splitlines()
        ]
        rows[insert_at:insert_at] = guard_rows
        return cls(
            view.path,
            source,
            "\n".join(rows) + "\n",
            "mutable default argument shared across calls",
        )


def _endswith_argument(test: ast.AST) -> Optional[str]:
    """The literal a guard tests with `.endswith(...)`, if it is one."""
    for node in ast.walk(test):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "endswith"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
    return None


def _comment_target(view: FunctionView, node: ast.AST) -> Optional[str]:
    """Read the "before -> after" example in the comment on a guard's line.

    The comment is not decoration here: it is the author's own statement of
    what the rule should do, which makes it the cheapest possible oracle.
    """
    line_no = getattr(node, "lineno", None)
    if line_no is None:
        return None
    rows = view.source.splitlines()
    for offset in (0, 1, -1):
        index = line_no - view.start + offset
        if 0 <= index < len(rows) and "#" in rows[index]:
            comment = rows[index].split("#", 1)[1]
            if "->" in comment:
                return comment.split("->", 1)[1].strip()
    return None


def _comment_example(rows: List[str], node: ast.AST, offset: int):
    """The (before, after) pair in the comment next to a guard, if present."""
    body_line = node.body[0].lineno if node.body else None
    if body_line is None:
        return None
    index = body_line - offset
    if not (0 <= index < len(rows)):
        return None
    row = rows[index]
    if "#" not in row or "->" not in row:
        return None
    comment = row.split("#", 1)[1]
    before, _, after = comment.partition("->")
    before, after = before.strip(), after.strip()
    if not before or not after:
        return None
    return before, after


def _slice_shape(value: ast.AST):
    """(characters stripped, appended text) for `w[:-k] + "x"` or `w[:-k]`."""
    appended = ""
    node = value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        right = node.right
        if not (isinstance(right, ast.Constant) and isinstance(right.value, str)):
            return (None, "")
        appended = right.value
        node = node.left
    if not isinstance(node, ast.Subscript):
        return (None, "")
    sl = node.slice
    if not isinstance(sl, ast.Slice) or sl.lower is not None or sl.step is not None:
        return (None, "")
    upper = sl.upper
    if not (isinstance(upper, ast.UnaryOp) and isinstance(upper.op, ast.USub)
            and isinstance(upper.operand, ast.Constant)):
        return (None, "")
    return (upper.operand.value, appended)


def _minimal_slice(before: str, after: str, tested: str):
    """The (strip, append) that turns ``before`` into ``after``.

    Preferred form is the natural one for an inflection rule: strip the whole
    tested suffix, then append whatever relates the root to the result.
    """
    if not before.endswith(tested):
        return None
    root = before[: len(before) - len(tested)]
    if after.startswith(root):
        return (len(tested), after[len(root):])
    return None

