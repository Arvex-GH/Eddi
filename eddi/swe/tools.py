"""Tools: the hands an agent needs to do software engineering.

A language model without tools can only describe a fix. Every capability gap
between "knows Python" and "resolves a GitHub issue" is closed here: reading
the right file, finding the symbol, editing exactly one location, running the
tests, and reading the result back as ground truth.

Design rules, each learned from watching real agents fail:

* Every tool returns a *budgeted* result. An unbounded `grep` over a large
  repository floods the context and the model loses the thread. Results are
  truncated with an explicit marker so the caller knows it was cut.
* Edits are exact-match and must be unique. A fuzzy edit that silently lands
  in the wrong place is worse than a failed edit, because the failure is
  invisible until much later.
* Nothing is trusted until it is run. `run()` returns stdout, stderr and the
  exit code, and that output -- not the model's belief -- is what the loop
  acts on.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Files never worth reading: they blow the budget and carry no meaning.
SKIP_DIRS = frozenset(
    """.git __pycache__ node_modules .venv venv env build dist .mypy_cache
    .pytest_cache .ruff_cache .tox htmlcov .idea .vscode target out bin obj
    .next .nuxt coverage""".split()
)

SKIP_SUFFIXES = (
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip",
    ".gz", ".tar", ".whl", ".pkl", ".npy", ".bin", ".lock",
)

#: Roughly four characters per token. Budgets are stated in characters to keep
#: the tool layer independent of whichever tokenizer the model uses.
DEFAULT_BUDGET = 6000


@dataclass
class ToolResult:
    """What a tool gives back: text for the model, and whether it worked."""

    ok: bool
    text: str
    meta: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


def _clip(text: str, budget: int) -> str:
    """Truncate to a character budget, saying so rather than silently cutting."""
    if len(text) <= budget:
        return text
    head = text[: int(budget * 0.75)]
    tail = text[-int(budget * 0.2):]
    dropped = len(text) - len(head) - len(tail)
    return f"{head}\n... [{dropped} characters omitted] ...\n{tail}"


class Toolbox:
    """File and shell tools rooted at one repository.

    The root is a hard boundary: a path that resolves outside it is refused.
    An agent that can write anywhere is an agent that can destroy the machine
    it is meant to be fixing, and there is no task that requires it.
    """

    def __init__(self, root: str, command_timeout: float = 120.0):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"not a directory: {root}")
        self.command_timeout = command_timeout
        self.writes: List[Tuple[str, str]] = []
        self.commands: List[str] = []

    # ------------------------------------------------------------- internals
    def _resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"path escapes the repository: {path}")
        return candidate

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)).replace(os.sep, "/")

    def _walk(self) -> Iterable[Path]:
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [
                d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
            ]
            for name in filenames:
                if name.endswith(SKIP_SUFFIXES):
                    continue
                yield Path(dirpath) / name

    # --------------------------------------------------------------- reading
    def ls(self, pattern: str = "*", limit: int = 200) -> ToolResult:
        """List files whose path matches a glob. `**/x` matches at any depth."""
        matches: List[str] = []
        for path in self._walk():
            rel = self._relative(path)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
                matches.append(rel)
        matches.sort()
        if not matches:
            return ToolResult(False, f"no files match {pattern!r}", {"count": 0})
        shown = matches[:limit]
        body = "\n".join(shown)
        if len(matches) > limit:
            body += f"\n... [{len(matches) - limit} more files]"
        return ToolResult(True, body, {"count": len(matches)})

    def read(
        self,
        path: str,
        start: int = 1,
        end: Optional[int] = None,
        budget: int = DEFAULT_BUDGET,
    ) -> ToolResult:
        """Read a file, or a 1-indexed line range, with line numbers.

        Line numbers matter: an edit proposal is far more reliable when the
        model has seen the exact numbering the edit will be described against.
        """
        target = self._resolve(path)
        if not target.is_file():
            return ToolResult(False, f"no such file: {path}")
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return ToolResult(False, f"cannot read {path}: {exc}")
        first = max(1, start)
        last = len(lines) if end is None else min(len(lines), end)
        numbered = "\n".join(
            f"{i:>5}  {lines[i - 1]}" for i in range(first, last + 1)
        )
        header = f"{path} (lines {first}-{last} of {len(lines)})"
        return ToolResult(
            True, f"{header}\n{_clip(numbered, budget)}", {"lines": len(lines)}
        )

    def grep(
        self,
        pattern: str,
        glob: str = "*",
        limit: int = 60,
        budget: int = DEFAULT_BUDGET,
        ignore_case: bool = True,
    ) -> ToolResult:
        """Search file contents. Returns `path:line: text` for each hit."""
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            return ToolResult(False, f"bad pattern {pattern!r}: {exc}")
        hits: List[str] = []
        files_hit = 0
        for path in self._walk():
            rel = self._relative(path)
            if glob != "*" and not (
                fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(path.name, glob)
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            found = False
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    found = True
                    if len(hits) >= limit:
                        break
            if found:
                files_hit += 1
            if len(hits) >= limit:
                hits.append(f"... [stopped at limit {limit}]")
                break
        if not hits:
            return ToolResult(False, f"no match for {pattern!r}", {"count": 0})
        return ToolResult(True, _clip("\n".join(hits), budget), {"count": len(hits)})

    def definitions(self, name: str, glob: str = "*.py") -> ToolResult:
        """Find where a symbol is defined, as opposed to merely mentioned."""
        pattern = (
            rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b"
            rf"|^\s*{re.escape(name)}\s*=\s*"
        )
        return self.grep(pattern, glob=glob, limit=20, ignore_case=False)

    # --------------------------------------------------------------- writing
    def write(self, path: str, content: str) -> ToolResult:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.writes.append((path, content))
        return ToolResult(True, f"wrote {len(content)} characters to {path}")

    def edit(
        self, path: str, old: str, new: str, count: int = 1
    ) -> ToolResult:
        """Replace an exact snippet. Refuses if it is not unique.

        Uniqueness is the whole safety property. An edit that matches twice
        and silently takes the first occurrence is how an agent "fixes" a bug
        in a file it never read, and the mistake surfaces only in review.
        """
        target = self._resolve(path)
        if not target.is_file():
            return ToolResult(False, f"no such file: {path}")
        text = target.read_text(encoding="utf-8")
        occurrences = text.count(old)
        if occurrences == 0:
            return ToolResult(False, f"snippet not found in {path}")
        if occurrences > count:
            return ToolResult(
                False,
                f"snippet appears {occurrences} times in {path}; "
                f"add surrounding context so it matches exactly once",
            )
        updated = text.replace(old, new, count)
        target.write_text(updated, encoding="utf-8")
        self.writes.append((path, updated))
        return ToolResult(
            True,
            f"edited {path}: replaced {occurrences} occurrence(s), "
            f"{len(text)} -> {len(updated)} characters",
        )

    # --------------------------------------------------------------- running
    def run(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        budget: int = DEFAULT_BUDGET,
    ) -> ToolResult:
        """Run a shell command inside the repository and capture everything.

        Output is the ground truth of the whole loop. A command that exits
        non-zero is not a failure of the tool -- it is information, and it is
        reported as such so the caller can decide.
        """
        # An escaping or missing directory is a refusable request, not a crash:
        # every other tool reports failure as a value so the loop can react.
        try:
            workdir = self._resolve(cwd) if cwd else self.root
        except ValueError as exc:
            return ToolResult(False, str(exc))
        if not workdir.is_dir():
            return ToolResult(False, f"no such directory: {cwd}")
        self.commands.append(command)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout or self.command_timeout,
                # A command that waits for input hangs forever in an agent loop.
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                False, f"timed out after {timeout or self.command_timeout}s: {command}"
            )
        except OSError as exc:
            return ToolResult(False, f"cannot run {command!r}: {exc}")
        parts = [f"$ {command}"]
        if proc.stdout.strip():
            parts.append(proc.stdout.rstrip())
        if proc.stderr.strip():
            parts.append("[stderr]\n" + proc.stderr.rstrip())
        parts.append(f"[exit {proc.returncode}]")
        return ToolResult(
            proc.returncode == 0,
            _clip("\n".join(parts), budget),
            {"returncode": proc.returncode},
        )

    # -------------------------------------------------------------- registry
    def spec(self) -> List[Dict[str, str]]:
        """A compact description of the tools, for a prompt or a model."""
        return [
            {"name": "ls", "args": "pattern", "does": "list files matching a glob"},
            {"name": "read", "args": "path[,start,end]", "does": "read a file with line numbers"},
            {"name": "grep", "args": "pattern[,glob]", "does": "search contents: path:line: text"},
            {"name": "definitions", "args": "name", "does": "find where a symbol is defined"},
            {"name": "edit", "args": "path, old, new", "does": "replace a unique snippet"},
            {"name": "write", "args": "path, content", "does": "create or overwrite a file"},
            {"name": "run", "args": "command[,cwd]", "does": "run a shell command"},
        ]
