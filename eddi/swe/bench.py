"""A local benchmark of real, verifiable software engineering tasks.

Why not SWE-bench directly
--------------------------
SWE-bench and Terminal-Bench are the right targets and the wrong starting
point. Both require downloading repositories and images (hundreds of
gigabytes), they are too slow for an edit-compile-run cycle, and a score on
them tells us *that* we are failing without telling us *which step* failed.

This suite is built the opposite way round. Each task is a real repository --
several files, a real bug, a real test suite -- and each one carries:

* a **goal** phrased the way an issue is phrased, so localisation is tested
* a **probe**: the exact command that fails before the fix and passes after
* a **hidden test** that the agent never sees, so it cannot game the probe

The hidden test is the important part. An agent that learns "run the probe and
declare victory" will pass our loop and fail SWE-bench. An agent that has to
make a test it has never read pass has learned the real skill.

Task shape
----------
Each task directory contains a small Python package and a test file. The bug
is always a genuine defect of the kind that appears in real issues: an
off-by-one, a wrong default, a broken guard, a mutation of shared state.

Everything is stdlib-only and finishes in milliseconds, so the whole suite can
run on every commit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class BenchTask:
    """One reproducible engineering task."""

    name: str
    goal: str
    files: Dict[str, str]
    #: Command that fails now and must pass afterwards.
    probe: str
    #: Test the agent never sees; the real pass/fail signal.
    hidden_test: str
    hidden_name: str = "test_hidden.py"
    tags: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# The tasks. Each bug is small, realistic and invisible unless you read code.
# ---------------------------------------------------------------------------

TASKS: List[BenchTask] = [
    BenchTask(
        name="off-by-one-chunk",
        goal=(
            "The chunker drops the last element of a list. "
            "chunk([1, 2, 3, 4, 5], 2) returns [[1, 2], [3, 4]] and loses 5."
        ),
        files={
            "pack/chunker.py": '''\
"""Split a sequence into fixed-size chunks."""


def chunk(items, size):
    """Split ``items`` into lists of at most ``size``."""
    if size <= 0:
        raise ValueError("size must be positive")
    out = []
    for start in range(0, len(items), size):
        out.append(items[start:start + size])
    return out[:-1]
''',
            "test_chunker.py": '''\
import unittest

from pack.chunker import chunk


class TestChunk(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(chunk([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_leaves_a_remainder(self):
        self.assertEqual(chunk([1, 2, 3], 2), [[1, 2], [3]])

    def test_empty(self):
        self.assertEqual(chunk([], 3), [])

    def test_single(self):
        self.assertEqual(chunk([1], 3), [[1]])


if __name__ == "__main__":
    unittest.main()
''',
        },
        probe="python3 -m unittest test_chunker -q",
        hidden_name="test_hidden.py",
        hidden_test='''\
import unittest

from pack.chunker import chunk


class TestChunkHidden(unittest.TestCase):
    def test_many_chunks(self):
        self.assertEqual(
            chunk(list(range(11)), 3),
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]],
        )

    def test_size_one(self):
        self.assertEqual(chunk([1, 2], 1), [[1], [2]])

    def test_size_larger_than_items(self):
        self.assertEqual(chunk([1, 2], 5), [[1, 2]])

    def test_does_not_mutate_input(self):
        data = [1, 2, 3]
        chunk(data, 2)
        self.assertEqual(data, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
''',
        tags=("off-by-one", "easy"),
    ),
    BenchTask(
        name="mutable-default",
        goal=(
            "accumulate() keeps state between calls. Calling accumulate('a') "
            "and then accumulate('b') returns 'ab' instead of 'b'."
        ),
        files={
            "pack/accumulate.py": '''\
"""Join a stream of strings into a running total."""


def accumulate(word, history=[]):
    """Append ``word`` to ``history`` and return the joined string."""
    history.append(word)
    return "".join(history)


def reset(history):
    """Clear a history list in place."""
    history.clear()
    return history
''',
            "test_accumulate.py": '''\
import unittest

from pack.accumulate import accumulate


class TestAccumulate(unittest.TestCase):
    def test_single_call(self):
        self.assertEqual(accumulate("a", []), "a")

    def test_two_calls_same_list(self):
        seen = []
        self.assertEqual(accumulate("a", seen), "a")
        self.assertEqual(accumulate("b", seen), "ab")

    def test_default_is_fresh(self):
        # No list passed: each call must start from nothing.
        self.assertEqual(accumulate("solo"), "solo")


if __name__ == "__main__":
    unittest.main()
''',
        },
        probe="python3 -m unittest test_accumulate -q",
        hidden_test='''\
import unittest

from pack.accumulate import accumulate


class TestAccumulateHidden(unittest.TestCase):
    def test_calls_are_independent(self):
        self.assertEqual(accumulate("a"), "a")
        self.assertEqual(accumulate("b"), "b")

    def test_explicit_list_still_works(self):
        seen = []
        accumulate("x", seen)
        self.assertEqual(seen, ["x"])

    def test_default_is_not_shared(self):
        first = accumulate("p")
        second = accumulate("q")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
''',
        tags=("mutable-default", "classic"),
    ),
    BenchTask(
        name="guard-clause-inverted",
        goal=(
            "Percentages above 100 are accepted, and negative ages are rejected "
            "only sometimes. The validation in parse_age has its condition the "
            "wrong way round."
        ),
        files={
            "pack/validate.py": '''\
"""Validation helpers for user supplied numbers."""


def parse_age(value):
    """Return ``value`` as an age, raising ValueError if implausible."""
    age = int(value)
    if age < 0:
        raise ValueError(f"age must be non-negative: {value}")
    if age >= 150:
        raise ValueError(f"age must be at most 150: {value}")
    return age


def parse_percent(value):
    """Return ``value`` as a percentage in [0, 100]."""
    pct = float(value)
    if pct < 0 or pct > 100:
        raise ValueError(f"percent must be between 0 and 100: {value}")
    return pct
''',
            "test_validate.py": '''\
import unittest

from pack.validate import parse_age, parse_percent


class TestValidate(unittest.TestCase):
    def test_percent_ok(self):
        self.assertEqual(parse_percent("50"), 50.0)

    def test_percent_high_rejected(self):
        with self.assertRaises(ValueError):
            parse_percent("150")

    def test_age_ok(self):
        self.assertEqual(parse_age("30"), 30)


if __name__ == "__main__":
    unittest.main()
''',
        },
        probe="python3 -m unittest test_validate -q",
        hidden_test='''\
import unittest

from pack.validate import parse_age, parse_percent


class TestValidateHidden(unittest.TestCase):
    def test_zero_age_is_valid(self):
        self.assertEqual(parse_age("0"), 0)

    def test_negative_age_rejected(self):
        with self.assertRaises(ValueError):
            parse_age("-1")

    def test_too_old_rejected(self):
        with self.assertRaises(ValueError):
            parse_age("200")

    def test_percent_bounds(self):
        self.assertEqual(parse_percent("0"), 0.0)
        self.assertEqual(parse_percent("100"), 100.0)


if __name__ == "__main__":
    unittest.main()
''',
        tags=("logic", "guard"),
    ),
    BenchTask(
        name="shared-cache-leak",
        goal=(
            "convert() returns wrong values when called with different units, "
            "because the unit cache is shared between instances of Converter. "
            "Converting 1 km then 1 m gives the kilometre answer both times."
        ),
        files={
            "pack/units.py": '''\
"""Length conversion with a per-instance unit table."""


class Converter:
    """Convert between length units."""

    _cache = {}

    def __init__(self, factors):
        self._cache = dict(factors)
        self.factors = factors

    def add(self, unit, metres):
        self._cache[unit] = metres

    def convert(self, value, unit):
        if unit not in self._cache:
            raise KeyError(f"unknown unit: {unit}")
        return value * self._cache[unit]


KM = {"km": 1000.0, "m": 1.0}
''',
            "test_units.py": '''\
import unittest

from pack.units import Converter, KM


class TestUnits(unittest.TestCase):
    def test_km(self):
        self.assertEqual(Converter(KM).convert(1, "km"), 1000.0)

    def test_m(self):
        self.assertEqual(Converter(KM).convert(1, "m"), 1.0)


if __name__ == "__main__":
    unittest.main()
''',
        },
        probe="python3 -m unittest test_units -q",
        hidden_test='''\
import unittest

from pack.units import Converter


class TestUnitsHidden(unittest.TestCase):
    def test_instances_are_independent(self):
        metric = Converter({"km": 1000.0})
        imperial = Converter({"mi": 1609.0})
        self.assertEqual(metric.convert(2, "km"), 2000.0)
        self.assertEqual(imperial.convert(2, "mi"), 3218.0)
        with self.assertRaises(KeyError):
            imperial.convert(1, "km")

    def test_add_does_not_leak(self):
        first = Converter({"m": 1.0})
        second = Converter({"m": 1.0})
        first.add("cm", 0.01)
        with self.assertRaises(KeyError):
            second.convert(1, "cm")


if __name__ == "__main__":
    unittest.main()
''',
        tags=("state", "classic"),
    ),
    BenchTask(
        name="binary-search-boundary",
        goal=(
            "find_index misses the last element of a sorted list. "
            "find_index([1, 3, 5, 7], 7) returns -1."
        ),
        files={
            "pack/search.py": '''\
"""Binary search over a sorted sequence."""


def find_index(items, target):
    """Return the index of ``target`` in sorted ``items``, or -1."""
    low, high = 0, len(items) - 1
    while low < high:
        mid = (low + high) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
''',
            "test_search.py": '''\
import unittest

from pack.search import find_index


class TestSearch(unittest.TestCase):
    def test_first(self):
        self.assertEqual(find_index([1, 3, 5, 7], 1), 0)

    def test_middle(self):
        self.assertEqual(find_index([1, 3, 5, 7], 3), 1)

    def test_absent(self):
        self.assertEqual(find_index([1, 3, 5, 7], 4), -1)

    def test_empty(self):
        self.assertEqual(find_index([], 1), -1)


if __name__ == "__main__":
    unittest.main()
''',
        },
        probe="python3 -m unittest test_search -q",
        hidden_test='''\
import unittest

from pack.search import find_index


class TestSearchHidden(unittest.TestCase):
    def test_last(self):
        self.assertEqual(find_index([1, 3, 5, 7], 7), 3)

    def test_single_present(self):
        self.assertEqual(find_index([9], 9), 0)

    def test_single_absent(self):
        self.assertEqual(find_index([9], 1), -1)

    def test_long_list(self):
        data = list(range(0, 100, 2))
        for target in data:
            self.assertEqual(find_index(data, target), data.index(target))
        self.assertEqual(find_index(data, 99), -1)


if __name__ == "__main__":
    unittest.main()
''',
        tags=("algorithm", "boundary"),
    ),
    BenchTask(
        name="retry-never-retries",
        goal=(
            "fetch() gives up immediately on a transient error even though it "
            "is supposed to retry twice. It raises on the first failure."
        ),
        files={
            "pack/retry.py": '''\
"""Call a function with retries on a transient failure."""


class Transient(Exception):
    """An error that may succeed if tried again."""


def fetch(fn, attempts=3, on_retry=None):
    """Call ``fn`` until it succeeds, up to ``attempts`` times in total."""
    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except Transient as exc:
            last = exc
            if on_retry is not None:
                on_retry(attempt)
            raise
    raise last if last is not None else RuntimeError("no attempts made")
''',
            "test_retry.py": '''\
import unittest

from pack.retry import Transient, fetch


class TestRetry(unittest.TestCase):
    def test_success_first_time(self):
        self.assertEqual(fetch(lambda: 1), 1)

    def test_stops_after_attempts(self):
        def always():
            raise Transient("nope")

        with self.assertRaises(Transient):
            fetch(always, attempts=3)

    def test_retries_a_transient_failure(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise Transient("not yet")
            return "ok"

        self.assertEqual(fetch(flaky), "ok")


if __name__ == "__main__":
    unittest.main()
''',
        },
        probe="python3 -m unittest test_retry -q",
        hidden_test='''\
import unittest

from pack.retry import Transient, fetch


class TestRetryHidden(unittest.TestCase):
    def test_recovers_on_second_attempt(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise Transient("first")
            return "ok"

        self.assertEqual(fetch(flaky), "ok")
        self.assertEqual(len(calls), 2)

    def test_attempt_count_exact(self):
        calls = []

        def always():
            calls.append(1)
            raise Transient("nope")

        with self.assertRaises(Transient):
            fetch(always, attempts=4)
        self.assertEqual(len(calls), 4)

    def test_on_retry_called(self):
        seen = []

        def flaky():
            if len(seen) < 1:
                raise Transient("once")
            return 7

        self.assertEqual(fetch(flaky, on_retry=seen.append), 7)
        self.assertEqual(seen, [0])


if __name__ == "__main__":
    unittest.main()
''',
        tags=("control-flow", "off-by-one"),
    ),
    BenchTask(
        name="stemmer-slice",
        goal=(
            "Searching for 'studies' no longer finds sentences "
            "containing 'study'. Singular and plural nouns ending in "
            "-y stop sharing a stem, so retrieval misses them. The "
            "plural rule in `_stem` strips one character too few."
        ),
        files={
            "pack/words.py": '''"""Reduce a word to the root shared by its inflections."""


def _stem(word):
    """Strip a plural suffix, keeping the root the singular also has."""
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-2]                            # studies -> study
    elif w.endswith("es") and len(w) > 4:
        w = w[:-1]                            # sizes -> size
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]                            # catalysts -> catalyst
    return w


def same_root(a, b):
    """True when two surface forms share a stem."""
    return _stem(a) == _stem(b)
''',
            "test_words.py": '''import unittest

from pack.words import _stem, same_root


class TestWords(unittest.TestCase):
    def test_plural_matches_singular(self):
        self.assertTrue(same_root("catalyst", "catalysts"))

    def test_word_is_not_mangled(self):
        self.assertEqual(_stem("speed"), "speed")


if __name__ == "__main__":
    unittest.main()
''',
        },
        probe="python3 -m unittest test_words -q",
        hidden_test='''import unittest

from pack.words import _stem, same_root


class TestWordsHidden(unittest.TestCase):
    def test_y_plurals_share_a_root(self):
        for singular, plural in (
            ("study", "studies"),
            ("theory", "theories"),
            ("penny", "pennies"),
        ):
            self.assertTrue(same_root(singular, plural), singular)

    def test_no_dangling_vowel_from_a_suffix(self):
        # "studies" once stemmed to "studi": the -ies suffix was stripped but
        # the y it stands for was never restored, so the root kept a vowel that
        # only exists in the plural.
        for plural in ("studies", "theories", "pennies"):
            self.assertFalse(_stem(plural).endswith("i"), plural)

    def test_other_plurals_still_work(self):
        self.assertTrue(same_root("size", "sizes"))
        self.assertTrue(same_root("catalyst", "catalysts"))


if __name__ == "__main__":
    unittest.main()
''',
        tags=("silent", "slice", "retrieval"),
    ),
]



# ---------------------------------------------------------------------------
# Running the suite
# ---------------------------------------------------------------------------

@dataclass
class TaskOutcome:
    name: str
    probe_failed_before: bool
    probe_passes_after: bool
    hidden_passes: bool
    iterations: int
    detail: str = ""

    @property
    def resolved(self) -> bool:
        """Resolved means the hidden test passes -- not that the probe does."""
        return self.hidden_passes


def _run(cmd: str, cwd: Path, timeout: float = 60.0) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd, shell=True, cwd=str(cwd), capture_output=True, text=True,
        timeout=timeout, stdin=subprocess.DEVNULL,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def materialise(task: BenchTask, destination: Path, hide_tests: bool = False) -> Path:
    """Write a task to disk. With ``hide_tests`` the visible tests are removed.

    Hiding the visible tests is what stops an agent from reading the expected
    values and patching to them. It must reason about behaviour, not about
    assertions -- exactly what SWE-bench requires, where the graded tests are
    never in the repository.
    """
    root = destination / task.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for rel, content in task.files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if hide_tests and Path(rel).name.startswith("test_"):
            continue
        path.write_text(content, encoding="utf-8")
    (root / "pack" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "pack" / "__init__.py").write_text("", encoding="utf-8")
    return root


def score_task(
    task: BenchTask,
    solve: "callable",
    workspace: Optional[Path] = None,
    hide_visible_tests: bool = True,
) -> TaskOutcome:
    """Run one task end to end: break it, solve it, grade it with the hidden test."""
    base = Path(workspace or tempfile.mkdtemp(prefix="eddi-bench-"))
    root = materialise(task, base, hide_tests=hide_visible_tests)

    # In hidden mode the visible test module does not exist, so the probe
    # cannot be the pass/fail signal there -- the hidden test is. The probe is
    # still run so its failure is recorded, and a missing module is not counted
    # as evidence that the bug was reproduced.
    probe = task.probe
    before_code, before_out = _run(probe, root)
    probe_failed_before = before_code != 0 and "ModuleNotFoundError" not in before_out

    iterations, detail = 0, ""
    try:
        result = solve(root)
        if isinstance(result, tuple):
            iterations, detail = result
        elif isinstance(result, int):
            iterations = result
    except Exception as exc:  # a crash is a failed task, not a failed suite
        detail = f"solver raised {type(exc).__name__}: {exc}"

    after_code, after_out = _run(probe, root)

    # Grade with the hidden test, copied in only now.
    hidden = root / task.hidden_name
    hidden.write_text(task.hidden_test, encoding="utf-8")
    hidden_code, hidden_out = _run(f"python3 -m unittest {task.hidden_name[:-3]} -q", root)
    if not detail:
        detail = hidden_out.strip().splitlines()[-1] if hidden_out.strip() else ""

    if workspace is None:
        shutil.rmtree(base, ignore_errors=True)

    probe_usable = "ModuleNotFoundError" not in after_out
    return TaskOutcome(
        name=task.name,
        probe_failed_before=probe_failed_before,
        probe_passes_after=after_code == 0 if probe_usable else False,
        hidden_passes=hidden_code == 0,
        iterations=iterations,
        detail=detail,
    )


def run_suite(solve, tasks: Optional[Sequence[BenchTask]] = None,
              hide_visible_tests: bool = True) -> List[TaskOutcome]:
    chosen = list(tasks if tasks is not None else TASKS)
    return [score_task(task, solve, hide_visible_tests=hide_visible_tests) for task in chosen]


def format_outcomes(outcomes: Sequence[TaskOutcome]) -> str:
    lines = []
    resolved = sum(1 for o in outcomes if o.resolved)
    for outcome in outcomes:
        mark = "PASS" if outcome.resolved else "fail"
        extra = "" if outcome.probe_failed_before else "  (probe did not fail before)"
        lines.append(
            f"  [{mark}] {outcome.name:24} hidden={'yes' if outcome.hidden_passes else 'no '}"
            f" probe={'yes' if outcome.probe_passes_after else 'no '}"
            f" edits={outcome.iterations}{extra}"
        )
        if not outcome.resolved and outcome.detail:
            lines.append(f"         {outcome.detail[:100]}")
    lines.append(f"  resolved {resolved}/{len(outcomes)}")
    return "\n".join(lines)
