"""Where does symbolic coverage end?

The five strategies solve seven tasks. That proves they cover seven bug
families; it says nothing about the eighth. This measures the boundary
honestly by presenting bugs whose fix has no structural signature the
strategies can name, and recording that the loop fails rather than guessing.

The result is the empirical definition of what the model has to supply.
"""

import tempfile
from pathlib import Path

from eddi.swe.bench import BenchTask, score_task
from eddi.swe.__main__ import _solver_factory

# A refactor, not a repair: two functions have had their bodies transposed.
# Nothing is off by one, no guard is inverted, no default is mutable. The only
# description of the defect is "these two blocks are in the wrong place", which
# is a shape the strategies do not know how to look for.
TRANSPOSED = '''def area(width, height):
    """Return the area of a rectangle."""
    return width + height + 2


def perimeter(width, height):
    """Return the distance around a rectangle."""
    return width * height
'''

TRANSPOSED_HIDDEN = '''import unittest

from shapes.geometry import area, perimeter


class TestGeometry(unittest.TestCase):
    def test_area_multiplies(self):
        self.assertEqual(area(3, 4), 12)

    def test_perimeter_adds_doubled(self):
        self.assertEqual(perimeter(3, 4), 14)


if __name__ == "__main__":
    unittest.main()
'''

# A missing branch: no existing case handles negative numbers, so a new
# precondition has to be introduced. The strategies edit existing lines; none
# of them inserts a statement that was never there.
MISSING_BRANCH = '''def shipping_cost(weight_kg):
    """Cost to ship a parcel, in euro."""
    base = 4.0
    if weight_kg > 10:
        return base + (weight_kg - 10) * 1.5
    return base
'''

MISSING_BRANCH_HIDDEN = '''import unittest

from shop.shipping import shipping_cost


class TestShipping(unittest.TestCase):
    def test_light_parcel(self):
        self.assertEqual(shipping_cost(2), 4.0)

    def test_heavy_parcel(self):
        self.assertEqual(shipping_cost(12), 7.0)

    def test_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            shipping_cost(-1)


if __name__ == "__main__":
    unittest.main()
'''

TASKS = [
    BenchTask(
        name="out-of-distribution-transposed",
        goal=(
            "`area` returns width + height + 2 and `perimeter` returns "
            "width * height. The two function bodies appear to have been "
            "swapped. Restore the correct behaviour of both."
        ),
        files={
            "shapes/geometry.py": TRANSPOSED,
            "test_geometry.py": "import unittest\n\nfrom shapes.geometry import area, perimeter\n\n\n"
            "class TestGeometry(unittest.TestCase):\n"
            "    def test_area_multiplies(self):\n"
            "        self.assertEqual(area(3, 4), 12)\n\n\n"
            'if __name__ == "__main__":\n    unittest.main()\n',
        },
        probe="python3 -m unittest test_geometry -q",
        hidden_test=TRANSPOSED_HIDDEN,
        tags=("out-of-distribution", "refactor"),
    ),
    BenchTask(
        name="out-of-distribution-missing-branch",
        goal=(
            "`shipping_cost` accepts a negative weight and returns the base "
            "price. A negative weight should raise `ValueError` instead."
        ),
        files={
            "shop/shipping.py": MISSING_BRANCH,
            "test_shipping.py": "import unittest\n\nfrom shop.shipping import shipping_cost\n\n\n"
            "class TestShipping(unittest.TestCase):\n"
            "    def test_light_parcel(self):\n"
            "        self.assertEqual(shipping_cost(2), 4.0)\n\n\n"
            'if __name__ == "__main__":\n    unittest.main()\n',
        },
        probe="python3 -m unittest test_shipping -q",
        hidden_test=MISSING_BRANCH_HIDDEN,
        tags=("out-of-distribution", "insertion"),
    ),
]


def main():
    solve = _solver_factory(use_proposer=True)
    print("bugs whose fix has no shape the strategies can name:")
    print()
    for task in TASKS:
        outcome = score_task(task, solve)
        verdict = "SOLVED" if outcome.hidden_passes else "unsolved"
        print(f"  {task.name:34} {verdict}")
        print(f"      edits attempted: {outcome.iterations}")
        if outcome.detail:
            print(f"      reported: {outcome.detail[:110]}")
        print()
    print("An unsolved task that reports why is worth more than a guess that")
    print("happens to pass: the first is a measured boundary, the second is")
    print("luck. These two mark the edge of what structure alone can reach,")
    print("and therefore the exact job the model is hired to do.")


if __name__ == "__main__":
    main()
