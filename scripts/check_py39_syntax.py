"""Fail if any shipped module uses syntax newer than Python 3.9.

The package declares ``requires-python >= 3.9`` but CI runs on 3.10/3.12
because ``actions/setup-python`` cannot reliably provision 3.9. This gate
keeps the promise honest without depending on that download.
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    bad: list[str] = []
    for path in sorted((ROOT / "eddi").rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
        except SyntaxError as exc:
            bad.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    if bad:
        print("Python 3.9 incompatible syntax:")
        for line in bad:
            print("  " + line)
        return 1
    print("all modules parse as Python 3.9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
