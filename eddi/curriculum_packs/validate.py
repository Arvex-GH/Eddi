"""Validate a curriculum pack before it is taught.

A pack is data, and data at scale rots quietly. Three failure modes matter
and none of them raise an error on their own:

* a duplicated ``(subject, relation)`` pair, where the second fact silently
  overwrites the first;
* a subject or object that is empty, or that contains a character the
  tokeniser will split, so the fact can never be retrieved;
* a subject that no question could ever name, which makes the fact
  unreachable even though it was written correctly.

Run it directly to check every pack::

    python3 -m eddi.curriculum_packs.validate
"""

from __future__ import annotations

import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

Fact = Tuple[str, str, str]

def check_facts(facts: Sequence[Fact], name: str = "pack") -> List[str]:
    """Return a list of human-readable problems. Empty means clean."""
    from eddi.language import tokenize

    problems: List[str] = []
    seen: Dict[Tuple[str, str], str] = {}
    for i, fact in enumerate(facts):
        if len(fact) != 3:
            problems.append(f"{name}[{i}]: expected 3 items, got {len(fact)}")
            continue
        subject, relation, obj = (str(x) for x in fact)
        if not subject or not relation or not obj:
            problems.append(f"{name}[{i}]: empty field in {fact!r}")
            continue
        for field, value in (("subject", subject), ("relation", relation), ("object", obj)):
            # Ask the real tokeniser rather than guessing at a charset: a value
            # must survive as one token or a lookup can never match it. This is
            # how '19.3_g_per_cm3' was caught, which splits at the decimal point.
            if len(tokenize(value)) != 1:
                problems.append(
                    f"{name}[{i}]: {field} {value!r} tokenises as "
                    f"{tokenize(value)}, so it cannot be retrieved as one symbol"
                )
        key = (subject, relation)
        if key in seen:
            problems.append(
                f"{name}[{i}]: duplicate {key} -- {seen[key]!r} would be "
                f"overwritten by {obj!r}"
            )
        seen[key] = obj
    return problems


def check_prose(prose: Iterable[str], name: str = "pack") -> List[str]:
    problems: List[str] = []
    for i, sentence in enumerate(prose):
        if not sentence or not sentence.strip():
            problems.append(f"{name}: prose[{i}] is empty")
        elif not sentence.rstrip().endswith((".", "?", "!")):
            problems.append(f"{name}: prose[{i}] does not end a sentence: {sentence[:50]!r}")
    return problems


def check_case_duplicates(packs: Dict[str, Sequence[Fact]]) -> List[str]:
    """One name must not exist in two spellings.

    The knowledge base keys on the exact symbol, so "DNA" and "dna" are two
    different entities: a fact taught to one is invisible to the other. That
    split is silent -- both look correct in the source -- and it cost a real
    benchmark failure, where "what is the shape of dna" could not see the
    (DNA, shape, double_helix) fact.
    """
    spellings: Dict[str, Dict[str, List[str]]] = {}
    for pack, facts in packs.items():
        for subject, _relation, obj in facts:
            for value in (subject, obj):
                spellings.setdefault(value.lower(), {}).setdefault(value, []).append(pack)
    problems: List[str] = []
    for lowered, variants in spellings.items():
        if len(variants) > 1:
            described = ", ".join(f"{v!r}" for v in sorted(variants))
            problems.append(
                f"{lowered!r} exists as {described}; the knowledge base treats "
                f"these as different entities"
            )
    return problems


def check_pack(facts: Sequence[Fact], prose: Sequence[str], name: str) -> List[str]:
    return check_facts(facts, name) + check_prose(prose, name)


def validate_all() -> int:
    """Check every pack in the registry. Returns the process exit code."""
    from eddi.curriculum import PACKS

    total_facts = 0
    total_problems = 0
    for pack_name, pack in PACKS.items():
        facts = pack.get("facts", [])
        prose = pack.get("prose", [])
        total_facts += len(facts)
        problems = check_pack(facts, prose, pack_name)
        total_problems += len(problems)
        status = "ok" if not problems else f"{len(problems)} problem(s)"
        print(f"  {pack_name:22} {len(facts):5} facts  {len(prose):4} prose   {status}")
        for problem in problems[:20]:
            print(f"      - {problem}")
    clashes = check_case_duplicates({name: pack.get("facts", []) for name, pack in PACKS.items()})
    if clashes:
        print(f"\n  {len(clashes)} name(s) spelled two ways across packs:")
        for clash in clashes:
            print(f"      - {clash}")
        total_problems += len(clashes)
    print(f"\n  {total_facts} facts across {len(PACKS)} packs, {total_problems} problem(s)")
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(validate_all())
