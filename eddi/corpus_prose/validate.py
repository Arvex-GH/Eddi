"""Check the prose corpus before it is trained on.

An n-gram model imitates whatever it is given, so defects in the corpus become
defects in the output. Five checks catch the ones that matter:

* **duplicates** -- a repeated sentence is trained twice and skews the model;
* **near-duplicates** -- almost identical sentences make the model confident
  about one phrasing and blind to others;
* **unsentence-like text** -- a fragment with no verb teaches the model that
  fragments are sentences;
* **non-ASCII** -- curly quotes and dashes are not in the vocabulary and split
  awkwardly at tokenisation;
* **too-short or too-long lines** -- a five word line has too little context to
  learn a transition from, and a very long line drifts off topic.

Run it directly::

    python3 -m eddi.corpus_prose.validate
"""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence

#: Words that make a line a sentence rather than a heading. A hand written
#: list was too small and flagged good sentences ("An IP address identifies a
#: machine"), so this is a suffix test plus the irregular forms that no suffix
#: rule can catch.
_VERB_HINTS = frozenset(
    """is are was were be been being has have had does do did can could will
    would should must may might fell fell rose grew became began came went
    took gave made sold held led won lost spread stood ran brought bought
    taught thought kept left met paid read sent set spent spoke built burnt
    """ .split()
    + """fallen risen grown become begun come gone taken given made sold held
    led won lost spread stood run brought bought taught thought kept left met
    paid read sent set spent spoken built burnt wrote written drove driven
    chose chosen broke broken drew drawn knew known saw seen""".split()
)
_VERB_SUFFIXES = ("s", "es", "ed", "ing", "ies", "ied")


def _looks_verbal(word: str) -> bool:
    """True if a word is plausibly a verb form.

    Deliberately permissive. This check exists to catch headings and labels
    ("Speed of light", "Chapter four"), not to grade grammar, and a false
    positive on a good sentence wastes more time than a false negative.
    """
    cleaned = word.strip(".,!?;:\"'()").lower()
    if not cleaned:
        return False
    if cleaned in _VERB_HINTS:
        return True
    if len(cleaned) > 3 and cleaned.endswith(_VERB_SUFFIXES):
        return True
    return False

MIN_WORDS = 6
MAX_WORDS = 55


def check_sentences(sentences: Sequence[str], name: str = "corpus") -> List[str]:
    """Return a list of human-readable problems. Empty means clean."""
    problems: List[str] = []
    seen: Dict[str, int] = {}

    for index, raw in enumerate(sentences):
        if not isinstance(raw, str):
            problems.append(f"{name}[{index}]: not a string ({raw!r})")
            continue
        text = raw.strip()
        if not text:
            problems.append(f"{name}[{index}]: empty line")
            continue
        if any(ord(ch) > 127 for ch in text):
            offending = sorted({ch for ch in text if ord(ch) > 127})
            problems.append(f"{name}[{index}]: non-ASCII {offending} in {text[:40]!r}")
        if not text.endswith((".", "?", "!")):
            problems.append(f"{name}[{index}]: does not end a sentence: {text[:40]!r}")
        words = text.split()
        if len(words) < MIN_WORDS:
            problems.append(f"{name}[{index}]: only {len(words)} words: {text!r}")
        if len(words) > MAX_WORDS:
            problems.append(f"{name}[{index}]: {len(words)} words, too long: {text[:40]!r}")
        lowered = " ".join(text.lower().split())
        if lowered in seen:
            problems.append(
                f"{name}[{index}]: duplicate of [{seen[lowered]}]: {text[:50]!r}"
            )
        else:
            seen[lowered] = index
        if not any(_looks_verbal(word) for word in words):
            problems.append(f"{name}[{index}]: no verb, reads like a fragment: {text[:50]!r}")

    return problems


def check_near_duplicates(
    sentences: Sequence[str], name: str = "corpus", threshold: float = 0.9
) -> List[str]:
    """Flag pairs that differ so little the model learns only one phrasing."""
    import difflib

    problems: List[str] = []
    seen: Dict[str, int] = {}
    for index, raw in enumerate(sentences):
        key = " ".join(raw.lower().split()).rstrip(".")
        if key in seen:
            continue
        seen[key] = index
    items = list(seen.items())
    for i, (text_i, index_i) in enumerate(items):
        for text_j, index_j in items[i + 1 :]:
            # cheap length gate before the quadratic similarity test
            if abs(len(text_i) - len(text_j)) > 25:
                continue
            ratio = difflib.SequenceMatcher(None, text_i, text_j).ratio()
            if ratio >= threshold:
                problems.append(
                    f"{name}: [{index_i}] and [{index_j}] are {ratio:.0%} identical: "
                    f"{text_i[:40]!r} vs {text_j[:40]!r}"
                )
    return problems


def validate_all() -> int:
    """Check every corpus module. Returns the process exit code."""
    from eddi.corpus_prose import MODULES
    from eddi.language import tokenize

    total_sentences = 0
    total_tokens = 0
    total_problems = 0
    for name, sentences in MODULES.items():
        problems = check_sentences(sentences, name)
        total_sentences += len(sentences)
        total_tokens += len(tokenize(" ".join(sentences)))
        total_problems += len(problems)
        status = "ok" if not problems else f"{len(problems)} problem(s)"
        print(f"  {name:20} {len(sentences):5} sentences  {status}")
        for problem in problems[:15]:
            print(f"      - {problem}")

    print(f"\n  {total_sentences} sentences, {total_tokens} tokens, {total_problems} problem(s)")
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(validate_all())
