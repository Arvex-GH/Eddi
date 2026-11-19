"""Prose corpus: the text Eddi learns to write from.

Facts teach Eddi *what is true*. Prose teaches it *how to say things*, which
is a separate skill and a separate kind of learning: an n-gram model trained
on 14k tokens can produce a grammatical clause and nothing longer, and its
held-out perplexity is worse than a bigram baseline.

This package holds that corpus, one module per domain, in the same spirit as
``eddi/curriculum_packs/`` for facts.

Why prose lives here and not in ``curriculum.PACKS``: the capability suite
builds an agent from every registered pack, and 100k tokens of prose would
make it slow for no benefit. The suite checks *answers*, which come from
facts; the corpus is used when building or training a brain.

Each module exposes ``SENTENCES: List[str]``. The rules for a good sentence
here are stricter than for the fact packs:

* it must be **true**, or at least a defensible generalisation;
* it must be **self-contained**, because a sentence quoted from memory has no
  surrounding context to explain it;
* it must read as **ordinary English**, because that is what the model learns
  to imitate -- a sentence that reads like a dictionary entry teaches the
  model to write dictionary entries.
"""

from __future__ import annotations

from typing import Dict, List

from . import everyday
from . import explanation
from . import history_stories
from . import nature
from . import people_ideas
from . import science_explained

MODULES: Dict[str, List[str]] = {
    "explanation": explanation.SENTENCES,
    "nature": nature.SENTENCES,
    "science_explained": science_explained.SENTENCES,
    "history_stories": history_stories.SENTENCES,
    "people_ideas": people_ideas.SENTENCES,
    "everyday": everyday.SENTENCES,
}


def all_sentences() -> List[str]:
    """Every corpus sentence, in a stable order."""
    out: List[str] = []
    for sentences in MODULES.values():
        out.extend(sentences)
    return out


def stats() -> Dict[str, int]:
    """Counts per module and in total, so growth is visible."""
    from eddi.language import tokenize

    per_module = {name: len(sentences) for name, sentences in MODULES.items()}
    tokens = len(tokenize(" ".join(all_sentences())))
    per_module["_total_sentences"] = sum(per_module.values())
    per_module["_total_tokens"] = tokens
    return per_module


if __name__ == "__main__":
    for name, value in stats().items():
        label = name.lstrip("_")
        print(f"  {label:22} {value:7}")
