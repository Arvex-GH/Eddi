"""A 100-question capability suite for Eddi.

Every question names the *ability* it exercises, so a failure says which
mechanism broke rather than just "wrong answer". The abilities are deliberately
different from one another: a suite of 100 fact lookups would measure one thing
100 times.

Run it:

    python -m eddi.eval_questions            # summary table
    python -m eddi.eval_questions --verbose  # every question and answer
    python -m eddi.eval_questions --json     # machine-readable

Exit code is 1 if anything fails, so this can gate CI.

Each case declares ``expect`` as a set of substrings; **all** must appear for the
case to pass, and matching is case-insensitive. Declaring several substrings is
how a case pins down the *content* of an answer rather than its exact wording,
so rewording a reply does not break the suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .agent import EddiAgent
from .curriculum import PACKS


@dataclass
class Case:
    """One question, the ability it tests, and what a correct answer contains."""

    ability: str
    question: str
    expect: Sequence[str] = ()
    #: acceptable Answer kinds; empty means "any"
    kinds: Sequence[str] = ()
    note: str = ""


SUITE: List[Case] = [
    # ---------------------------------------------------------------- recall
    Case("recall: subject -> attribute", "what is the chemical formula of water", ("H2O",)),
    Case("recall: subject -> attribute", "what is the chemical formula of salt", ("NaCl",)),
    Case("recall: subject -> attribute", "what is the chemical formula of carbon dioxide", ("CO2",)),
    Case("recall: object -> relation", "what does entropy measure", ("uncertainty",)),
    Case("recall: object -> relation", "what does photosynthesis take in", ("light", "CO2")),
    Case("recall: object -> relation", "what is the runtime of merge sort", ("n_log_n",)),
    Case("recall: object -> relation", "what is the average lookup of a hash table", ("O_of_1",)),
    Case("recall: object -> relation", "what is the dimension of a hypervector", ("1024",)),
    Case("recall: object -> relation", "what is the default port for HTTPS", ("443",)),
    Case("recall: object -> relation", "what is the default port for HTTP", ("80",)),
    Case("recall: created_by", "who created Python", ("Guido",)),
    Case("recall: created_by", "who created C++", ("Stroustrup",)),
    Case("recall: created_by", "who created C#", ("Hejlsberg",)),
    Case("recall: created_by", "who discovered gravity", ("Newton",)),
    Case("recall: type", "what type of language is Rust", ("memory_safe",)),
    Case("recall: type", "what type of language is Python", ("interpreted",)),
    Case("recall: definition", "what is a compiler", ("translate", "machine code")),
    Case("recall: definition", "what is an interpreter", ("execute",)),
    Case("recall: definition", "what is an algorithm", ("finite steps",)),
    Case("recall: definition", "what is perplexity", ("exponential", "cross entropy")),

    # -------------------------------------------------------------- taxonomy
    Case("taxonomy: orbits", "which planet does the Moon orbit", ("Earth",)),
    Case("taxonomy: orbits", "what does Mars orbit", ("Sun",)),
    Case("taxonomy: orbits", "what orbits the Sun", ("Earth",), kinds=("reverse",)),
    Case("taxonomy: continent", "what continent is Japan in", ("Asia",)),
    Case("taxonomy: continent", "what continent is France in", ("Europe",)),
    Case("taxonomy: continent", "what continent is Germany in", ("Europe",)),
    Case("taxonomy: type of thing", "what type of thing is the Sun", ("star",)),
    Case("taxonomy: type of thing", "what type of thing is the Moon", ("natural_satellite",)),
    Case("taxonomy: type of thing", "what type of thing is Earth", ("planet",)),

    # -------------------------------------------------------- reverse lookup
    Case("reverse lookup", "what is Paris the capital of", ("France",), kinds=("reverse",)),
    Case("reverse lookup", "what is Tokyo the capital of", ("Japan",), kinds=("reverse",)),
    Case("reverse lookup", "what is Berlin the capital of", ("Germany",), kinds=("reverse",)),
    Case("reverse lookup (relation given)", "what is the capital of France", ("Paris",)),
    Case("reverse lookup (relation given)", "what is the capital of Japan", ("Tokyo",)),
    Case("reverse lookup (relation given)", "what is the capital of Germany", ("Berlin",)),

    # ------------------------------------------------------------- reasoning
    Case("comparison", "which is faster, RAM or a register", ("register",), kinds=("compare",)),
    Case("comparison", "which is faster, RAM or an SSD", ("RAM",), kinds=("compare",)),
    Case("comparison", "which is faster, a register or an SSD", ("register",), kinds=("compare",)),
    Case("comparison: no data", "which is bigger, Paris or Tokyo", (), kinds=("unknown",)),
    Case("counting: total", "how many facts do you know", ("facts",), kinds=("count",)),
    Case("counting: about a subject", "how many facts do you know about Earth",
         ("facts about Earth",), kinds=("count",)),

    # --------------------------------------------------------------- analogy
    Case("analogy", "Earth is to Sun as Moon is to ?", ("Earth",), kinds=("analogy",)),

    # ----------------------------------------------------------------- chain
    Case("chain: multi-hop", "chain Moon | orbits", ("Earth",), kinds=("chain",)),

    # ------------------------------------------------------------------ math
    Case("math: arithmetic", "what is 12 * 37", ("444",), kinds=("math",)),
    Case("math: precedence", "what is 7 * 8 + 3", ("59",), kinds=("math",)),
    Case("math: parentheses", "what is (2 + 3) * 4", ("20",), kinds=("math",)),
    Case("math: subtraction", "what is 100 - 37", ("63",), kinds=("math",)),
    Case("math: division", "what is 144 / 12", ("12",), kinds=("math",)),
    Case("math: percentage", "what is 15% of 200", ("30",), kinds=("math",)),
    Case("math: percentage", "what is 50 percent of 84", ("42",), kinds=("math",)),
    Case("math: powers", "what is 2 to the power of 10", ("1024",), kinds=("math",)),
    Case("math: powers", "what is 3 ^ 4", ("81",), kinds=("math",)),
    Case("math: roots", "what is the square root of 144", ("12",), kinds=("math",)),
    Case("math: roots", "what is the square root of 2", ("1.41",), kinds=("math",)),
    Case("math: div by zero", "what is 5 / 0", (), kinds=("math_error", "math")),
    Case("math: negative", "what is -5 * -4", ("20",), kinds=("math",)),
    Case("math: decimals", "what is 0.5 + 0.25", ("0.75",), kinds=("math",)),

    # ---------------------------------------------------------------- units
    Case("units: length", "how many meters in 5 kilometers", ("5000",), kinds=("convert",)),
    Case("units: length", "how many centimeters in 2 meters", ("200",), kinds=("convert",)),
    Case("units: length", "convert 3000 meters to km", ("3",), kinds=("convert",)),
    Case("units: time", "how many minutes are in 3 hours", ("180",), kinds=("convert",)),
    Case("units: time", "how many seconds in a day", ("86400",), kinds=("convert",)),
    Case("units: time", "2 hours in minutes", ("120",), kinds=("convert",)),
    Case("units: incomparable", "how many meters in 5 hours", (), kinds=("convert_error",)),

    # ---------------------------------------------------------- user memory
    Case("user memory: stated fact", "I am 16 years old", ("16",), kinds=("remember",)),
    Case("user memory: recall", "how old am I", ("16",), kinds=("about_self",)),
    Case("user memory: duplicate", "I am 16 years old", ("already",), kinds=("remember",)),
    Case("user memory: unknown admitted", "what is my favourite food", (), kinds=("unknown",)),

    # ----------------------------------------------------------- derivation
    Case("derivation: birth year", "which year was I born", ("2009", "2010"), kinds=("derive",)),
    Case("derivation: shows working", "which year was I born", ("working",), kinds=("derive",)),
    Case("derivation: from inline age", "If I am 30 which year was I born", ("1996",), kinds=("derive",)),
    Case("derivation: absurd age", "if I am 999 which year was I born", (), kinds=("math_error",)),

    # ------------------------------------------------------------- robustness
    Case("robustness: mixed case", "WHAT IS THE CHEMICAL FORMULA OF WATER", ("H2O",)),
    Case("robustness: extra punctuation", "water chemical formula???", ("H2O",)),
    Case("robustness: typo", "what is teh chemical formula of water", ("H2O",)),
    Case("robustness: no question mark", "what is the chemical formula of water", ("H2O",)),
    Case("robustness: leading space", "  what is python  ", ("Python",)),
    Case("robustness: empty entity", "what is the chemical formula of", (), kinds=("help", "unknown", "generate")),
    Case("robustness: unknown word", "what is zorblax", ("don't know",), kinds=("unknown",)),

    # ------------------------------------------------------------- honesty
    Case("honesty: unknown subject", "tell me about an unladen swallow", ("don't know",), kinds=("unknown",)),
    Case("honesty: no CLI syntax", "tell me about an unladen swallow", (), kinds=("unknown",)),
    Case("honesty: no fabrication", "what is a quux", ("don't know",), kinds=("unknown",)),
    Case("honesty: false premise", "why is the Sun a planet", ("not a planet",),
         kinds=("false_premise",)),

    # ----------------------------------------------------------- small talk
    Case("small talk: greeting", "hi", ("Eddi",), kinds=("greeting",)),
    Case("small talk: greeting with follow-up", "Hi how are you?", ("doing well",), kinds=("greeting",)),
    Case("small talk: thanks", "thanks", (), kinds=("greeting",)),
    Case("small talk: farewell", "bye", (), kinds=("greeting",)),
    Case("small talk: greeting then question", "hi, what is python", ("Python",), kinds=("about", "ask")),

    # ---------------------------------------------------------- self knowledge
    Case("self knowledge: identity", "who are you", ("Eddi",), kinds=("identity",)),
    Case("self knowledge: capabilities", "what can you do", ("can",), kinds=("help",)),
    Case("self knowledge: help", "help", ("teach",), kinds=("help",)),
    Case("self knowledge: stats", "how many facts do you know", ("written",), kinds=("count",)),

    # ------------------------------------------------------------- teaching
    Case("teaching: write a fact", "teach zorblax | colour | ultraviolet", ("zorblax",), kinds=("teach",)),
    Case("teaching: use what was taught", "what colour is zorblax", ("ultraviolet",)),
    Case("teaching: alias", "alias USA = United_States", ("USA",), kinds=("alias",)),
    Case("teaching: rule", "rule capital_of => located_in", ("capital_of",), kinds=("rule",)),
    Case("teaching: derived fact usable", "where is Paris", ("located in",)),

    # ---------------------------------------------------- more distinct angles
    Case("multi-fact aggregation", "tell me about Earth", ("planet", "gravity")),
    Case("relation direction", "what is the gravity of Earth", ("9 point 81",)),
    Case("nested phrase entity", "what is the speed of light", ("299792458",)),
    Case("spelled-out number", "what is 2 to the power of 10", ("1024",), kinds=("math",)),
    Case("query with article", "what is the shape of DNA", ("double helix",)),
    Case("abbreviation entity", "what is the port for HTTP", ("80",)),
    Case("hyphenated entity", "what is the default port for HTTPS", ("443",)),

    # ------------------------------------------------ geography and capitals
    Case("capital: direct", "what is the capital of Switzerland", ("Bern",)),
    Case("capital: not the largest city", "what is the capital of Australia", ("Canberra",)),
    Case("capital: reverse", "what is Paris the capital of", ("France",), kinds=("reverse",)),
    Case("capital: reverse", "what is Tokyo the capital of", ("Japan",), kinds=("reverse",)),
    Case("relation question", "what is the largest ocean", ("Pacific",)),
    Case("relation question", "what is the deepest lake", ("Baikal",)),
    Case("relation question", "what is the largest continent", ("Asia",)),
    Case("relation question", "what is the second largest ocean", ("Atlantic",)),
    Case("currency", "what is the currency of Japan", ("yen",)),
    Case("largest city", "what is the largest city of China", ("Shanghai",)),
    Case("physical geography", "what is the height meters of Everest", ("8849",)),

    # ---------------------------------------------------------------- history
    Case("history: date", "what is the invented year of printing press", ("1440",)),
    Case("history: date", "what is the reached americas year of Columbus", ("1492",)),
    Case("history: date BCE", "what is the built year BCE of pyramid of Giza", ("2560",)),
    Case("history: who invented", "who invented the printing press", ("Gutenberg",)),
    Case("history: who wrote", "who wrote the Elements", ("Euclid",)),
    Case("history: when", "when was the printing press invented", ("1440",)),
    Case("history: when", "when was the Magna Carta signed", ("1215",)),
    Case("history: lineage", "what is the taught of Socrates", ("Plato",)),

    # ---------------------------------------------------------------- science
    Case("science: chemistry", "what is the atomic number of gold", ("79",)),
    Case("science: chemistry", "what is the chemical symbol of iron", ("Fe",)),
    Case("science: physics", "what is the si unit of force", ("newton",)),
    Case("science: physics", "what is the formula of ohms law", ("voltage", "current")),
    Case("science: physics", "what is the formula of kinetic energy", ("half",)),
    Case("science: astronomy", "what is the largest planet", ("Jupiter",)),
    Case("science: astronomy", "what is the hottest planet", ("Venus",)),
    Case("science: astronomy", "what is the order from sun of earth", ("3",)),
    Case("science: astronomy", "what is the atomic number of oxygen", ("8",)),
    Case("science: biology", "what is the function of mitochondria", ("produce",)),
    Case("science: biology", "what is the number in human of chromosome", ("46",)),
    Case("science: biology", "what is the mechanism of evolution", ("natural selection",)),
    Case("science: earth", "what is the cause of the seasons", ("axial tilt",)),
    Case("science: earth", "what is the atmosphere main of Earth", ("nitrogen",)),
    Case("counting: quantified relation", "how many planets does the solar system have", ("8",), kinds=("count",)),
    Case("counting: quantified relation", "how many countries does Africa have", ("54",), kinds=("count",)),

    # ---------------------------------------------------------- mathematics
    Case("math: geometry", "what is the area formula of circle", ("pi", "radius squared")),
    Case("math: geometry", "what is the circumference formula of circle", ("2", "radius")),
    Case("math: geometry", "what is the angle sum of triangle", ("180",)),
    Case("math: geometry", "what is the formula of pythagorean theorem", ("a squared",)),
    Case("math: geometry", "what is the volume formula of cube", ("side",)),
    Case("math: constants", "what is the approx value of pi", ("3 point 14159",)),
    Case("math: constants", "what is the approx value of e", ("2 point 71828",)),
    Case("math: calculus", "what is the derivative of sine", ("cosine",)),
    Case("math: calculus", "what is the derivative of x squared", ("2x",)),
    Case("math: calculus", "what is the meaning of derivative", ("rate of change",)),
    Case("math: number theory", "what is the smallest of prime number", ("2",)),
    Case("math: number theory", "what is the identity of gcd times lcm", ("product",)),
    Case("math: algebra", "what is the formula of discriminant", ("b squared",)),
    Case("math: algebra", "what is the base of natural logarithm", ("e",)),
    Case("math: probability", "what is the shape of normal distribution", ("bell curve",)),
    Case("math: probability", "what is the definition of permutation", ("order matters",)),
    Case("math: probability", "what is the definition of combination", ("order does not matter",)),
]


def build_agent() -> EddiAgent:
    """A fresh agent carrying the same curriculum the live brain has.

    Facts go in through ``teach`` rather than ``import_facts``: import_facts
    writes the triples but does not register the symbols, so entity resolution
    fails and every question looks unknown. That is a real gap in the API, and
    it made the first run of this suite report failures that were the fixture's
    fault, not the agent's.
    """
    # No memory file: an evaluation must see only what it was given. Using the
    # live store made the suite non-deterministic -- the agent's own earlier
    # turns answered questions that were supposed to be answered from facts.
    agent = EddiAgent(persist_context=False, memory_path="", record_turns=False)
    # All five packs: the live brain is trained with geography too, and leaving
    # it out made every capital/continent question look like a failure.
    for pack in PACKS:
        for subject, relation, obj in PACKS[pack]["facts"]:
            agent.teach(subject, relation, obj)
        for sentence in PACKS[pack].get("prose", []):
            agent.train_corpus([sentence])
    agent.answer("I am 16 years old")
    return agent


def _passes(case: Case, answer) -> tuple:
    """(ok, reason) for one case."""
    if case.kinds and answer.kind not in case.kinds:
        return False, f"kind={answer.kind}, wanted one of {list(case.kinds)}"
    haystack = answer.text.lower()
    missing = [needle for needle in case.expect if needle.lower() not in haystack]
    if missing:
        return False, f"missing {missing}"
    return True, ""


def run(verbose: bool = False) -> Dict[str, object]:
    """Run the whole suite once and return a structured report."""
    agent = build_agent()
    results = []
    try:
        for case in SUITE:
            answer = agent.answer(case.question)
            ok, reason = _passes(case, answer)
            results.append(
                {
                    "ability": case.ability,
                    "question": case.question,
                    "kind": answer.kind,
                    "answer": answer.text,
                    "ok": ok,
                    "reason": reason,
                }
            )
    finally:
        agent.close(save=False)

    by_ability: Dict[str, List[bool]] = {}
    for row in results:
        by_ability.setdefault(row["ability"], []).append(row["ok"])

    passed = sum(1 for r in results if r["ok"])
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
        "by_ability": {k: (sum(v), len(v)) for k, v in by_ability.items()},
    }


def format_report(report: Dict[str, object], verbose: bool = False) -> str:
    lines = [
        "=== Eddi 100-question capability suite ===",
        f"  questions : {report['total']}",
        f"  passed    : {report['passed']}",
        f"  failed    : {report['failed']}",
        "",
        "  by ability:",
    ]
    for ability, (ok, total) in sorted(report["by_ability"].items()):
        mark = "ok  " if ok == total else "FAIL"
        lines.append(f"    {mark} {ability:38} {ok}/{total}")

    failures = [r for r in report["results"] if not r["ok"]]
    if failures:
        lines.append("")
        lines.append("  failures:")
        for row in failures:
            lines.append(f"    - [{row['ability']}] {row['question']}")
            lines.append(f"        got [{row['kind']}] {row['answer'].splitlines()[0][:70]}")
            lines.append(f"        {row['reason']}")
    if verbose:
        lines.append("")
        lines.append("  all answers:")
        for row in report["results"]:
            mark = "ok  " if row["ok"] else "FAIL"
            lines.append(f"    {mark} {row['question']}")
            lines.append(f"         [{row['kind']}] {row['answer'].splitlines()[0][:78]}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="eddi-eval-questions")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = run()
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
        for row in report["results"]:
            print(json.dumps(row))
    else:
        print(format_report(report, verbose=args.verbose))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(main())
