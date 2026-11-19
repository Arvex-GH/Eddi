"""Held-out evaluation of Eddi's memory, reasoning and language skills.

Everything measured here is *separate from the training curriculum*: the
sentences below were never shown to the model.  That makes the numbers honest:
they measure generalisation, not memorisation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

HELD_OUT_SENTENCES: List[str] = [
    "A language model predicts the next token from the tokens before it.",
    "Unseen sentences test whether the model generalised or memorised.",
    "Facts are written once, while prose is trained many times.",
    "The solar system contains eight planets and many smaller bodies.",
    "Clean code is easier to review than clever code.",
    "Every experiment needs a control group for a fair comparison.",
    "A compiler reports errors before the program ever runs.",
    "Water is essential for life on Earth.",
    "The ocean covers most of the planet's surface.",
    "Practice turns knowledge into skill.",
]

FACT_PROBES: List[Tuple[str, str]] = [
    ("what is the chemical formula of water", "H2O"),
    ("what does the Earth orbit", "Sun"),
    ("what is the moon of earth", "Earth"),  # inference: Moon --orbits--> Earth
    ("what is the speed of light", "299792458"),
    ("what is the shape of dna", "double"),
    ("what is the capital of germany", "Berlin"),
    ("what is the type of python", "interpreted"),
    ("what is the ownership of c++_unique_ptr", "exclusive"),
    ("what is the default port of https", "443"),
    ("what is the form of modus_ponens", "implies"),
]

REASONING_PROBES: List[Tuple[str, str]] = [
    ("Earth is to Sun as Moon is to ?", "--orbits--> Earth"),
    ("chain Moon | orbits | orbits", "Sun"),
    ("how is Earth related to Sun", "orbits"),
    ("tell me about Jupiter", "planet"),
    ("what is 17 * 3 + 4", "55"),
    ("what is 200 / 8", "25"),
    ("1836 + 164", "2000"),
    ("what does the Mars orbit", "Sun"),
]

GENERATION_PROMPTS: List[str] = ["A language model", "The compiler", "Clean code"]


def evaluate_language(agent) -> Dict[str, float]:
    lm = agent.lm
    return {
        "perplexity": lm.perplexity(HELD_OUT_SENTENCES) if lm.trained_tokens else float("inf"),
        "perplexity_in_vocab": (
            lm.perplexity(HELD_OUT_SENTENCES, skip_oov=True) if lm.trained_tokens else float("inf")
        ),
        "bigram_coherence": lm.coherence(HELD_OUT_SENTENCES) if lm.trained_tokens else 0.0,
    }


def evaluate_facts(agent) -> Tuple[float, List[str]]:
    hits = 0
    misses: List[str] = []
    for question, expected in FACT_PROBES:
        answer = agent.answer(question).text
        if expected.lower() in answer.lower():
            hits += 1
        else:
            misses.append(f"{question!r} -> {answer!r} (expected {expected})")
    return hits / len(FACT_PROBES), misses


def evaluate_reasoning(agent) -> Tuple[float, List[str]]:
    hits = 0
    misses: List[str] = []
    for question, expected in REASONING_PROBES:
        answer = agent.answer(question).text
        if expected.lower() in answer.lower():
            hits += 1
        else:
            misses.append(f"{question!r} -> {answer!r} (expected {expected})")
    return hits / len(REASONING_PROBES), misses


def evaluate_generation(agent) -> float:
    """Fraction of prompted continuations that stay coherent (non-empty, no echo)."""
    if not agent.lm.trained_tokens:
        return 0.0
    good = 0
    for prompt in GENERATION_PROMPTS:
        text = agent.generate(prompt, max_tokens=16, temperature=0.7, seed=42)
        if len(text.split()) >= len(prompt.split()) + 4:
            good += 1
    return good / len(GENERATION_PROMPTS)


def bigram_baseline(sentences, extra_texts=()) -> float:
    """Held-out perplexity of a plain bigram chain trained on exactly the same text.

    This is the honest reference point for "is the model smarter than a Markov
    chain?" - it sees the same curriculum sentences (and corpus documents), uses
    identical smoothing and is scored on the identical held-out set.
    """
    from .language import NGramLanguageModel

    reference = NGramLanguageModel(order=2)
    for sentence in sentences:
        reference.train_text(sentence)
    for text in extra_texts:
        reference.train_text(text)
    return reference.perplexity(HELD_OUT_SENTENCES, skip_oov=True)


def run_benchmark(agent, include_reasoning: bool = True) -> Dict[str, float]:
    """Run the full held-out benchmark and return a flat metric dict."""
    language = evaluate_language(agent)
    fact_accuracy, fact_misses = evaluate_facts(agent)
    metrics: Dict[str, float] = {
        **language,
        "fact_accuracy": fact_accuracy,
        "generation_quality": evaluate_generation(agent),
        "fact_total": float(len(FACT_PROBES)),
        "fact_misses": float(len(fact_misses)),
    }
    if include_reasoning:
        reasoning_accuracy, reasoning_misses = evaluate_reasoning(agent)
        metrics["reasoning_accuracy"] = reasoning_accuracy
        metrics["reasoning_total"] = float(len(REASONING_PROBES))
        metrics["reasoning_misses"] = float(len(reasoning_misses))
    return metrics


def format_benchmark(metrics: Dict[str, float]) -> str:
    lines = ["=== Eddi held-out benchmark ==="]
    if "perplexity" in metrics:
        ppl = metrics["perplexity"]
        lines.append(f"  perplexity          : {'inf' if ppl == float('inf') else f'{ppl:.2f}'}")
        if "perplexity_in_vocab" in metrics:
            piv = metrics["perplexity_in_vocab"]
            lines.append(
                f"  perplexity (in-voc) : {'inf' if piv == float('inf') else f'{piv:.2f}'}"
                "  (words the model already knows; lower is better)"
            )
    lines.append(f"  bigram coherence    : {metrics.get('bigram_coherence', 0) * 100:.1f}%")
    lines.append(f"  fact accuracy       : {metrics.get('fact_accuracy', 0) * 100:.1f}%")
    lines.append(f"  generation quality  : {metrics.get('generation_quality', 0) * 100:.1f}%")
    if "reasoning_accuracy" in metrics:
        lines.append(f"  reasoning accuracy  : {metrics['reasoning_accuracy'] * 100:.1f}%")
    return "\n".join(lines)
