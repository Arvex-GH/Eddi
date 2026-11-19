"""Training pipeline: turn Eddi from an empty shell into a trained agent.

``train`` is the one-call entry point used by the CLI, the desktop app and the
tests.  It performs real, measurable learning:

1. **Facts** are written into the hyperdimensional knowledge base.
2. **Prose** is streamed through the 1M-token disk-backed context and into the
   interpolated n-gram language model, repeated over several *epochs*.
3. **Held-out evaluation** measures perplexity, bigram coherence, factual
   accuracy and reasoning accuracy before and after training, so the "smart"
   claim is verifiable instead of anecdotal.
4. **Persistence** saves a portable brain (facts + model) to a pickle file.

The same code trains from the built-in curriculum or from any directory of text
files (``corpus_dirs``), e.g. a local copy of Wikipedia or arXiv metadata.
"""

from __future__ import annotations

import glob
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from . import curriculum, evaluation
from .agent import EddiAgent
from .paths import default_brain_path, default_memory_path

from .language import tokenize

BRAIN_FORMAT = "eddi-brain-v2"


@dataclass
class TrainingConfig:
    packs: Sequence[str] = curriculum.ALL_PACKS
    corpus_dirs: Sequence[str] = ()
    corpus_glob: str = "*.txt"
    epochs: int = 3
    order: int = 3
    weight: int = 1
    evaluate: bool = True
    prune_min_count: int = 1
    save_path: Optional[str] = None
    max_corpus_tokens: int = 2_000_000
    seed: Optional[int] = 1234
    #: Train on the prose in eddi/corpus_prose as well as the pack prose. That
    #: corpus exists to make the language model fluent; it is off for the small
    #: builds (tests, the capability suite) that only need facts.
    use_prose_corpus: bool = True


@dataclass
class TrainingReport:
    epochs: int
    facts: int
    sentences: int
    documents: int
    corpus_tokens: int
    lm_vocab: int
    lm_order: int
    train_seconds: float
    perplexity_before: Optional[float] = None
    perplexity_after: Optional[float] = None
    perplexity_in_vocab: Optional[float] = None
    perplexity_bigram_baseline: Optional[float] = None
    #: Share of prompted continuations that stay on the trained sentence. This
    #: is the metric that tracks usable generation; perplexity does not.
    completion_accuracy: Optional[float] = None
    coherence_after: Optional[float] = None
    fact_accuracy: Optional[float] = None
    reasoning_accuracy: Optional[float] = None
    saved_to: Optional[str] = None
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def pretty(self) -> str:
        lines = [
            "=== Eddi training report ===",
            f"  epochs            : {self.epochs}",
            f"  facts written     : {self.facts}",
            f"  sentences trained : {self.sentences}",
            f"  extra documents   : {self.documents}",
            f"  tokens trained    : {self.corpus_tokens}",
            f"  vocabulary        : {self.lm_vocab}",
            f"  model order       : {self.lm_order}",
            f"  training time     : {self.train_seconds:.2f}s",
        ]
        if self.perplexity_before is not None:
            lines.append(f"  perplexity before : {self.perplexity_before:.2f}")
        if self.perplexity_after is not None:
            lines.append(f"  perplexity after  : {self.perplexity_after:.2f}  (lower is better)")
        if self.perplexity_in_vocab is not None:
            lines.append(f"  perplexity in-voc : {self.perplexity_in_vocab:.2f}  (known words only)")
        if self.completion_accuracy is not None:
            lines.append(
                f"  completion        : {self.completion_accuracy * 100:.0f}%  "
                "(prompted continuations that stay on the sentence - this is the one to watch)"
            )
        if self.perplexity_bigram_baseline is not None and self.perplexity_after:
            gain = self.perplexity_bigram_baseline / self.perplexity_after
            if gain >= 1.0:
                verdict = f"this model is {gain:.2f}x better than a bigram chain"
            else:
                verdict = (
                    f"bigram is {1 / gain:.2f}x better on this tiny corpus - "
                    "add more training text for this model to win"
                )
            lines.append(f"  bigram baseline   : {self.perplexity_bigram_baseline:.2f}  -> {verdict}")
        if self.coherence_after is not None:
            lines.append(f"  bigram coherence  : {self.coherence_after * 100:.1f}%")
        if self.fact_accuracy is not None:
            lines.append(f"  fact accuracy     : {self.fact_accuracy * 100:.1f}%")
        if self.reasoning_accuracy is not None:
            lines.append(f"  reasoning accuracy: {self.reasoning_accuracy * 100:.1f}%")
        if self.saved_to:
            lines.append(f"  brain saved to    : {self.saved_to}")
        return "\n".join(lines)


def suggest_order(corpus_tokens: int) -> int:
    """Pick an n-gram order that the corpus can actually support.

    A deeper model is strictly better *when there is enough text to fill it*:
    with a 4.8k-token curriculum, order 3 blended sentences together and
    answered "water boils at one hundred eighty degrees" by splicing a
    geometry sentence onto a chemistry one, while order 5 reproduced both
    correctly. But a deep order on a tiny corpus only adds contexts that are
    seen once, so the thresholds below rise with the order.

    These numbers are a floor, not a target: the README's advice to raise the
    order on large corpora still applies.
    """
    if corpus_tokens >= 200_000:
        return 6
    if corpus_tokens >= 20_000:
        return 5
    if corpus_tokens >= 2_000:
        return 4
    return 3


def _seed_untrained(agent: EddiAgent) -> None:
    """Give the 'before' evaluation a fair baseline: the facts but no language."""
    for pack in curriculum.ALL_PACKS:
        for s, r, o in curriculum.pack_facts(pack):
            agent.teach(s, r, o)


def _iter_corpus_files(corpus_dirs: Iterable[str], pattern: str) -> Iterable[str]:
    for directory in corpus_dirs:
        for path in sorted(glob.glob(os.path.join(directory, "**", pattern), recursive=True)):
            yield path


def train(
    agent: Optional[EddiAgent] = None,
    config: Optional[TrainingConfig] = None,
    verbose: bool = False,
    log=None,
) -> TrainingReport:
    """Train (or continue training) an agent and return a measurable report."""
    cfg = config or TrainingConfig()
    agent = agent or EddiAgent(lm_order=cfg.order, persist_context=False)
    emit = log or (print if verbose else (lambda *_: None))

    if cfg.evaluate:
        emit("· baseline evaluation (facts only, untrained language model)")
        before = evaluation.run_benchmark(agent, include_reasoning=False)

    t0 = time.time()
    epoch_sentences: List[str] = []
    for pack in cfg.packs:
        for s, r, o in curriculum.pack_facts(pack):
            agent.teach(s, r, o)
        epoch_sentences.extend(curriculum.pack_prose(pack))
    if cfg.use_prose_corpus:
        from .corpus_prose import all_sentences

        prose = all_sentences()
        if prose:
            epoch_sentences.extend(prose)
            emit(f"· prose corpus: {len(prose)} sentences")

    # Read each corpus file once, then replay it every epoch (streaming, not stored twice).
    corpus_docs: List[str] = []
    for path in _iter_corpus_files(cfg.corpus_dirs, cfg.corpus_glob):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if text.strip():
            corpus_docs.append(text)

    documents = len(corpus_docs)
    tokens = 0
    budget_hit = False
    for epoch in range(max(1, cfg.epochs)):
        emit(f"· epoch {epoch + 1}/{cfg.epochs}: {len(epoch_sentences)} curriculum sentences, "
             f"{documents} corpus documents")
        for sentence in epoch_sentences:
            # ingest_text trains the language model and stores the text, but it
            # does not record the sentence for generation scoring; train_corpus
            # does both, so use it and count the tokens separately.
            agent.train_corpus([sentence])
            tokens += len(tokenize(sentence))
        for text in corpus_docs:
            if tokens >= cfg.max_corpus_tokens:
                budget_hit = True
                break
            chunk = text
            if len(chunk) > 4_000_000:  # keep a single ingest bounded for huge files
                chunk = chunk[:4_000_000]
            tokens += agent.ingest_text(chunk)
        if budget_hit:
            emit(f"· corpus token budget reached ({cfg.max_corpus_tokens})")
            break

    if cfg.prune_min_count > 1:
        removed = agent.lm.prune(cfg.prune_min_count)
        emit(f"· pruned {removed} rare n-gram contexts")

    train_seconds = time.time() - t0
    report = TrainingReport(
        epochs=max(1, cfg.epochs),
        facts=len(agent.kb),
        sentences=len(epoch_sentences) * max(1, cfg.epochs),
        documents=documents,
        corpus_tokens=tokens,
        lm_vocab=agent.lm.vocab_size,
        lm_order=agent.lm.order,
        train_seconds=train_seconds,
    )

    if cfg.evaluate:
        emit("· held-out evaluation")
        after = evaluation.run_benchmark(agent)
        report.perplexity_after = after["perplexity"]
        report.perplexity_before = before["perplexity"]
        report.perplexity_in_vocab = after.get("perplexity_in_vocab")
        report.perplexity_bigram_baseline = evaluation.bigram_baseline(
            epoch_sentences, corpus_docs
        )
        report.coherence_after = after["bigram_coherence"]
        report.completion_accuracy = after.get("completion_accuracy")
        report.fact_accuracy = after["fact_accuracy"]
        report.reasoning_accuracy = after["reasoning_accuracy"]
        report.details = {
            "generation_quality": after["generation_quality"],
            "reasoning_total": after["reasoning_total"],
        }

    if cfg.save_path:
        save_brain(agent, cfg.save_path)
        report.saved_to = cfg.save_path
        emit(f"· brain saved to {cfg.save_path}")

    return report


# --------------------------------------------------------------- persistence
def save_brain(agent: EddiAgent, path: Optional[str] = None) -> str:
    """Persist facts and language model into one portable brain file."""
    path = path or default_brain_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "format": BRAIN_FORMAT,
        "facts": [(s, r, o) for s, r, o, _hv in agent.kb.facts],
        "language_model": agent.lm.to_payload(),
        "transitions": dict(agent.transitions),
        # writable intelligence that is NOT obtained by training
        "rules": list(agent.kb.rules),
        "aliases": dict(agent.kb.user_aliases),
        "derived": sorted(agent.kb.derived),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_brain(
    path: Optional[str] = None,
    agent: Optional[EddiAgent] = None,
    context_path: Optional[str] = None,
    persist_context: bool = False,
) -> EddiAgent:
    """Load a saved brain; returns a ready-to-answer agent."""
    path = path or default_brain_path()
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "format" not in payload:
        raise ValueError(f"Not an Eddi brain file: {path}")
    if payload["format"] != BRAIN_FORMAT:
        raise ValueError(f"Unsupported brain format {payload['format']!r} (expected {BRAIN_FORMAT})")

    agent = agent or EddiAgent(
        context_path=context_path,
        persist_context=persist_context,
        brain_path=path,
        memory_path=default_memory_path(),
    )
    direct = [f for f in payload.get("facts", []) if tuple(f) not in set(
        tuple(d) for d in payload.get("derived", [])
    )]
    agent.kb.import_facts(direct)
    for alias, canonical in payload.get("aliases", {}).items():
        agent.kb.user_aliases[alias] = canonical
    for premise, conclusion in payload.get("rules", []):
        agent.kb.add_rule(premise, conclusion)
    agent.kb.derived = {tuple(d) for d in payload.get("derived", [])}
    agent.kb.apply_rules()
    from .language import NGramLanguageModel

    agent.lm = NGramLanguageModel.from_payload(payload["language_model"])
    agent.transitions = payload.get("transitions", {})
    return agent


def brain_exists(path: Optional[str] = None) -> bool:
    return os.path.exists(path or default_brain_path())
