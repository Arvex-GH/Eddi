"""Eddi's language brain: a trainable interpolated back-off n-gram model.

Design goals
------------
* **Actually learnable** - counts are updated from any text corpus.  No GPU and
  no backpropagation required; training is O(tokens).
* **Smarter than a Markov chain** - two aligned channels are learned at every
  order: exact tokens and *word-type classes* (a cheap stemmer that maps
  ``runs/running/ran`` to one class, plus a number class).  Contexts that were
  never seen as exact token sequences can still be answered through their type
  history, which is how the model generalises beyond memorised n-grams.
* **Calibrated** - every prediction is a probability distribution obtained by
  recursive interpolation of orders 1..N, so the model can be scored with
  perplexity instead of hand-waving.
* **Composable** - entropy sampling with a repetition penalty lets Eddi write
  complete sentences instead of looping on "the the the".

Standard library only, so it ships in the Windows/Linux standalone releases.
"""

from __future__ import annotations

import math
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ABSOLUTE_DISCOUNT = 0.75

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+#']+|[^\sA-Za-z0-9_+#']")
_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)*$")

_SUFFIXES = (
    ("ies", "y"),
    ("sses", "ss"),
    ("ing", ""),
    ("edly", ""),
    ("ed", ""),
    ("ly", ""),
    ("es", ""),
    ("s", ""),
)


def tokenize(text: str) -> List[str]:
    """Split text into word/punctuation tokens (numbers and symbols preserved)."""
    return _TOKEN_RE.findall(text)


def word_type(token: str) -> str:
    """Cheap word-type class: ``running``/``runs`` share the type ``run``.

    Only used for context smoothing, so an approximate stemmer is fine - and
    much faster than training embeddings.
    """
    if token in (BOS, EOS, UNK):
        return token
    if _NUMBER_RE.match(token):
        return "<num>"
    if all(not ch.isalnum() for ch in token):
        return "<punct>"
    lowered = token.lower()
    for suffix, replacement in _SUFFIXES:
        margin = 2 if suffix in ("s", "es") else 3
        if len(lowered) > len(suffix) + margin and lowered.endswith(suffix):
            return lowered[: -len(suffix)] + replacement
    return lowered


class NGramLanguageModel:
    """Interpolated back-off n-gram model over token and word-type channels."""

    def __init__(
        self,
        order: int = 3,
        type_weight: float = 0.2,
        uniform_mix: float = 0.05,
    ):
        self.order = max(1, int(order))
        self.type_weight = float(type_weight)
        # probability mass reserved for unseen words (epsilon-smoothing)
        self.uniform_mix = float(uniform_mix)
        self.vocab: Dict[str, int] = {}
        self._words: List[str] = []
        # counts[n][(kind, ctx...)] = {next_id: count}; n == len(context)
        self.counts: List[Dict[Tuple, Dict[int, int]]] = [{} for _ in range(self.order + 1)]
        self.trained_tokens = 0
        for special in (UNK, BOS, EOS, "<num>", "<punct>"):
            self._token_id(special)

    # ------------------------------------------------------------------ vocab
    def _token_id(self, token: str) -> int:
        tid = self.vocab.get(token)
        if tid is None:
            tid = len(self._words)
            self.vocab[token] = tid
            self._words.append(token)
        return tid

    def _id_to_token(self, tid: int) -> str:
        return self._words[tid] if 0 <= tid < len(self._words) else UNK

    def _ids(self, tokens: Sequence[str], add: bool = False) -> List[int]:
        if add:
            return [self._token_id(t) for t in tokens]
        unk = self.vocab.get(UNK, 0)
        return [self.vocab.get(t, unk) for t in tokens]

    @property
    def vocab_size(self) -> int:
        return len(self._words)

    # --------------------------------------------------------------- training
    def train_text(self, text: str, weight: int = 1, split_sentences: bool = True) -> int:
        """Train on a raw string; returns the number of tokens consumed."""
        if split_sentences:
            total = 0
            for sentence in split_sentences_keep_punct(text):
                total += self.train_text(sentence, weight=weight, split_sentences=False)
            return total
        tokens = tokenize(text)
        if tokens:
            self.train_tokens(tokens, weight=weight)
        return len(tokens)

    def train_tokens(self, tokens: Sequence[str], weight: int = 1) -> None:
        ids = [self._token_id(BOS), *self._ids(tokens, add=True), self._token_id(EOS)]
        typed = [self._token_id(word_type(self._id_to_token(t))) for t in ids]
        for i in range(1, len(ids)):
            target = ids[i]
            max_history = min(self.order - 1, i)
            for kind, channel in (("t", ids), ("y", typed)):
                for n in range(0, max_history + 1):
                    key = (kind, *channel[i - n : i]) if n else (kind,)
                    dist = self.counts[n].get(key)
                    if dist is None:
                        dist = self.counts[n][key] = {}
                    dist[target] = dist.get(target, 0) + weight
        self.trained_tokens += len(ids) - 2

    def train_sentences(self, sentences: Iterable[str], weight: int = 1) -> int:
        seen = 0
        for sentence in sentences:
            seen += self.train_text(sentence, weight=weight)
        return seen

    # -------------------------------------------------------------- prediction
    def _level_distribution(self, ids: List[int], typed: List[int], n: int) -> Dict[int, float]:
        """Smoothed distribution for one order: exact-token and word-type channels."""
        if n == 0:
            token_dist = self.counts[0].get(("t",))
            type_dist = self.counts[0].get(("y",))
        else:
            if len(ids) < n or len(typed) < n:
                return {}
            token_dist = self.counts[n].get(("t", *ids[-n:]))
            type_dist = self.counts[n].get(("y", *typed[-n:]))
        merged: Dict[int, float] = {}
        for dist, weight_scale in ((token_dist, 1.0 - self.type_weight), (type_dist, self.type_weight)):
            if not dist:
                continue
            total = sum(dist.values())
            if total <= 0:
                continue
            for tid, count in dist.items():
                merged[tid] = merged.get(tid, 0.0) + weight_scale * (count / total)
        total = sum(merged.values())
        if total <= 0:
            return {}
        return {tid: p / total for tid, p in merged.items()}

    def distribution(self, context: Sequence[str]) -> Dict[int, float]:
        """Next-token distribution from absolute-discounted interpolated back-off.

        Each learned order (up to ``order - 1`` context tokens) discounts a fixed
        amount from every observed continuation and hands the freed mass to the
        next shorter context.  The interpolation weight depends on the evidence:
        a 4-gram seen once gets a large back-off weight, a 4-gram seen a hundred
        times dominates the estimate.  The final distribution keeps a small
        uniform floor, so unseen words are still possible (never probability 0).
        """
        vocab = [t for t in range(self.vocab_size) if self._id_to_token(t) not in (BOS, EOS)]
        if not vocab:
            return {}
        ids = self._ids(context)
        typed = self._ids([word_type(t) for t in context])
        dist = self._level_distribution(ids, typed, 0)
        if not dist:  # cold start: uniform over the vocabulary
            return {tid: 1.0 / len(vocab) for tid in vocab}

        max_order = min(self.order - 1, len(ids))
        for n in range(1, max_order + 1):
            level = self._level_distribution(ids, typed, n)
            if not level:
                continue
            counts = self._raw_counts(ids, typed, n)
            total = sum(counts.values()) if counts else 0
            if total <= 0:
                continue
            alpha = min(0.9, ABSOLUTE_DISCOUNT * len(counts) / total)
            lower_mass = sum(dist.values()) or 1.0
            merged: Dict[int, float] = {}
            for tid, p in level.items():
                merged[tid] = (1.0 - alpha) * p
            for tid, p in dist.items():
                merged[tid] = merged.get(tid, 0.0) + alpha * (p / lower_mass)
            dist = merged

        # epsilon floor so unknown words still receive probability
        floor = self.uniform_mix / len(vocab)
        out = {tid: (1.0 - self.uniform_mix) * p + floor for tid, p in dist.items()}
        for tid in vocab:
            out.setdefault(tid, floor)
        total = sum(out.values())
        if total <= 0:
            return {tid: 1.0 / len(vocab) for tid in vocab}
        return {tid: p / total for tid, p in out.items()}

    def _raw_counts(self, ids: List[int], typed: List[int], n: int) -> Dict[int, int]:
        return (
            self.counts[n].get(("t", *ids[-n:]))
            or self.counts[n].get(("y", *typed[-n:]))
            or {}
        )

    def predict(self, context: Sequence[str], top_k: int = 5) -> List[Tuple[str, float]]:
        dist = self.distribution(context)
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        out: List[Tuple[str, float]] = []
        for tid, p in ranked:
            token = self._id_to_token(tid)
            if token in (BOS, EOS):
                continue
            out.append((token, p))
            if len(out) >= top_k:
                break
        return out

    def generate(
        self,
        prompt: str,
        max_tokens: int = 24,
        temperature: float = 0.85,
        top_k: int = 24,
        repetition_penalty: float = 1.18,
        seed: Optional[int] = None,
    ) -> str:
        """Continue ``prompt`` with entropy-sampled, repetition-aware decoding."""
        tokens = tokenize(prompt)
        if not tokens or max_tokens <= 0:
            return ""
        rng = random.Random(seed)
        out = list(tokens)
        # Only echo the prompt when the model can actually continue it; otherwise
        # answer from scratch (e.g. "what is speed" must not repeat itself).
        continuation = self.continuation_strength(tokens)
        prefix = list(out) if continuation >= 0.25 else []
        out = [BOS, *prefix]
        skip = {self.vocab[BOS]}
        recent: List[int] = []
        for _ in range(max_tokens):
            dist = self.distribution(out)
            if not dist:
                break
            if temperature > 0:
                scaled = {tid: p ** (1.0 / temperature) for tid, p in dist.items() if p > 0}
            else:
                scaled = dict(dist)
            for tid in recent[-48:]:
                if tid in scaled:
                    scaled[tid] /= repetition_penalty
            ranked = sorted(scaled.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k)]
            weight_total = sum(w for _, w in ranked)
            if weight_total <= 0:
                break
            if temperature <= 0:  # greedy decoding is deterministic
                chosen = ranked[0][0]
            else:
                r = rng.random() * weight_total
                chosen = ranked[-1][0]
                for tid, weight in ranked:
                    r -= weight
                    if r <= 0:
                        chosen = tid
                        break
            token = self._id_to_token(chosen)
            if token == EOS or token in skip:
                continue
            out.append(token)
            recent.append(chosen)
        return detokenize([t for t in out if t != BOS])

    def continuation_strength(self, tokens: Sequence[str], max_order: Optional[int] = None) -> float:
        """How strongly this context predicts its own continuation (0..1).

        Used to decide whether to echo a prompt or generate a fresh answer.
        """
        ids = self._ids(tokens)
        if not ids:
            return 0.0
        max_n = min(self.order, max_order or self.order, len(ids))
        for n in range(max_n, 0, -1):
            if self.counts[n].get(("t", *ids[-n:])):
                return n / self.order
        return 0.0

    # ------------------------------------------------------------------ scoring
    def logprob(self, tokens: Sequence[str]) -> Tuple[float, int]:
        ids = [self.vocab.get(BOS, 0), *self._ids(tokens), self.vocab.get(EOS, 0)]
        total = 0.0
        scored = 0
        for i in range(1, len(ids)):
            context = [self._id_to_token(t) for t in ids[max(0, i - self.order) : i]]
            dist = self.distribution(context)
            total += math.log(max(dist.get(ids[i], 1e-9), 1e-9))
            scored += 1
        return total, scored

    def perplexity(self, sentences: Iterable[str], skip_oov: bool = False) -> float:
        """Held-out perplexity. Lower is better; random guessing is ~vocab size.

        With ``skip_oov=True`` words outside the vocabulary are not scored, which
        makes the number comparable across corpora of different vocabulary sizes.
        Both numbers are reported by the benchmark for transparency.
        """
        total, scored = 0.0, 0
        for sentence in sentences:
            tokens = tokenize(sentence)
            if len(tokens) < 2:
                continue
            if skip_oov:
                tokens = [t for t in tokens if t in self.vocab]
                if len(tokens) < 2:
                    continue
            lp, n = self.logprob(tokens)
            total += lp
            scored += n
        if scored == 0:
            return float("inf")
        return math.exp(-total / scored)

    def coherence(self, sentences: Iterable[str]) -> float:
        """Fraction of generated bigrams seen during training (0..1)."""
        known = total = 0
        for sentence in sentences:
            tokens = [BOS, *tokenize(sentence), EOS]
            for i in range(1, len(tokens)):
                total += 1
                key = ("t", self.vocab.get(tokens[i - 1], self.vocab[UNK]))
                if self.counts[1].get(key):
                    known += 1
        return known / total if total else 0.0

    # ------------------------------------------------------------- persistence
    def to_payload(self) -> dict:
        return {
            "order": self.order,
            "type_weight": self.type_weight,
            "uniform_mix": self.uniform_mix,
            "words": list(self._words),
            "counts": [
                [[list(key), [[tid, count] for tid, count in dist.items()]] for key, dist in table.items()]
                for table in self.counts
            ],
            "trained_tokens": self.trained_tokens,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "NGramLanguageModel":
        model = cls(
            order=payload.get("order", 4),
            type_weight=payload.get("type_weight", 0.35),
            uniform_mix=payload.get("uniform_mix", 0.05),
        )
        model._words = list(payload.get("words", []))
        model.vocab = {w: i for i, w in enumerate(model._words)}
        model.counts = [{} for _ in range(model.order + 1)]
        for n, table in enumerate(payload.get("counts", [])):
            if n > model.order:
                break
            for key, dist in table:
                model.counts[n][tuple(key)] = {int(t): int(c) for t, c in dist}
        model.trained_tokens = payload.get("trained_tokens", 0)
        for special in (UNK, BOS, EOS, "<num>", "<punct>"):
            model._token_id(special)
        return model

    def prune(self, min_count: int = 1) -> int:
        """Drop rare n-gram contexts to keep saved brains small. Returns removed keys."""
        removed = 0
        for n in range(2, self.order + 1):
            table = self.counts[n]
            for key in list(table):
                if sum(table[key].values()) < min_count:
                    del table[key]
                    removed += 1
        return removed


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences_keep_punct(text: str) -> List[str]:
    """Split a document into sentence-ish units without losing punctuation."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def detokenize(tokens: Sequence[str]) -> str:
    """Join tokens back into readable text with sensible spacing/punctuation."""
    out = ""
    no_space_before = {",", ".", "!", "?", ";", ":", ")", "]", "}", "%", "’"}
    no_space_after = {"(", "[", "{", "$", "#", "@"}
    for i, token in enumerate(tokens):
        if not out:
            out = token
        elif token in no_space_before:
            out += token
        elif i > 0 and tokens[i - 1] in no_space_after:
            out += token
        elif token.startswith("'") and len(token) > 1:
            out += token
        else:
            out += " " + token
    return out
