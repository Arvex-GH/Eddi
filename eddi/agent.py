"""Eddi Agent: Writable HDC memory + trained language model + 1M-token context + reasoning."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from .context import TieredContext1M
from .hdc import WritableKnowledgeBase, normalise_symbol
from .language import NGramLanguageModel, tokenize
from .paths import default_context_path

_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")


def _token_hash(token: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little"
    )


class EddiAgent:
    """The public brain.

    Combines:

    * :class:`WritableKnowledgeBase` - writable hyperdimensional facts
    * :class:`NGramLanguageModel` - trained interpolated language model
    * :class:`TieredContext1M` - 1M-token disk-backed context (L1 RAM + L2 mmap)
    * :class:`~eddi.reasoning.Reasoner` - question routing and composition
    """

    def __init__(
        self,
        context_path: Optional[str] = None,
        max_tokens: int = 1_000_000,
        persist_context: bool = True,
        lm_order: int = 3,
    ):
        self.kb = WritableKnowledgeBase()
        self.lm = NGramLanguageModel(order=lm_order)
        self.transitions: Dict[Tuple[str, ...], Counter] = {}
        self._ctx: Optional[TieredContext1M] = None
        self._persist_context = persist_context
        self._context_path = context_path or default_context_path("agent_context")
        self._max_tokens = max_tokens
        self._reasoner = None

    # ---------------------------------------------------------------- context
    @property
    def ctx(self) -> TieredContext1M:
        if self._ctx is None:
            self._ctx = TieredContext1M(path=self._context_path, max_tokens=self._max_tokens)
        return self._ctx

    @property
    def reasoner(self):
        if self._reasoner is None:
            from .reasoning import Reasoner

            self._reasoner = Reasoner(self)
        return self._reasoner

    # ---------------------------------------------------------------- learning
    def ingest_text(self, text: str) -> int:
        """Feed a document to the context store and the language model."""
        tokens = tokenize(text)
        if not tokens:
            return 0
        if self._persist_context:
            ctx = self.ctx
            for tok in tokens:
                h = _token_hash(tok)
                ctx.append(h, h ^ (h >> 17))
            ctx.flush()
        for i in range(1, len(tokens)):
            self.transitions.setdefault((tokens[i - 1],), Counter())[tokens[i]] += 1
        self.lm.train_text(text)
        return len(tokens)

    def teach(self, subject: str, relation: str, obj: str) -> int:
        """Write one fact. Any stored rule that matches is applied immediately.

        Returns the number of facts this write caused (the fact itself plus any
        facts derived by rules).
        """
        self.kb.write_fact(subject, relation, obj)
        return 1 + self.kb.apply_rules()

    write_fact = teach

    def teach_many(self, facts: Iterable[Tuple[str, str, str]]) -> int:
        count = self.kb.import_facts(facts)
        self.kb.apply_rules()
        return count

    def teach_rule(self, premise: str, conclusion: str) -> Tuple[str, str]:
        """Write an inference rule (no training): ``X --premise--> Y`` implies ``X --conclusion--> Y``."""
        return self.kb.add_rule(premise, conclusion)

    def teach_alias(self, alias: str, canonical: str) -> str:
        """Write a synonym so Eddi understands a phrase it was never trained on."""
        return self.kb.add_alias(alias, canonical)

    def ingest_file(self, path: str) -> int:
        from .pipeline import ingest_file

        return ingest_file(self, path)

    def train_corpus(self, sentences: Iterable[str], weight: int = 1) -> int:
        return self.lm.train_sentences(sentences, weight=weight)

    # --------------------------------------------------------------- thinking
    def ask(self, subject: str, relation: str) -> List[str]:
        hits = self.kb.query(subject, relation)
        return [f"{obj} (confidence={sim:.2f})" for obj, sim in hits]

    def answer(self, text: str):
        """Route a message through the reasoning layer and return an ``Answer``."""
        return self.reasoner.answer(text)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 25,
        temperature: float = 0.85,
        seed: Optional[int] = None,
    ) -> str:
        """Continue a prompt with the trained language model (no model yet -> n-gram)."""
        if self.lm.trained_tokens:
            return self.lm.generate(
                prompt, max_tokens=max_tokens, temperature=temperature, seed=seed
            )
        return self._legacy_generate(prompt, max_tokens)

    def _legacy_generate(self, prompt: str, max_tokens: int = 25) -> str:
        """Bigram fallback for agents that were never trained (kept for compatibility)."""
        tokens = tokenize(prompt)
        if not tokens:
            return ""
        out = list(tokens)
        for _ in range(max_tokens):
            ctx2 = tuple(out[-2:]) if len(out) >= 2 else None
            ctx1 = (out[-1],)
            candidates = self.transitions.get(ctx2) if ctx2 and ctx2 in self.transitions else self.transitions.get(ctx1)
            if not candidates:
                break
            out.append(candidates.most_common(1)[0][0])
        return " ".join(out)

    # -------------------------------------------------------------- entities
    def resolve_entity(self, text: str) -> Optional[str]:
        """Map a phrase to a known subject/object (exact, normalised, fuzzy)."""
        if not text:
            return None
        text = _ZERO_WIDTH_RE.sub("", text).strip()
        kb = self.kb
        if text in kb.symbols:
            return text
        resolved = kb.resolve(text)
        if resolved is not None:
            return resolved
        # try each token run against known symbols
        words = text.split()
        for size in range(len(words), 0, -1):
            for start in range(0, len(words) - size + 1):
                candidate = " ".join(words[start : start + size])
                resolved = kb.resolve(candidate)
                if resolved is not None:
                    return resolved
        return None

    def resolve_relation(self, text: str) -> Optional[str]:
        """Map a phrase like 'the speed of light' to a stored relation key."""
        if not text:
            return None
        relation = self.resolve_entity(text)
        if relation is not None:
            known = set(kb_relation for _, kb_relation in self._all_relations())
            if relation in known:
                return relation
        key = normalise_symbol(text)
        for relation in self._all_relations():
            if relation[1] == key:
                return relation[1]
        return None

    def _all_relations(self):
        seen = set()
        for subject, relation, _obj, _hv in self.kb.facts:
            if (subject, relation) not in seen:
                seen.add((subject, relation))
                yield (subject, relation)

    # --------------------------------------------------------------- skills
    def relate(self, a: str, b: str) -> List[str]:
        """Find relations that connect two entities in either direction."""
        ea = self.resolve_entity(a) or a
        eb = self.resolve_entity(b) or b
        out: List[str] = []
        for relation, obj, _conf in self.kb.facts_about(ea):
            target = self.resolve_entity(obj) or obj
            if target == eb:
                out.append(f"{ea} --{relation}--> {eb}")
        for relation, obj, _conf in self.kb.facts_about(eb):
            target = self.resolve_entity(obj) or obj
            if target == ea:
                out.append(f"{eb} --{relation}--> {ea}")
        return out

    def chain(self, start: str, relations: List[str]) -> List[str]:
        resolved_start = self.resolve_entity(start) or start
        resolved_relations = [self.resolve_relation(r) or r for r in relations]
        return self.kb.chain(resolved_start, resolved_relations)

    def stats(self) -> Dict[str, int]:
        ctx_count = self._ctx.count if self._ctx is not None else 0
        return {
            "facts": len(self.kb),
            "symbols": len(self.kb.symbols),
            "lm_vocab": self.lm.vocab_size,
            "lm_order": self.lm.order,
            "trained_tokens": self.lm.trained_tokens,
            "context_tokens": ctx_count,
        }

    # --------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None
        if self._persist_context and self._context_path:
            try:
                import os

                if os.path.exists(self._context_path):
                    os.remove(self._context_path)
            except OSError:
                pass

    def __enter__(self) -> "EddiAgent":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
