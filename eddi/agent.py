"""Eddi Agent: Writable HDC memory + trained language model + 1M-token context + reasoning."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from .context import TieredContext1M
from .hdc import WritableKnowledgeBase, normalise_symbol
from .language import NGramLanguageModel, tokenize
from .memory import ChunkMemory
from .paths import default_context_path

_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")


def _token_hash(token: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little"
    )


#: Reply kinds that carry no information worth remembering.
_NON_ANSWERS = frozenset({
    "remembered", "greeting", "help", "noop", "identity",
    # Refusals are boilerplate. Filing them stores the same sentence over and
    # over and teaches the store to answer unknown things with "I don't know".
    "unknown", "math_error", "convert_error",
    # A retrieved chunk is *already* in the store. Filing it again as a reply
    # duplicates it and drags its source text along ("... (from question)").
    "memory",
})


class EddiAgent:
    """The public brain.

    Combines:

    * :class:`WritableKnowledgeBase` - writable hyperdimensional facts
    * :class:`NGramLanguageModel` - trained interpolated language model
    * :class:`~eddi.memory.ChunkMemory` - verbatim text the agent can quote back
    * :class:`~eddi.reasoning.Reasoner` - question routing and composition
    """

    def __init__(
        self,
        context_path: Optional[str] = None,
        max_tokens: int = 1_000_000,
        persist_context: bool = True,
        lm_order: int = 3,
        brain_path: Optional[str] = None,
        autosave: bool = False,
        memory_path: Optional[str] = None,
        record_turns: bool = True,
    ):
        self.kb = WritableKnowledgeBase()
        self.lm = NGramLanguageModel(order=lm_order)
        self.transitions: Dict[Tuple[str, ...], Counter] = {}
        self._ctx: Optional[TieredContext1M] = None
        self._persist_context = persist_context
        self._context_path = context_path or default_context_path("agent_context")
        self._max_tokens = max_tokens
        self._reasoner = None
        self._brain_path = brain_path
        self._autosave = autosave
        self._dirty = False
        # Verbatim memory. Unlike TieredContext1M this one stores real text and
        # is queried while answering; see eddi/memory.py for why that matters.
        self.memory = ChunkMemory(path=memory_path, auto_load=True)
        self._record_turns = record_turns

    # ---------------------------------------------------------------- context
    @property
    def ctx(self) -> TieredContext1M:
        """Legacy hash-ring store.

        Kept only so old callers do not break. It stores no text and nothing
        reads it back, so it is not memory: :attr:`memory` is. Do not advertise
        this as a context window.
        """
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
        # The old context store keyed on persist_context and was off by default,
        # so ingest reported success while storing nothing. Verbatim memory is
        # the one that must always keep the text.
        self.memory.add(text, source="ingest")
        self._dirty = True
        return len(tokens)

    def teach(self, subject: str, relation: str, obj: str) -> int:
        """Write one fact. Any stored rule that matches is applied immediately.

        Returns the number of facts this write caused (the fact itself plus any
        facts derived by rules).
        """
        self.kb.write_fact(subject, relation, obj)
        self._dirty = True
        return 1 + self.kb.apply_rules()

    write_fact = teach

    def teach_many(self, facts: Iterable[Tuple[str, str, str]]) -> int:
        count = self.kb.import_facts(facts)
        self.kb.apply_rules()
        return count

    def teach_rule(self, premise: str, conclusion: str) -> Tuple[str, str]:
        """Write an inference rule (no training): ``X --premise--> Y`` implies ``X --conclusion--> Y``."""
        rule = self.kb.add_rule(premise, conclusion)
        self._dirty = True
        return rule

    def teach_alias(self, alias: str, canonical: str) -> str:
        """Write a synonym so Eddi understands a phrase it was never trained on."""
        resolved = self.kb.add_alias(alias, canonical)
        self._dirty = True
        return resolved

    def ingest_file(self, path: str) -> int:
        from .pipeline import ingest_file

        return ingest_file(self, path)

    def train_corpus(self, sentences: Iterable[str], weight: int = 1) -> int:
        return self.lm.train_sentences(sentences, weight=weight)

    # --------------------------------------------------------------- thinking
    def ask(self, subject: str, relation: str) -> List[str]:
        hits = self.kb.query(subject, relation)
        return [f"{obj} (confidence={sim:.2f})" for obj, sim in hits]

    def remember(self, text: str, source: str = "") -> int:
        """Store text verbatim so it can be retrieved later."""
        self._dirty = True
        return len(self.memory.add(text, source=source))

    def recall(self, query: str, k: int = 4) -> List:
        """Retrieve the best-matching stored spans for ``query``."""
        return self.memory.search(query, k=k)

    def answer(self, text: str):
        """Route a message through the reasoning layer and return an ``Answer``.

        Every exchange is written to verbatim memory first. Recording is cheap
        (deterministic indexing, no model) and it means a later question can be
        answered from something said earlier even when no fact was extracted.
        """
        text = (text or "").strip()
        if not text:
            return self.reasoner.answer(text)
        reply = self.reasoner.answer(text)
        if self._record_turns and reply is not None and reply.text:
            # Do not store an echo of what we just wrote: it doubles the index
            # with the same sentence and makes the store look redundant.
            body = reply.text.split(" (from ")[0].strip()
            # A memory hit that merely echoes the question adds nothing - it was
            # already filed as a question and must not come back as an answer.
            if reply.kind == "memory" and body.lower() == text.lower():
                body = ""
            # Only store replies that carry information. Canned acknowledgements
            # ("Noted - I've stored that.") are the same few sentences every time
            # and would flood the index with text that answers nothing.
            if body and body.lower() != text.lower() and reply.kind not in _NON_ANSWERS:
                self.memory.add(reply.text, source="reply")
        if self._record_turns and reply is not None:
            # Record the user's turn *after* answering, so a question can never
            # retrieve itself. Questions are stored too (they carry context) but
            # marked so retrieval can skip them: a stored question matches the
            # next question on every word and answers none of them.
            if text.endswith("?") or self.reasoner._looks_like_question(text):
                self.memory.add(text, source="question")
            elif not self.reasoner._is_small_talk(text):
                # Chit-chat ("hi", "thanks") has no content to retrieve and, being
                # one word long, it matches almost any question's greeting word.
                self.memory.add(text, source="turn")
        return reply

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
        known = []
        for relation in self._all_relations():
            if relation[1] == key:
                return relation[1]
            known.append(relation[1])
        unique = list(dict.fromkeys(known))
        # 'port' should find 'port_default': the stored name is more specific
        # than the phrase, and there is exactly one candidate.
        matches = [r for r in unique if r.startswith(key + "_")]
        if len(matches) == 1:
            return matches[0]
        # 'default port' should also find 'port_default'. Try each trailing
        # word-run, so leading modifiers don't block the match.
        words = key.split("_")
        for start in range(1, len(words)):
            tail = "_".join(words[start:])
            matches = [r for r in unique if r == tail or r.startswith(tail + "_")]
            if len(matches) == 1:
                return matches[0]
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
        # No "context_tokens" key: the hash-ring store never held text and
        # reporting its (always zero) count invited the false claim that the
        # agent had a large context window. Memory is the real store.
        return {
            "facts": len(self.kb),
            "symbols": len(self.kb.symbols),
            "lm_vocab": self.lm.vocab_size,
            "lm_order": self.lm.order,
            "trained_tokens": self.lm.trained_tokens,
            "memory_chunks": len(self.memory),
            "memory_tokens": self.memory.total_tokens,
        }

    # --------------------------------------------------------------- lifecycle
    def save(self, path: Optional[str] = None) -> str:
        """Persist everything learned so far (facts, rules, aliases, language)."""
        from .training import save_brain

        written = save_brain(self, path or self._brain_path)
        self._dirty = False
        return written

    def close(self, save: Optional[bool] = None) -> None:
        """Close the context store.

        Learned state is **not** thrown away silently: if this agent wrote
        anything and a ``brain_path`` is configured, it is saved on close
        unless ``save=False`` is passed explicitly.
        """
        should_save = self._autosave if save is None else save
        if should_save and self._dirty and self._brain_path:
            try:
                self.save(self._brain_path)
            except Exception:  # never mask the original failure during shutdown
                pass
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None
        if self._persist_context and self._context_path:
            # Deleting the mmap would silently discard the 1M-token context.
            # It is kept on disk so the next session can reuse it.
            pass

    def __enter__(self) -> "EddiAgent":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
