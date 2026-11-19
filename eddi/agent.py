"""Built-in Eddi Agent: combines WritableKnowledgeBase + TieredContext1M + Writable Sequence Memory."""

import hashlib
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from .context import TieredContext1M
from .hdc import WritableKnowledgeBase


class EddiAgent:
    def __init__(self, context_path: str = "/tmp/eddi_agent_context.mmap"):
        self.kb = WritableKnowledgeBase()
        self.ctx = TieredContext1M(path=context_path, max_tokens=1_000_000)
        self.transitions: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

    def _tok_hash(self, token: str) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")

    def ingest_text(self, text: str) -> int:
        tokens = text.split()
        for i, tok in enumerate(tokens):
            h = self._tok_hash(tok)
            sig = h ^ (h >> 17)
            self.ctx.append(h, sig)
            if i >= 1:
                self.transitions[(tokens[i - 1],)][tok] += 1
            if i >= 2:
                self.transitions[(tokens[i - 2], tokens[i - 1])][tok] += 1
        return len(tokens)

    def generate(self, prompt: str, max_tokens: int = 25) -> str:
        tokens = prompt.split()
        if not tokens:
            return ""
        out = list(tokens)
        for _ in range(max_tokens):
            ctx2 = tuple(out[-2:]) if len(out) >= 2 else None
            ctx1 = (out[-1],)
            candidates = self.transitions.get(ctx2) if ctx2 and ctx2 in self.transitions else self.transitions.get(ctx1)
            if not candidates:
                break
            next_tok = candidates.most_common(1)[0][0]
            out.append(next_tok)
        return " ".join(out)

    def teach(self, subject: str, relation: str, obj: str) -> None:
        self.kb.write_fact(subject, relation, obj)

    def ask(self, subject: str, relation: str) -> List[str]:
        hits = self.kb.query(subject, relation)
        return [f"{obj} (confidence={sim:.2f})" for obj, sim in hits]

    def close(self) -> None:
        self.ctx.close()
