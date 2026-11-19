"""Writable Hyperdimensional Computing (HDC) / Vector Symbolic Architecture (VSA) Core.

Unlike neural networks whose weights must be learned via backpropagation, HDC weights
are algebraic hypervectors that can be directly written, bound, bundled, and queried
in O(1) per concept.
"""

import hashlib
import math
from typing import Dict, List, Tuple


class HyperVector:
    """1024-bit packed binary hypervector represented as a Python arbitrary-precision int."""

    BITS = 1024
    MASK = (1 << BITS) - 1

    def __init__(self, value: int):
        self.value = value & self.MASK

    @classmethod
    def from_symbol(cls, symbol: str) -> "HyperVector":
        """Deterministically project any symbol string into a quasi-orthogonal hypervector."""
        chunks = []
        for i in range(4):  # 4 * 256 bits = 1024 bits
            h = hashlib.sha256(f"{i}:{symbol}".encode("utf-8")).digest()
            chunks.append(int.from_bytes(h, "big"))
        val = (chunks[0] << 768) | (chunks[1] << 512) | (chunks[2] << 256) | chunks[3]
        return cls(val)

    def bind(self, other: "HyperVector") -> "HyperVector":
        """Bind two hypervectors (XOR). Self-inverse: bind(A, B).bind(B) == A."""
        return HyperVector(self.value ^ other.value)

    def similarity(self, other: "HyperVector") -> float:
        """Normalized Hamming similarity in [0.0, 1.0]. Random vectors ~ 0.5."""
        diff = (self.value ^ other.value).bit_count()
        return 1.0 - (diff / self.BITS)


class WritableKnowledgeBase:
    """Directly writable symbolic-neural memory using HDC binding."""

    def __init__(self) -> None:
        self.symbols: Dict[str, HyperVector] = {}
        self.facts: List[Tuple[str, str, str, HyperVector]] = []

    def get_or_create(self, name: str) -> HyperVector:
        if name not in self.symbols:
            self.symbols[name] = HyperVector.from_symbol(name)
        return self.symbols[name]

    def write_fact(self, subject: str, relation: str, obj: str) -> HyperVector:
        """Write a structured fact directly into hyperdimensional memory without training."""
        s_hv = self.get_or_create(subject)
        r_hv = self.get_or_create(relation)
        o_hv = self.get_or_create(obj)
        bound = s_hv.bind(r_hv).bind(o_hv)
        self.facts.append((subject, relation, obj, bound))
        return bound

    def query(self, subject: str, relation: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Unbind (subject, relation) and find the closest matching object in memory."""
        s_hv = self.get_or_create(subject)
        r_hv = self.get_or_create(relation)
        probe = s_hv.bind(r_hv)

        scores: List[Tuple[str, float]] = []
        for s, r, o, fact_hv in self.facts:
            candidate_obj_hv = fact_hv.bind(probe)
            sim = candidate_obj_hv.similarity(self.get_or_create(o))
            if sim > 0.65:
                scores.append((o, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
