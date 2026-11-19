"""Writable Hyperdimensional Computing (HDC) / Vector Symbolic Architecture (VSA) core.

Unlike neural networks whose weights must be learned via gradient descent, HDC
memories are algebraic hypervectors that can be *written* directly, bound,
bundled and cleaned-up in O(1) per concept.

What is stored for every fact ``(subject, relation, object)``:

* ``bound``    – ``S xor R xor O`` (1024-bit packed binary hypervector)
* ``_by_sr``   – a writable hash index ``(subject, relation) -> [objects]``
                 giving O(1) exact lookup instead of an O(N) memory scan
* ``_superpos`` – a lazily built *majority-vote bundle* of all bound vectors for
                 that key.  Unbinding the probe ``S xor R`` from the bundle gives
                 a denoised superposition of the object vectors, which is then
                 cleaned up against the symbol memory ("HDC cleanup memory").

The superposition is what makes the memory robust: even if a probe or a stored
fact is partially corrupted, the majority vote still recovers the right object.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Python 3.10+
    _popcount = int.bit_count  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Python 3.9 fallback

    def _popcount(x: int) -> int:  # type: ignore[misc]
        return bin(x).count("1")


_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_NORMALISE_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def normalise_symbol(name: str) -> str:
    """Normalise a human phrase into a stable KB key.

    ``"the Speed of Light?"`` -> ``"speed_of_light"``
    """
    text = _ARTICLE_RE.sub("", name.strip())
    text = _NORMALISE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.replace(" ", "_")


class HyperVector:
    """1024-bit packed binary hypervector represented as a Python integer."""

    BITS = 1024
    MASK = (1 << BITS) - 1
    BYTES = BITS // 8

    def __init__(self, value: int):
        self.value = value & self.MASK

    # ------------------------------------------------------------------ creation
    @classmethod
    def from_symbol(cls, symbol: str) -> "HyperVector":
        """Deterministically project any symbol string into a quasi-orthogonal vector."""
        chunks = []
        for i in range(4):  # 4 * 256 bits = 1024 bits
            h = hashlib.sha256(f"{i}:{symbol}".encode("utf-8")).digest()
            chunks.append(int.from_bytes(h, "big"))
        val = (chunks[0] << 768) | (chunks[1] << 512) | (chunks[2] << 256) | chunks[3]
        return cls(val)

    @classmethod
    def from_bytes(cls, data: bytes) -> "HyperVector":
        return cls(int.from_bytes(data, "little"))

    @classmethod
    def bundle(cls, vectors: Sequence["HyperVector"]) -> "HyperVector":
        """Majority-vote superposition of two or more hypervectors."""
        vectors = list(vectors)
        if not vectors:
            return cls(0)
        if len(vectors) == 1:
            return cls(vectors[0].value)
        counts = [0] * cls.BITS
        for hv in vectors:
            v = hv.value
            while v:
                lsb = v & -v
                counts[lsb.bit_length() - 1] += 1
                v ^= lsb
        threshold = len(vectors) / 2.0
        out = 0
        for bit, c in enumerate(counts):
            if c > threshold:
                out |= 1 << bit
        return cls(out)

    # ---------------------------------------------------------------- algebra
    def bind(self, other: "HyperVector") -> "HyperVector":
        """Bind two hypervectors (XOR). Self-inverse: ``bind(a, b).bind(b) == a``."""
        return HyperVector(self.value ^ other.value)

    def unbind(self, other: "HyperVector") -> "HyperVector":
        """Unbind is the same XOR operation (kept as an explicit, readable intent)."""
        return self.bind(other)

    def permute(self, shift: int = 1) -> "HyperVector":
        """Cyclic bit rotation - encodes order/position without extra vectors."""
        shift %= self.BITS
        v = self.value
        return HyperVector(((v << shift) | (v >> (self.BITS - shift))) & self.MASK)

    def similarity(self, other: "HyperVector") -> float:
        """Normalised Hamming similarity in ``[0.0, 1.0]`` (random pairs ~0.5)."""
        diff = _popcount(self.value ^ other.value)
        return 1.0 - (diff / self.BITS)

    def distance(self, other: "HyperVector") -> int:
        """Hamming distance in bits."""
        return _popcount(self.value ^ other.value)

    def to_bytes(self) -> bytes:
        return self.value.to_bytes(self.BYTES, "little")

    # ---------------------------------------------------------------- dunders
    def __xor__(self, other: "HyperVector") -> "HyperVector":
        return self.bind(other)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HyperVector) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"HyperVector(0x{self.value:0{self.BYTES * 2}x})"


class WritableKnowledgeBase:
    """Directly writable symbolic memory with O(1) retrieval and HDC cleanup."""

    def __init__(self) -> None:
        self.symbols: Dict[str, HyperVector] = {}
        self.facts: List[Tuple[str, str, str, HyperVector]] = []
        self._by_sr: Dict[Tuple[str, str], List[str]] = {}
        self._by_s: Dict[str, List[Tuple[str, str]]] = {}
        self._by_o: Dict[str, List[Tuple[str, str]]] = {}
        self._object_hvs: Dict[Tuple[str, str], List[HyperVector]] = {}
        self._superpos: Dict[Tuple[str, str], HyperVector] = {}
        self._alias: Dict[str, str] = {}

    # ------------------------------------------------------------------ symbols
    def get_or_create(self, name: str) -> HyperVector:
        if name not in self.symbols:
            self.symbols[name] = HyperVector.from_symbol(name)
            self._alias.setdefault(normalise_symbol(name), name)
        return self.symbols[name]

    def resolve(self, name: str) -> Optional[str]:
        """Resolve a human phrase to a stored symbol (exact, normalised, fuzzy)."""
        if name in self.symbols:
            return name
        key = normalise_symbol(name)
        if key in self._alias:
            return self._alias[key]
        lowered = {s.lower(): s for s in self.symbols}
        if key in lowered:
            return lowered[key]
        if not self.symbols:
            return None
        match = difflib.get_close_matches(key, list(self._alias), n=1, cutoff=0.82)
        if match:
            return self._alias[match[0]]
        return None

    # ------------------------------------------------------------------ writing
    def write_fact(self, subject: str, relation: str, obj: str) -> HyperVector:
        """Write a structured fact directly into hyperdimensional memory (no training)."""
        s_hv = self.get_or_create(subject)
        r_hv = self.get_or_create(relation)
        o_hv = self.get_or_create(obj)
        bound = s_hv.bind(r_hv).bind(o_hv)

        key = (subject, relation)
        objects = self._by_sr.setdefault(key, [])
        if obj in objects:  # refresh recency instead of duplicating
            objects.remove(obj)
            objects.append(obj)
            self._object_hvs[key].remove(o_hv)
            self._object_hvs[key].append(o_hv)
        else:
            objects.append(obj)
            self._object_hvs.setdefault(key, []).append(o_hv)
            self._by_s.setdefault(subject, []).append((relation, obj))
            self._by_o.setdefault(obj, []).append((subject, relation))
            self.facts.append((subject, relation, obj, bound))
        self._superpos.pop(key, None)  # invalidate cached bundle
        return bound

    def import_facts(self, facts: Iterable[Tuple[str, str, str]]) -> int:
        count = 0
        for s, r, o in facts:
            self.write_fact(s, r, o)
            count += 1
        return count

    # ------------------------------------------------------------------ reading
    def query(self, subject: str, relation: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Exact O(1) lookup of ``(subject, relation)`` objects, most recent first."""
        resolved = self.resolve(subject) or subject
        resolved_rel = self.resolve(relation) or relation
        objects = self._by_sr.get((resolved, resolved_rel), [])
        return [(o, 1.0) for o in reversed(objects[-top_k:])]

    def probe(self, subject: str, relation: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """HDC noise-robust query: unbind a superposition and clean up against memory.

        Works even when the query arrives with corrupted/noisy bits, because the
        stored bundle is a majority vote over every fact written for the key.
        """
        resolved = self.resolve(subject) or subject
        resolved_rel = self.resolve(relation) or relation
        key = (resolved, resolved_rel)
        objects = self._by_sr.get(key)
        if not objects:
            return []
        if len(objects) == 1:
            return [(objects[0], 1.0)]

        bundle = self._superpos.get(key)
        if bundle is None:
            bound_vectors = [
                self.symbols[resolved].bind(self.symbols[resolved_rel]).bind(o_hv)
                for o_hv in self._object_hvs[key]
            ]
            bundle = HyperVector.bundle(bound_vectors)
            self._superpos[key] = bundle

        recovered = bundle.bind(self.symbols[resolved].bind(self.symbols[resolved_rel]))
        scores = [(o, recovered.similarity(self.symbols[o])) for o in objects]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def facts_about(self, subject: str, limit: Optional[int] = None) -> List[Tuple[str, str, float]]:
        """All ``(relation, object, confidence)`` triples known about a subject."""
        resolved = self.resolve(subject)
        if resolved is None:
            return []
        out: List[Tuple[str, str, float]] = []
        seen = set()
        for relation, obj in reversed(self._by_s.get(resolved, [])):
            if (relation, obj) in seen:
                continue
            seen.add((relation, obj))
            confidence = 1.0
            if len(self._by_sr.get((resolved, relation), [])) > 1:
                for candidate, sim in self.probe(resolved, relation, top_k=64):
                    if candidate == obj:
                        confidence = sim
                        break
            out.append((relation, obj, confidence))
            if limit is not None and len(out) >= limit:
                break
        return out

    def subjects(self) -> List[str]:
        return list(self._by_s)

    def relations(self, subject: str) -> List[str]:
        resolved = self.resolve(subject) or subject
        return [r for r, _ in self._by_s.get(resolved, [])]

    def analogy(self, a: str, b: str, c: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Solve ``a is to b as c is to ?`` using relation transfer.

        Returns ``(relation, object, score)`` candidates sorted by score.  The
        score combines how well the source pair matches under a relation with a
        bonus for target subjects whose fact is already stored.
        """
        ra = self.resolve(a) or a
        rb = self.resolve(b) or b
        rc = self.resolve(c) or c
        scores: Dict[Tuple[str, str], float] = {}

        for relation, obj in self._by_s.get(ra, []):
            source = self._symbol_similarity(obj, rb)
            if source < 0.6:
                continue
            for target in self._by_sr.get((rc, relation), []):
                bonus = 0.05 if target in self._by_o else 0.0
                key = (relation, target)
                scores[key] = max(scores.get(key, 0.0), min(1.0, source + bonus))

        # Reverse-relation phrasing: "Sun is to Earth as ? is to Moon"
        if not scores:
            for relation, subject in self._by_o.get(rb, []):
                source = self._symbol_similarity(subject, ra)
                if source < 0.6:
                    continue
                for target, rel2 in self._by_o.get(rc, []):
                    if rel2 == relation:
                        key = (relation, target)
                        scores[key] = max(scores.get(key, 0.0), source)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(relation, obj, score) for (relation, obj), score in ranked[:top_k]]

    def chain(self, start: str, relations: Sequence[str]) -> List[str]:
        """Multi-hop traversal: follow ``relations`` from ``start``, return objects."""
        current = [self.resolve(start) or start]
        for relation in relations:
            nxt: List[str] = []
            for node in current:
                nxt.extend(obj for obj, _ in self.query(node, relation, top_k=64))
            current = nxt
            if not current:
                break
        return current

    def _symbol_similarity(self, a: str, b: str) -> float:
        """Similarity that understands names as well as hypervector geometry."""
        if a == b:
            return 1.0
        if normalise_symbol(a) == normalise_symbol(b):
            return 1.0
        hv_a = self.symbols.get(a)
        hv_b = self.symbols.get(b)
        if hv_a is None or hv_b is None:
            return 0.0
        ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
        return max(hv_a.similarity(hv_b), ratio)

    # ------------------------------------------------------------------ dunders
    def __len__(self) -> int:
        return len(self.facts)
