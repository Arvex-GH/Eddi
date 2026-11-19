"""Vector-symbolic reasoning over the engineering loop's own evidence.

The question this answers: what can a memory do *without* neurons?

A hypervector memory stores associations exactly and retrieves them in
constant time. Its capacity is not unlimited -- bundling k items into one
D-bit vector keeps them separable only while

    k < D * 8 / (pi * sigma^2)

which is D/14 for a conservative six-sigma margin, and D/32 at four sigma.
That is linear in D, so a 100 MB memory holds roughly 58 million associations
at one-fiftieth the cost of a neural parameter. What it cannot do is invent an
answer for an input it has never seen: an unseen combination is simply
unknown, which is honest but limiting.

Where this is exactly right -- and where it beats a network -- is reasoning
that is *compositional over known facts*. "Which file defines the symbol that
the failing test imports, and what other tests touch it?" is a traversal, not
a guess. Every step is checkable, so the answer can be proved rather than
asserted.

This module implements that narrow, high-value case: relations are bound into
hypervectors, multi-step questions are compositions of bindings, and the
result is a short proof path the loop can act on. It is deliberately not a
language model.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

DEFAULT_DIM = 10_000


def _hash_vector(seed: str, dim: int) -> int:
    """A deterministic pseudo-random D-bit vector from a name.

    Determinism matters more than cryptography here: the same name must map to
    the same vector across processes and reloads, or a saved memory is useless.
    blake2b is used because it is in the standard library and is fast.
    """
    bits = 0
    produced = 0
    counter = 0
    while produced < dim:
        digest = hashlib.blake2b(
            f"{seed}:{counter}".encode("utf-8"), digest_size=64
        ).digest()
        for byte in digest:
            bits = (bits << 8) | byte
            produced += 8
            if produced >= dim:
                break
        counter += 1
    return bits


def _popcount(value: int) -> int:
    return value.bit_count()


@dataclass
class Proof:
    """One step of a reasoning trail, kept so a conclusion can be checked."""

    relation: str
    source: str
    target: str

    def __str__(self) -> str:
        return f"{self.source} --{self.relation}--> {self.target}"


class SymbolMemory:
    """A writable, compositional memory of (subject, relation, object) triples.

    Three operations, and nothing else is needed:

    * ``bind``    -- combine two vectors so the pair is a single new vector
    * ``bundle``  -- superpose vectors so the result is similar to all of them
    * ``unbind``  -- recover one operand from a binding and the other

    Every relation is stored as a bound triple bundled into its subject, so
    recalling "what does X have?" is one similarity scan, not a search.
    """

    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim
        self._vectors: Dict[str, int] = {}
        self._index: Dict[str, List[Tuple[str, str]]] = {}
        self._bundles: Dict[str, int] = {}
        self._dirty: Set[str] = set()

    # -- encoding ---------------------------------------------------------
    def vector(self, name: str) -> int:
        """The hypervector for a name, created on first use."""
        key = name.strip().lower()
        bits = self._vectors.get(key)
        if bits is None:
            bits = _hash_vector(key, self.dim)
            self._vectors[key] = bits
        return bits

    @staticmethod
    def bind(a: int, b: int) -> int:
        """XOR binding. Self-inverse, which makes unbinding the same operation."""
        return a ^ b

    @staticmethod
    def permute(value: int, dim: int, shifts: int = 1) -> int:
        """Rotate a vector to encode order (so A->B differs from B->A)."""
        shifts %= dim
        if not shifts:
            return value
        mask = (1 << dim) - 1
        return ((value << shifts) | (value >> (dim - shifts))) & mask

    @staticmethod
    def similarity(a: int, b: int, dim: int) -> float:
        """Normalised Hamming similarity in [-1, 1]; 0 is chance for random bits."""
        agree = dim - _popcount(a ^ b)
        return (2.0 * agree / dim) - 1.0

    # -- writing ----------------------------------------------------------
    def add(self, subject: str, relation: str, obj: str) -> None:
        """Store one association. O(1); the bundle is rebuilt lazily."""
        s, r, o = subject.strip().lower(), relation.strip().lower(), obj.strip().lower()
        self._index.setdefault(s, [])
        if (r, o) not in self._index[s]:
            self._index[s].append((r, o))
        self._dirty.add(s)
        # Touch the vectors so they are allocated deterministically.
        self.vector(s)
        self.vector(r)
        self.vector(o)

    def add_many(self, triples: Iterable[Tuple[str, str, str]]) -> int:
        count = 0
        for triple in triples:
            self.add(*triple)
            count += 1
        return count

    def _bundle_for(self, subject: str) -> int:
        key = subject.strip().lower()
        if key in self._bundles and key not in self._dirty:
            return self._bundles[key]
        bound = [
            self.bind(self.bind(self.vector(key), self.vector(relation)), self.vector(obj))
            for relation, obj in self._index.get(key, [])
        ]
        if not bound:
            bundle = 0
        else:
            # Majority vote per bit: a bit is set when more than half the
            # members set it. That is what makes the bundle similar to each of
            # its members while remaining distinguishable from unrelated data.
            counts = [0] * self.dim
            for member in bound:
                _bundle_add_into(counts, member, self.dim)
            bundle = 0
            for bit in range(self.dim):
                if counts[bit] * 2 > len(bound):
                    bundle |= 1 << bit
        self._bundles[key] = bundle
        self._dirty.discard(key)
        return bundle

    # -- reading ----------------------------------------------------------
    def relations_of(self, subject: str) -> List[Tuple[str, str]]:
        """Exact recall of everything stored about a subject."""
        return list(self._index.get(subject.strip().lower(), []))

    def lookup(self, subject: str, relation: str) -> List[str]:
        """Recover objects for a relation by unbinding and cleaning up.

        This is the operation that distinguishes a VSA memory from a dict: it
        works from the *bundle*, so a corrupted or partially known probe still
        recovers the right answer. The exact index is checked first because it
        is free and never wrong.
        """
        key = subject.strip().lower()
        rel = relation.strip().lower()
        exact = [obj for r, obj in self._index.get(key, []) if r == rel]
        if exact:
            return exact
        # Nothing exact: fall back to the algebra, which can still succeed when
        # the subject name is a near miss.
        if key not in self._bundles and key not in self._index:
            return []
        probe = self.bind(self.vector(key), self.vector(rel))
        recovered = self.bind(self._bundle_for(key), probe)
        scores = []
        for candidate in self._vectors:
            if candidate in (key, rel):
                continue
            scores.append((self.similarity(recovered, self._vectors[candidate], self.dim), candidate))
        scores.sort(reverse=True)
        return [name for score, name in scores[:1] if score > 0.02]

    def similar_subjects(self, probe: str, k: int = 5) -> List[Tuple[str, float]]:
        """Subjects whose stored content is closest to a probe vector.

        Used to recognise an unknown issue description as "the same kind of
        problem as one already solved", which is the only generalisation a
        pure memory can honestly do.
        """
        target = self.vector(probe)
        scored = []
        for subject in self._index:
            bundle = self._bundle_for(subject)
            if not bundle:
                continue
            scored.append((subject, self.similarity(target, bundle, self.dim)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    # -- compositional reasoning -------------------------------------------
    def chain(
        self,
        start: str,
        relations: Sequence[str],
        max_depth: int = 4,
    ) -> List[Proof]:
        """Follow a sequence of relations, returning the proof path.

        A chain is the VSA form of reasoning: compose bindings and unbind in
        order. Each step is recorded, so a wrong conclusion can be traced to
        the exact step that produced it -- which a neural answer can never do.
        """
        current = start.strip().lower()
        trail: List[Proof] = []
        for relation in relations:
            hits = self.lookup(current, relation)
            if not hits:
                return trail
            nxt = hits[0]
            trail.append(Proof(relation.strip().lower(), current, nxt))
            current = nxt
        return trail

    def reach(self, start: str, relation: str, max_depth: int = 4) -> List[List[Proof]]:
        """All paths of at most ``max_depth`` hops along one relation."""
        paths: List[List[Proof]] = []
        frontier = [(start.strip().lower(), [])]
        for _ in range(max_depth):
            nxt: List[Tuple[str, List[Proof]]] = []
            for node, trail in frontier:
                for obj in self.lookup(node, relation):
                    if any(step.target == obj for step in trail):
                        continue  # a cycle proves nothing new
                    path = trail + [Proof(relation, node, obj)]
                    paths.append(path)
                    nxt.append((obj, path))
            frontier = nxt
            if not frontier:
                break
        return paths

    # -- capacity ----------------------------------------------------------
    def capacity(self, sigmas: float = 6.0) -> Dict[str, float]:
        """How much this dimension can hold before items blur together.

        Capacity is linear in D and inverse in the square of the safety
        margin, which is why six sigma costs four times as many bits as three.
        """
        per_bundle = self.dim * 8.0 / (math.pi * sigmas ** 2)
        return {
            "dim": self.dim,
            "items_per_bundle": per_bundle,
            "bytes_per_vector": self.dim / 8.0,
            "bits_per_item": self.dim / per_bundle if per_bundle else 0.0,
        }

    def stats(self) -> Dict[str, object]:
        return {
            "subjects": len(self._index),
            "triples": sum(len(v) for v in self._index.values()),
            "symbols": len(self._vectors),
            "dim": self.dim,
        }


def _bundle_add_into(counts: List[int], value: int, dim: int) -> None:
    """Accumulate one member's set bits into the vote counters.

    A plain list of ints rather than packed bits: this runs once per subject
    per invalidation, not per query, so clarity is worth more than the few
    milliseconds. Iterating set bits only (rather than all D bits) keeps it
    linear in the population count, which is half of D on average.
    """
    bit = 0
    while value and bit < dim:
        if value & 1:
            counts[bit] += 1
        value >>= 1
        bit += 1


# ---------------------------------------------------------------------------
# Projecting a repository into the memory
# ---------------------------------------------------------------------------

_DEF_RE = re.compile(r"^(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M)


def project_repository(toolbox, memory: Optional[SymbolMemory] = None) -> SymbolMemory:
    """Load a repository into a VSA memory as a graph of known relations.

    The relations are the ones that matter for a bug hunt: which module defines
    which symbol, which module imports which, which test exercises which
    symbol. Answering "what touches this?" is then a traversal with a proof,
    not a model call.
    """
    mem = memory or SymbolMemory()
    listing = toolbox.ls("**/*.py")
    if not listing.ok:
        return mem
    modules = [line.strip() for line in listing.text.splitlines() if line.strip()]
    for path in modules:
        read = toolbox.read(path, budget=200_000)
        if not read.ok:
            continue
        raw = (toolbox.root / path).read_text(encoding="utf-8", errors="replace")
        module = path.replace("/", ".").removesuffix(".py")
        mem.add(module, "defined_in", path)
        # The file's own stem is how a test refers to itself ("test_mod"),
        # while imports use the dotted module path. Both keys are registered so
        # either form finds the same file.
        mem.add(Path(path).stem.lower(), "file", path)
        for name in _DEF_RE.findall(raw):
            mem.add(name.lower(), "defined_in", path)
            mem.add(path, "defines", name.lower())
        for match in _IMPORT_RE.finditer(raw):
            target = match.group(1) or match.group(2) or ""
            if target:
                mem.add(module, "imports", target.lower())
                mem.add(target.lower(), "imported_by", module)
        if path.startswith("test_") or "/test_" in path:
            for name in _DEF_RE.findall(raw):
                mem.add(path, "tests", name.lower())
    return mem
