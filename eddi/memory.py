"""Verbatim, retrievable memory for Eddi.

The previous 1M-token "context" stored ``(token_id, simhash)`` pairs in an mmap
ring. It could not return text, no code path queried it while answering, and
nothing wrote to it during normal use - it was a store, not a memory. This
module is the replacement.

Design
------
**Chunks of verbatim text.** Text is split into chunks of roughly
``chunk_tokens`` tokens on sentence/paragraph boundaries. Every chunk is stored
exactly as written and is addressable by integer id. Nothing is ever rewritten
or summarised, so a bad ranking costs latency, never information.

**Three indexes, all cheap and deterministic.**

* *terms* - an inverted index over lowercased word stems, with document
  frequencies for BM25 scoring.
* *identifiers* - exact keys for names like ``C++_RAII``, ``ERR_2``,
  ``/tmp/build.py``, matched as whole tokens so ``parse_1`` never matches
  ``parse_12``.
* *order* - chunks are stored in arrival order, so recency is free.

**Retrieval is query time.** A query is scored with BM25 over the term index,
boosted when it names an identifier that appears in a chunk. This needs no
model, costs microseconds, and cannot drift between runs.

Storage is an append-only JSONL file. Appending means a crash loses at most the
last line, and loading skips a partial trailing line rather than failing.

Usage::

    mem = ChunkMemory(path="~/.local/share/eddi/memory.jsonl")
    mem.add("Water has the chemical formula H2O.", source="lesson:chemistry")
    for hit in mem.search("what is the chemical formula of water", k=3):
        print(hit.chunk.text, hit.score)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Tokenising and stemming
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+#'./-]*")
_IDENT_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_ERRCODE_RE = re.compile(r"\b[A-Z]{2,}-?\d+\b")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")

STOPWORDS: Set[str] = set(
    """
    a an and are as at be been being but by can could did do does for from had has
    have he her here him his how i if in into is it its just me more most my no nor
    not of on or other our out over own same she should so some such than that the
    their them then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your
    """.split()
)


#: Words that end in a suffix but are not inflections of anything. Stripping
#: blindly is what turned "speed" into the nonsense stem "spe", which matched
#: nothing yet displaced real matches -- a question about catalysts lost to a
#: sentence about the speed of light because the two shared a broken stem.
_NOT_INFLECTIONS = frozenset(
    """speed seed feed need weed deed creed breed steed reed tweed bleed heed
    bend spend trend blend mend send lend rend tend wend
    hundred sacred hatred kindred naked wicked rugged jagged ragged
    bed red shed sled shred bedstead""".split()
)


def _stem(word: str) -> str:
    """Reduce a word to a root shared by its inflections, without inventing one.

    A stemmer exists so that "reaction" and "reactions" match, and so that
    "compute" and "computed" do. It must never *invent* a root that is shorter
    than a word: that produces a token which matches nothing yet competes with
    everything. Every rule here therefore keeps the result at least as long as
    a plausible English root, and a short list of words that merely look
    inflected is left alone.
    """
    w = word.lower()
    if len(w) <= 3:
        return w
    if w in _NOT_INFLECTIONS:
        return w
    for suffix in ("ational", "ization", "iveness", "fulness", "ousness"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]

    def plausible(root: str) -> bool:
        return len(root) >= 3 and any(c in "aeiouy" for c in root)

    def undouble(root: str) -> str:
        # "stopped" -> "stopp" -> "stop"; "planned" -> "plan".
        if len(root) > 3 and root[-1] == root[-2] and root[-1] not in "aeiou":
            return root[:-1]
        return root

    # --- plural and third person singular -----------------------------------
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"                      # studies -> study
    elif w.endswith("sses"):
        w = w[:-2]                            # processes -> process
    elif w.endswith("es") and len(w) > 4:
        root = w[:-2]
        if root.endswith(("s", "x", "z", "ch", "sh")):
            w = root                          # boxes -> box, processes -> process
        else:
            w = w[:-1]                        # sizes -> size
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        singular = w[:-1]                     # catalysts -> catalyst
        w = singular

    # --- past tense and progressive -----------------------------------------
    if w in _NOT_INFLECTIONS:
        return w
    if w.endswith("ed") and not w.endswith("eed"):
        # "used" is the past of "use", not a stem of "us"; "carried" keeps its
        # y so it meets "carry".
        if w.endswith("ied") and len(w) > 4:
            w = w[:-3] + "y"
        else:
            root = undouble(w[:-2])
            if plausible(root):
                w = root
            elif w not in _NOT_INFLECTIONS:
                w = w[:-1]                    # used -> use
    elif w.endswith("ing"):
        root = undouble(w[:-3])
        if plausible(root):
            w = root

    # --- silent final "e", so "freeze" and "freezes" meet -------------------
    if w in _NOT_INFLECTIONS:
        return w
    if len(w) >= 4 and w.endswith("e") and not w.endswith("ee"):
        root = w[:-1]
        if plausible(root):
            w = root
    return w


def terms_of(text: str) -> List[str]:
    """Content stems in ``text``: lowercased, stopworded, deduplicated."""
    out: List[str] = []
    seen: Set[str] = set()
    for raw in _WORD_RE.findall(text):
        low = raw.lower()
        if low in STOPWORDS or len(low) < 2:
            continue
        stem = _stem(low)
        if stem in seen:
            continue
        seen.add(stem)
        out.append(stem)
    return out


def identifiers_of(text: str) -> List[str]:
    """Names worth indexing exactly: snake_case, error codes, paths."""
    found: List[str] = []
    for raw in _WORD_RE.findall(text):
        token = raw.strip(".,;:!?()[]{}\\\"'`")
        if not token:
            continue
        if _IDENT_RE.fullmatch(token) or _ERRCODE_RE.fullmatch(token):
            found.append(token.lower())
        elif "/" in token or "\\" in token:
            found.append(token.lower())
    return list(dict.fromkeys(found))


def token_count(text: str) -> int:
    return max(1, len(text.split()))


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """One stored span of verbatim text."""

    id: int
    text: str
    source: str = ""
    ts: float = field(default_factory=time.time)
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = token_count(self.text)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "ts": self.ts,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Chunk":
        return cls(
            id=int(data["id"]),
            text=str(data["text"]),
            source=str(data.get("source", "")),
            ts=float(data.get("ts", 0.0) or 0.0),
            tokens=int(data.get("tokens", 0) or 0),
        )


@dataclass
class Hit:
    """A retrieval result: the chunk, its score, and why it scored."""

    chunk: Chunk
    score: float
    matched_terms: List[str] = field(default_factory=list)
    matched_identifiers: List[str] = field(default_factory=list)
    #: The heaviest query term, when this hit contains it. It names what the
    #: question is *about*, and a hit that lacks it cannot be the answer.
    subject: str = ""
    missing_subject: bool = False

    @property
    def text(self) -> str:
        return self.chunk.text


def split_into_chunks(
    text: str, chunk_tokens: int = 180, min_tokens: int = 12
) -> List[str]:
    """Split text on sentence boundaries into spans of about ``chunk_tokens``.

    Splitting on sentences rather than fixed token counts keeps every chunk
    readable on its own - a chunk that starts mid-sentence is much harder to use
    as evidence, even though it is the same number of tokens.
    """
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for sentence in sentences:
        n = token_count(sentence)
        if current and current_tokens + n > chunk_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += n
    if current:
        chunks.append(" ".join(current))

    # fold a runt tail into the previous chunk rather than storing a fragment
    if len(chunks) > 1 and token_count(chunks[-1]) < min_tokens:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + " " + tail
    return chunks


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class ChunkMemory:
    """Append-only, verbatim, queryable memory."""

    def __init__(
        self,
        path: Optional[str] = None,
        chunk_tokens: int = 180,
        auto_load: bool = True,
    ):
        self.path = Path(path).expanduser() if path else None
        self.chunk_tokens = chunk_tokens
        self.chunks: List[Chunk] = []
        self._by_id: Dict[int, Chunk] = {}
        self._postings: Dict[str, List[int]] = {}
        self._ident: Dict[str, List[int]] = {}
        self._lengths: List[int] = []
        self._total_tokens = 0
        self._next_id = 0
        self._seen: Set[str] = set()
        if auto_load and self.path and self.path.exists():
            self.load()

    # -- writing -----------------------------------------------------------
    def add(self, text: str, source: str = "") -> List[Chunk]:
        """Store ``text`` verbatim, split into chunks. Returns what was stored."""
        key = " ".join(text.split()).lower()
        if key and key in self._seen:
            return []
        if key:
            self._seen.add(key)
        added: List[Chunk] = []
        for piece in split_into_chunks(text, chunk_tokens=self.chunk_tokens):
            chunk = Chunk(id=self._next_id, text=piece, source=source)
            self._next_id += 1
            self._seen.add(" ".join(chunk.text.split()).lower())
            self._index(chunk)
            self.chunks.append(chunk)
            self._by_id[chunk.id] = chunk
            added.append(chunk)
            self._append_to_disk(chunk)
        return added

    def add_many(self, items: Iterable[Tuple[str, str]]) -> int:
        """``[(text, source), ...]`` -> number of chunks written."""
        n = 0
        for text, source in items:
            n += len(self.add(text, source=source))
        return n

    def _index(self, chunk: Chunk) -> None:
        for term in terms_of(chunk.text):
            self._postings.setdefault(term, []).append(chunk.id)
        for ident in identifiers_of(chunk.text):
            self._ident.setdefault(ident, []).append(chunk.id)
        self._lengths.append(chunk.tokens)
        self._total_tokens += chunk.tokens

    def _append_to_disk(self, chunk: Chunk) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    # -- loading -----------------------------------------------------------
    def load(self) -> int:
        """Read the store back, skipping a partial trailing line."""
        if not self.path or not self.path.exists():
            return 0
        count = 0
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # A crash mid-write leaves a partial line; keep what is good.
                    continue
                try:
                    chunk = Chunk.from_dict(data)
                except (KeyError, TypeError, ValueError):
                    continue
                if chunk.id in self._by_id:
                    continue
                self._seen.add(" ".join(chunk.text.split()).lower())
                self._index(chunk)
                self.chunks.append(chunk)
                self._by_id[chunk.id] = chunk
                self._next_id = max(self._next_id, chunk.id + 1)
                count += 1
        return count

    def compact(self) -> int:
        """Rewrite the file with only valid lines. Returns chunks written."""
        if not self.path:
            return 0
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)
        return len(self.chunks)

    # -- reading -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def get(self, chunk_id: int) -> Optional[Chunk]:
        return self._by_id.get(chunk_id)

    def search(
        self,
        query: str,
        k: int = 5,
        identifier_boost: float = 6.0,
        phrase_boost: float = 3.0,
        exclude_sources: Optional[Sequence[str]] = None,
    ) -> List[Hit]:
        """Rank chunks for a query with BM25 plus identifier and phrase boosts.

        Identifier matching is exact and dominates: if the query names a symbol
        that appears in a chunk, that chunk is almost always the right answer,
        while `k1`/`b` tuned for prose would bury it under common words.
        """
        if not self.chunks:
            return []
        q_terms = terms_of(query)
        q_idents = identifiers_of(query)
        if not q_terms and not q_idents:
            return []

        n_docs = len(self.chunks)
        avg_len = (self._total_tokens / n_docs) if n_docs else 1.0
        k1, b = 1.5, 0.75
        scores: Dict[int, float] = {}
        matched: Dict[int, List[str]] = {}

        for term in q_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for chunk_id in postings:
                chunk = self._by_id[chunk_id]
                length = max(1, chunk.tokens)
                tf = chunk.text.lower().count(term) or 1
                denom = tf + k1 * (1 - b + b * length / avg_len)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (tf * (k1 + 1)) / denom
                matched.setdefault(chunk_id, []).append(term)

        # Coverage: how much of the query a chunk actually contains. BM25 sums
        # evidence per term, so a chunk that repeats one common word can outrank
        # the single chunk that answers everything asked. Asked "why does a
        # catalyst speed up a reaction", a sentence about enzymes (which says
        # "speed" and "reactions" but never mentions a catalyst) beat both
        # catalyst sentences. A chunk matching every query term is worth far
        # more than one matching half of them, so the score is scaled by
        # coverage with a floor that keeps partial matches usable.
        subject = ""
        if q_terms:
            # Weight each query term by how rare it is. Covering "photosynthesis"
            # says far more than covering "take", and an unweighted ratio let a
            # sentence about logarithms win a question about photosynthesis
            # because both happened to contain one of the two words.
            # Terms that the query capitalises, which usually name an entity.
            capitalised = {
                _stem(w) for w in _WORD_RE.findall(query) if w[:1].isupper()
            }
            doc_freq: Dict[str, int] = {
                t: len(self._postings.get(t, ())) for t in q_terms
            }
            idf_of: Dict[str, float] = {}
            for term in q_terms:
                df = len(self._postings.get(term, ()))
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                # Rarity alone is not enough on a small store: "take" is as rare
                # here as "photosynthesis" simply because the corpus is young.
                # A longer word also names a narrower idea -- photosynthesis is
                # a whole subject, take is a verb that fits anywhere -- so the
                # two signals are multiplied.
                # Capitalisation in the original query is a strong signal that a
                # word names an entity ("the Moon", "Switzerland") rather than a
                # category ("planet", "chemical"). Categories are common nouns
                # that the store has in abundance; entities it has once. This
                # was chosen after "which planet does the Moon orbit" picked
                # "planet" as its subject and answered with the largest planet
                # rather than with the Moon's orbit.
                weighted = idf * len(term)
                if term in capitalised:
                    weighted *= 1.8
                if term in doc_freq and doc_freq[term] <= 1:
                    weighted *= 1.05
                idf_of[term] = weighted
            total_idf = sum(idf_of.values()) or 1.0
            # The heaviest term is the subject of the question: in "why does a
            # catalyst speed up a reaction" that is "catalyst", which names what
            # is being asked about. A chunk that never mentions the subject
            # cannot be the answer, however many supporting words it shares --
            # which is exactly how a sentence about enzymes displaced both
            # catalyst sentences, matching "speed" and "reaction" while
            # containing no catalysts at all.
            # Break ties by position: English puts the topic early, so in
            # "why does a catalyst speed up a reaction" the subject is
            # "catalyst", not the equally weighted "reaction" that follows it.
            order = {term: i for i, term in enumerate(q_terms)}
            subject = (
                max(idf_of, key=lambda t: (idf_of[t], -order.get(t, 0)))
                if idf_of
                else ""
            )
            for cid in scores:
                covered = set(matched.get(cid, []))
                share = sum(idf_of.get(t, 0.0) for t in covered) / total_idf
                factor = 0.35 + 0.65 * share ** 2
                if subject and subject not in covered:
                    factor *= 0.3
                scores[cid] *= factor

        ident_matches: Dict[int, List[str]] = {}
        for ident in q_idents:
            for chunk_id in self._ident.get(ident, []):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + identifier_boost
                ident_matches.setdefault(chunk_id, []).append(ident)

        # A contiguous phrase hit is strong evidence the chunk is *about* it.
        q_lower = " ".join(query.lower().split())
        for chunk_id in list(scores):
            if q_lower and len(q_lower) > 8 and q_lower in self._by_id[chunk_id].text.lower():
                scores[chunk_id] += phrase_boost

        if exclude_sources:
            # A stored *question* matches the next question on every word and
            # answers none of them: asked "why does a catalyst speed up a
            # reaction", the agent retrieved its own earlier phrasing of that
            # question and replied with it. Questions stay in the store (they
            # carry context) but retrieval can leave them out.
            skip = set(exclude_sources)
            scores = {
                cid: score
                for cid, score in scores.items()
                if self._by_id[cid].source not in skip
            }

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            Hit(
                chunk=self._by_id[cid],
                score=score,
                matched_terms=matched.get(cid, []),
                matched_identifiers=ident_matches.get(cid, []),
                subject=subject if subject in set(matched.get(cid, [])) else "",
                missing_subject=bool(subject) and subject not in set(matched.get(cid, [])),
            )
            for cid, score in ranked
        ]

    def recent(self, k: int = 5) -> List[Chunk]:
        return self.chunks[-k:] if k > 0 else []

    def stats(self) -> Dict[str, object]:
        return {
            "chunks": len(self.chunks),
            "tokens": self._total_tokens,
            "terms": len(self._postings),
            "identifiers": len(self._ident),
            "path": str(self.path) if self.path else None,
        }

    def digest(self) -> str:
        """Cheap fingerprint of the whole store, useful for save checks."""
        hasher = hashlib.blake2b(digest_size=8)
        for chunk in self.chunks:
            hasher.update(chunk.text.encode("utf-8"))
        return hasher.hexdigest()
