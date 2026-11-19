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


def _stem(word: str) -> str:
    """Conservative English suffix stripping, enough to match inflections."""
    w = word.lower()
    for suffix in ("ational", "ization", "iveness", "fulness", "ousness"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    for suffix in ("ingly", "edly", "ation", "ition", "ment", "ness", "ing", "ed", "es", "ly", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)]
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
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

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            Hit(
                chunk=self._by_id[cid],
                score=score,
                matched_terms=matched.get(cid, []),
                matched_identifiers=ident_matches.get(cid, []),
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
