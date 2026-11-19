"""1M-Token Tiered Disk-Backed Context Engine.

L1: Fixed-size sliding window in RAM (O(1) memory).
L2: Disk-backed mmap ring buffer + 16-bit LSH bucket index for fast retrieval
    across 1,000,000+ tokens without holding the context in RAM or VRAM.

Cross-platform: the backing file lives in the OS temp directory (``/tmp`` on
Linux, ``%TEMP%`` on Windows) and is memory-mapped with the standard library
``mmap`` module, so the same code runs on Windows, macOS and Linux.
"""

from __future__ import annotations

import mmap
import os
import struct
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from .paths import default_context_path

RECORD_FMT = "<QQ"  # (token_id: uint64, simhash: uint64)
RECORD_SIZE = struct.calcsize(RECORD_FMT)


class TieredContext1M:
    def __init__(
        self,
        path: Optional[str] = None,
        max_tokens: int = 1_000_000,
        l1_size: int = 2048,
        create: bool = True,
    ):
        self.path = path or default_context_path("py_context")
        self.max_tokens = max(1, int(max_tokens))
        self.l1_size = l1_size
        self.l1_window: Deque[Tuple[int, int]] = deque(maxlen=l1_size)
        self.buckets: Dict[int, List[int]] = {}
        self.count = 0
        self._closed = True
        self._mmap = None
        self._fd = -1

        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        total_bytes = self.max_tokens * RECORD_SIZE
        self._fd = os.open(str(target), os.O_RDWR | os.O_CREAT | (os.O_TRUNC if create else 0), 0o644)
        current = os.fstat(self._fd).st_size
        if current < total_bytes:
            os.ftruncate(self._fd, total_bytes)
        self._mmap = mmap.mmap(self._fd, total_bytes, access=mmap.ACCESS_WRITE)
        self._closed = False

    # ------------------------------------------------------------------ write
    def append(self, token_id: int, simhash: int) -> None:
        if self._closed or self.count >= self.max_tokens:
            return
        offset = self.count * RECORD_SIZE
        struct.pack_into(RECORD_FMT, self._mmap, offset, token_id & 0xFFFFFFFFFFFFFFFF, simhash)
        self.buckets.setdefault(simhash & 0xFFFF, []).append(self.count)
        self.l1_window.append((token_id, simhash))
        self.count += 1

    def append_many(self, pairs: "List[Tuple[int, int]]") -> int:
        written = 0
        for token_id, simhash in pairs:
            if self.count >= self.max_tokens:
                break
            self.append(token_id, simhash)
            written += 1
        return written

    # ------------------------------------------------------------------- read
    def query_lsh(self, simhash: int, limit: int = 8) -> List[Tuple[int, int, int]]:
        """Return ``(position, token_id, simhash)`` from disk-backed L2 without scanning."""
        if self._closed:
            return []
        bucket = self.buckets.get(simhash & 0xFFFF, [])
        results: List[Tuple[int, int, int]] = []
        for pos in bucket[-limit:]:
            tok, sig = struct.unpack_from(RECORD_FMT, self._mmap, pos * RECORD_SIZE)
            results.append((pos, tok, sig))
        return results

    def read(self, position: int) -> Optional[Tuple[int, int]]:
        """Random access into the mmap'd store."""
        if self._closed or not 0 <= position < self.count:
            return None
        return struct.unpack_from(RECORD_FMT, self._mmap, position * RECORD_SIZE)

    def l1(self) -> List[Tuple[int, int]]:
        return list(self.l1_window)

    # -------------------------------------------------------------- lifecycle
    def flush(self) -> None:
        if not self._closed:
            self._mmap.flush()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._mmap.flush()
        except (ValueError, BufferError):
            pass
        try:
            self._mmap.close()
        finally:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
            self._closed = True

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown helper
        try:
            self.close()
        except Exception:
            pass
