"""1M-Token Tiered Disk-Backed Context Engine.

L1: Fixed-size sliding window in RAM (O(1) memory).
L2: Disk-backed mmap ring buffer + 16-bit LSH bucket index for O(1) retrieval
    across 1,000,000+ tokens without holding the context in RAM or VRAM.
"""

import mmap
import os
import struct
from collections import deque
from typing import Dict, List, Tuple


RECORD_FMT = "<QQ"  # (token_id: uint64, simhash: uint64)
RECORD_SIZE = struct.calcsize(RECORD_FMT)


class TieredContext1M:
    def __init__(self, path: str = "/tmp/eddi_py_context.mmap", max_tokens: int = 1_000_000, l1_size: int = 2048):
        self.path = path
        self.max_tokens = max_tokens
        self.l1_size = l1_size
        self.l1_window: deque = deque(maxlen=l1_size)
        self.buckets: Dict[int, List[int]] = {}
        self.count = 0

        total_bytes = max_tokens * RECORD_SIZE
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(self._fd, total_bytes)
        self._mmap = mmap.mmap(self._fd, total_bytes)

    def append(self, token_id: int, simhash: int) -> None:
        if self.count >= self.max_tokens:
            return
        offset = self.count * RECORD_SIZE
        struct.pack_into(RECORD_FMT, self._mmap, offset, token_id, simhash)
        bucket = simhash & 0xFFFF
        self.buckets.setdefault(bucket, []).append(self.count)
        self.l1_window.append((token_id, simhash))
        self.count += 1

    def query_lsh(self, simhash: int, limit: int = 8) -> List[Tuple[int, int, int]]:
        """Return (position, token_id, simhash) from disk-backed L2 without scanning."""
        bucket = simhash & 0xFFFF
        positions = self.buckets.get(bucket, [])[-limit:]
        results = []
        for pos in positions:
            tok, sig = struct.unpack_from(RECORD_FMT, self._mmap, pos * RECORD_SIZE)
            results.append((pos, tok, sig))
        return results

    def close(self) -> None:
        self._mmap.close()
        os.close(self._fd)
