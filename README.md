# Eddi

A **Writable Hyperdimensional Computing (HDC) + 1M-Token Disk-Backed Context** agent architecture designed to run on consumer hardware without GPU VRAM bottlenecks.

## Architecture

1. **Writable Symbolic-Neural Memory (`eddi/hdc.py`):**
   - Uses 1024-bit packed binary Hyperdimensional Vectors (`HyperVector`) with algebraic XOR binding (`bind`) and Hamming similarity.
   - Facts and rules are written directly into memory (`write_fact`) without gradient descent or backpropagation.

2. **1M-Token Disk-Backed `mmap` Context (`src/eddi_core.cpp`, `eddi/context.py`):**
   - **L1 (RAM):** Fixed-size sliding window (`2048` tokens).
   - **L2 (Disk `mmap` + LSH):** Streams 1,000,000+ token records to a memory-mapped file with 16-bit LSH bucketing for $O(1)$ retrieval without holding the context in RAM or VRAM.

## Verified Benchmarks

- **C++20 Core (`bin/eddi_core bench 1000000`):**
  - Ingested **1,000,000 tokens** into disk-backed `mmap` context in **0.273s (~3.65M tokens/sec)**.
  - **1,000 LSH context queries** completed in **0.164 ms**.

## Quick Start

```bash
# Compile C++20 core & run 1M-token benchmark
mkdir -p bin && g++ -O3 -std=c++20 -march=native src/eddi_core.cpp -o bin/eddi_core
./bin/eddi_core bench 1000000

# Run unit tests & Python agent demo
python3 -m unittest discover -s tests -v
python3 -m eddi.cli --demo
```
