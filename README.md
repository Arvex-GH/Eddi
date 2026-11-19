# Eddi

**A writable hyperdimensional memory + a trained language model + a 1M-token disk-backed context — running on consumer hardware, no GPU and no VRAM bottleneck.**

Eddi is not a transformer and does not pretend to be one. It is a compact, local agent
that *remembers what you teach it*, *reasons over those memories*, *does math*, and
*writes text from a language model it has actually been trained on*. Everything runs on
Windows, macOS and Linux from a single Python package.

```
you> what is the moon of earth
eddi [ask]> Moon --orbits--> Earth

you> Earth is to Sun as Moon is to ?
eddi [analogy]> Moon --orbits--> Earth  (by analogy with Earth --orbits--> Sun, score 1.00)

you> chain Moon | orbits | orbits
eddi [chain]> Moon -> Sun

you> 17 * 3 + 4
eddi [math]> 55

you> The compiler
eddi [generate]> The compiler translates source code into machine code before the program runs.
```

---

## Install

### Windows / macOS / Linux (Python 3.9+)

```bash
git clone https://github.com/Arvex-GH/Eddi
cd Eddi
python -m pip install .
eddi --train      # train the built-in curriculum (~1 second)
eddi --chat       # talk to Eddi
eddi-gui          # optional desktop window (Tkinter)
```

Every tagged release ships **prebuilt binaries for Windows x86_64, Linux x86_64 and macOS x86_64**,
each with a standalone bundle (`eddi-standalone-<platform>.{zip,tar.gz}`) that contains the
C++ core, the Python package and a short README. Nothing needs to be compiled to use Eddi —
the C++ core is an optional accelerator.

### Windows notes

* No build tools required: `pip install .` gives you `eddi` and `eddi-gui` console entry points.
* Temporary files use `%TEMP%` (via `tempfile`); there is no hard-coded `/tmp` anywhere.
* The C++ core binary is `eddi_core.exe`; the prebuilt Windows core is compiled with MSYS2 g++
  on `windows-latest` in CI.

---

## What makes Eddi smart

| Capability | How it works | Where |
|---|---|---|
| **Writable memory** | Facts `(subject, relation, object)` are written as 1024-bit hypervectors: `bind(S, R, O) = S ⊕ R ⊕ O`. Learning is a write, not a training step. | `eddi/hdc.py` |
| **O(1) retrieval** | A writable hash index `(subject, relation) → objects` answers exactly; a lazily built **majority-vote bundle** plus HDC cleanup recovers objects even from noisy probes. | `eddi/hdc.py` |
| **Trained language model** | Interpolated back-off n-gram model (default: 2-token context, `--order 3`) with **absolute discounting**: interpolation weights depend on evidence, so sparse long contexts cannot overrule solid short ones. Deeper contexts (`--order 4/5`) pay off on larger corpora. | `eddi/language.py` |
| **Generalisation beyond n-grams** | A second learning channel over **word-type classes** (cheap stemming: `runs/running/ran → run`, numbers → `<num>`) lets unseen token contexts still be answered. | `eddi/language.py` |
| **Reasoning router** | Questions are parsed and routed: teach, ask, attribute, about, relate, chain (multi-hop), analogy, math, generate — with honest "I don't know". | `eddi/reasoning.py` |
| **1M-token context** | L1 sliding window in RAM + L2 `mmap` ring buffer with 16-bit LSH buckets; ~9.2M tokens/sec ingestion in the C++ core. | `eddi/context.py`, `src/eddi_core.cpp` |
| **Measurable training** | Held-out perplexity, in-vocabulary perplexity, bigram baseline, n-gram coherence, fact accuracy and reasoning accuracy after every training run. | `eddi/evaluation.py`, `eddi/training.py` |
| **Portable brain** | Facts + model + counts serialise to one `.pkl` (`eddi-brain-v2`), loadable on any platform. | `eddi/training.py` |

---

## Train it (a little or a lot)

```bash
# built-in curriculum: identity, science, computing, reasoning, geography
eddi --train --epochs 5

# learn from your own text files (any directory, recursive *.txt)
eddi --train --corpus ~/books --epochs 3 --max-tokens 2000000

# only some packs, different context length
eddi --train --pack science --pack computing --order 4

# inspect the resulting brain
eddi --bench --json
```

A training run prints a report like this:

```
=== Eddi training report ===
  facts written     : 115
  sentences trained : 360
  tokens trained    : 5032
  vocabulary        : 841
  model order       : 3
  perplexity before : inf
  perplexity after  : 325.25  (lower is better)
  perplexity in-voc : 119.15  (known words only)
  bigram baseline   : 91.14  -> bigram is 3.57x better on this tiny corpus - add more training text for this model to win
  bigram coherence  : 76.4%
  fact accuracy     : 100.0%
  reasoning accuracy: 100.0%
```

The held-out set is technical prose while the extra corpus is Shakespeare, so **cross-domain
perplexity is noisy**. Measured on one domain-mixed corpus (curriculum + 1 MB Shakespeare,
same held-out set) the comparison is:

| model | held-out perplexity |
|---|---|
| bigram chain (`order=2` context) | 1717 |
| 2-token context (`eddi --order 3`) | **846** |
| 3-token context (`eddi --order 4`) | 887 |

*(A 2-token context extracts roughly 2x more signal than a bigram chain here. Deep contexts
need much more data before they pay off - which is exactly what the `--corpus` flag is for.)*

The lesson is stated plainly rather than hidden: **more data is what makes Eddi smarter**,
and the `--corpus` flag is the lever.

The benchmark is **held out**: `eddi/evaluation.py` contains sentences and questions that
are never part of the training curriculum, so the numbers measure generalisation.

Trained on ~1 MB of Shakespeare (227k tokens, ~1.4 s at ~164k tokens/sec):

```
ROMEO: My lord, I am too much already: get thee gone, I say so.
predict(["the", "king", "is"]) -> dead, not, full, left, come
held-out perplexity (in-vocabulary): 164
```

### Growing it massively

Eddi is a from-scratch learner, so quality scales with the corpus you give it:

* point `--corpus` at a folder of public-domain books, Wikipedia text dumps or the
  arXiv metadata snapshot (`eddi/arxiv_ingest.py`);
* raise `--order` to 4 or 5 once the corpus is large (roughly: >1M tokens per context step);
* each extra epoch reinforces transitions the model already saw;
* `--max-tokens` sets a hard budget (default 2M) and the context store keeps streaming
  to disk beyond RAM;
* `--prune N` drops n-grams seen fewer than N times to keep saved brains small.

Eddi does **not** download anything on its own and never calls a cloud API.

---

## CLI reference

```
eddi                        status: brain, size, C++ core, held-out perplexity
eddi --demo                 capability tour (memory, analogy, chain, math, generation)
eddi --train [--pack P]... [--corpus DIR]... [--epochs N] [--order N] [--prune N]
eddi --bench [--json]       held-out benchmark with miss explanations
eddi --ask "..." [--explain]
eddi --chat [--save-on-exit]
eddi --serve [--host H] [--port 8765]   JSON API: GET /health /stats, POST /ask
eddi-gui                    Tkinter desktop app (train, chat, ingest, save)
```

`--brain PATH` points at a different brain file (default: `eddi_brain.pkl` in the OS temp dir;
use `--brain ./eddi_brain.pkl` to keep it next to your project).

### HTTP API

```bash
eddi --serve --port 8765 &
curl -s -X POST localhost:8765/ask -H 'Content-Type: application/json' \
     -d '{"text": "what is the moon of earth"}'
# {"text": "Moon --orbits--> Earth", "kind": "ask", "confidence": 1.0}
```

---

## Build the C++ core (optional)

```bash
# Linux / macOS
g++ -O3 -std=c++20 -Wall -Wextra src/eddi_core.cpp -o bin/eddi_core
./bin/eddi_core bench 1000000

# Windows (MSYS2 / MinGW-w64)
g++ -O3 -std=c++20 -Wall -Wextra src/eddi_core.cpp -o bin\eddi_core.exe
bin\eddi_core.exe bench 1000000
```

Verified on this repository: **1,000,000 tokens in 0.108 s (9.2M tokens/sec)**, and
1,000 LSH context queries in 0.25 ms. The C++ core is used by the release assets; the Python
agent falls back to `mmap` automatically when it is absent.

---

## Tests

```bash
python -m unittest discover -s tests -v     # 28 tests, cross-platform
```

The suite covers hypervector algebra (bind/unbind, bundles, permutation), fuzzy knowledge
lookup, analogy and multi-hop chaining, language-model normalisation/generalisation/persistence,
question routing, math sandboxing, training + brain round-trip, and Windows-safe paths.

CI (`.github/workflows/ci.yml`) runs the tests plus a train/benchmark quality gate on
**Ubuntu, Windows and macOS**. Releases (`.github/workflows/release.yml`) run the same tests
on Python 3.9 and 3.12, then build `windows-x86_64`, `linux-x86_64` and `macos-x86_64`
assets in a matrix (so the platform jobs never race on the same release) plus `sdist`/`wheel`.

---

## Architecture

```
                    ┌──────────────────────────── eddi/ ────────────────────────────┐
  you ──► eddi.cli  │ reasoner ──► hdc.WritableKnowledgeBase   (facts, analogy, chain) │
        eddi-gui    │        └──► language.NGramLanguageModel  (trained text model)     │
        /ask API    │        └──► context.TieredContext1M      (L1 RAM + L2 mmap + LSH)  │
                    │ training ──► curriculum / your corpus / evaluation ──► brain.pkl  │
                    └───────────────────────────────────────────────────────────────┘
                                   optional accelerator: src/eddi_core.cpp
```

* **Memory is written, not trained.** One `teach` call is one fact, forever.
* **Language is trained, not prompted.** No cloud, no API key, no GPU.
* **Answers are traceable.** Every response carries a `kind` (`ask`, `about`, `chain`,
  `analogy`, `math`, `generate`, `unknown`) and optional reasoning details (`--explain`).

## Limitations (honest ones)

* The language model is an n-gram model: it writes fluent short passages from its training
  distribution, but it does not follow complex instructions or understand long documents.
* Fact accuracy is measured on the built-in curriculum's held-out questions; questions far
  outside that vocabulary return `unknown` rather than a guess.
* The 1M-token context is a retrieval store and recorder, not attention: it streams, buckets
  and recalls tokens, while reasoning over long documents happens through the n-gram model
  and the knowledge base.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
