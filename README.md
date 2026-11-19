# Eddi

**A writable hyperdimensional memory + verbatim text memory + a trained language model — running on consumer hardware, no GPU and no VRAM bottleneck.**

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

## Where everything Eddi learns is stored

Learning lives in **two stores**, both on disk and both inspectable:

| Store | Contains | Where | Survives restart |
|---|---|---|---|
| **Brain file** | facts, rules, aliases, derived facts, the language model | `~/.local/share/eddi/eddi_brain.pkl` (override with `EDDI_BRAIN`) | yes |
| **Verbatim memory** | what you told it, as real text, chunked and searchable | `~/.local/share/eddi/eddi_memory.jsonl` (override with `EDDI_MEMORY`) | yes |

The brain is deliberately **not** in the temp directory: it is learned state, and temp
directories get cleaned by the OS. `eddi --status` shows both paths and whether the brain
exists.

Nothing is silently thrown away:

* `agent.close()` saves the brain when `autosave=True` (or a `brain_path` is set);
* `eddi --chat` always saves on exit unless you pass `--no-save`;
* `eddi --ask` prints a warning when a question wrote something and `--save` was not given;
* closing an agent **no longer deletes** the memory file, and reopening it restores every
  chunk so you continue instead of starting from zero.

## Two ways to make Eddi smarter

The original goal is preserved exactly: **you can write intelligence into Eddi without training anything.**

| | Write (instant, 0 tokens trained) | Train (minutes, scales with corpus) |
|---|---|---|
| **What you add** | facts, rules, aliases, hierarchies | fluent language, word statistics |
| **Cost** | one call, microseconds | O(corpus), ~164k tokens/sec |
| **Answers via** | exact lookup, forward chaining, analogy, multi-hop | n-gram continuation |
| **Survives save/load** | yes (`rules`, `aliases`, `derived` in the brain file) | yes (model in the brain file) |

```bash
# nothing is trained here - lm.trained_tokens stays 0
# --save persists whatever your question wrote (facts, rules, aliases)
eddi --ask "teach cat | is_a | mammal"          --save
eddi --ask "rule is_a => warm_blooded"          --save   # writable inference
eddi --ask "teach whale | is_a | mammal"        --save
eddi --ask "alias sea giant = whale"            --save
eddi --ask "what is the warm_blooded of sea giant"
#   -> whale --warm_blooded--> mammal
```

```
$ eddi --ask "teach whale | is_a | mammal" --save
Learned: (whale) --is_a--> (mammal) (+1 derived by rules)
```

**Writable inference.** A rule is a write, not a gradient step. `rule is_a => warm_blooded`
forward-chains over everything Eddi already knows:

* `rule is_a => is_a` computes the **transitive closure** of a hierarchy
  (`cat --is_a--> mammal --is_a--> animal` ⇒ `cat --is_a--> animal`),
* `rule is_a => warm_blooded` propagates the conclusion up that hierarchy,
* rules compose, run idempotently, and are re-checked on every `teach` so new
  facts immediately inherit;
* `kb.explain(subject, relation, object)` reports which written rule produced a fact
  (derived facts are tracked separately from written ones).

**Writable vocabulary.** `alias the red planet = Mars` makes Eddi understand a phrase
it was never trained on, because resolution happens before the language model is involved.

Scale check: a 120-level hierarchy (121 written facts) closes to **7,260 inferred facts
in 0.33 s**. A 1,000-fact rule base stays in the low milliseconds.

### What training still buys you

Rules give *deduction over what you wrote*; training gives *language*. Eddi needs both,
and they are independent - the table at the top of this section is the whole point.
`--train` is optional: a fresh agent with zero trained tokens already answers facts,
follows rules, resolves aliases, chains, and does analogies.

## Can Eddi build a good-looking web page?

Yes - but not by sampling tokens. Asked to "generate HTML" Eddi's n-gram model produces
`< div class = " title > < html >` - it has no model of tag nesting or CSS. So page
building goes through the **written** path, which is the whole point of the architecture:

```bash
eddi --web "landing page for a coffee shop"        # -> landing.html
eddi --web "dashboard for server metrics"          # -> dashboard.html
eddi --web "page for my dog" --layout landing
```

What it composes (13 components, 13 design tokens, 4 layouts - all written):

| Layer | Contents |
|---|---|
| Design tokens | palette, spacing, radii, shadow, system font stack |
| CSS | responsive, mobile breakpoint, hover/active states, accessible contrast |
| Components | `nav`, `hero`, `card`, `grid_three`, `table`, `section`, `button`, `button_js`, `footer`, `list`, `page` |
| Layouts | `landing`, `dashboard`, `article`, `blank` |
| Content packs | topic-aware copy for coffee, servers, dogs, HDC (add your own) |

Three example pages are in [`examples/pages/`](examples/pages): a coffee-shop landing page,
a server dashboard and an article - each a single self-contained file with no dependencies.

**It is extensible by writing, not training:**

```python
from eddi.html_library import HtmlBuilder
builder = HtmlBuilder()
builder.teach_component("card", "<article class='custom'>{title}</article>")
```

`trained_tokens` stays at 0 throughout. Tests verify the output parses with balanced tags,
carries a viewport meta and stylesheet, and escapes user text (so a title of
`<script>alert(1)</script>` cannot inject a script).

## What makes Eddi smart

| Capability | How it works | Where |
|---|---|---|
| **Writable memory** | Facts `(subject, relation, object)` are written as 1024-bit hypervectors: `bind(S, R, O) = S ⊕ R ⊕ O`. Learning is a write, not a training step. | `eddi/hdc.py` |
| **Writable inference** | `rule <rel1> => <rel2>` forward-chains (including transitive closure) over written facts; derived facts carry provenance and are explained on demand. | `eddi/hdc.py` |
| **Writable vocabulary** | `alias <phrase> = <canonical>` adds synonyms with no retraining. | `eddi/hdc.py` |
| **O(1) retrieval** | A writable hash index `(subject, relation) → objects` answers exactly; a lazily built **majority-vote bundle** plus HDC cleanup recovers objects even from noisy probes. | `eddi/hdc.py` |
| **Trained language model** | Interpolated back-off n-gram model (default: 2-token context, `--order 3`) with **absolute discounting**: interpolation weights depend on evidence, so sparse long contexts cannot overrule solid short ones. Deeper contexts (`--order 4/5`) pay off on larger corpora. | `eddi/language.py` |
| **Generalisation beyond n-grams** | A second learning channel over **word-type classes** (cheap stemming: `runs/running/ran → run`, numbers → `<num>`) lets unseen token contexts still be answered. | `eddi/language.py` |
| **Reasoning router** | Questions are parsed and routed: teach, rule, alias, ask, attribute, about, relate, chain (multi-hop), analogy, math, generate — with honest "I don't know". | `eddi/reasoning.py` |
| **Verbatim memory** | Real text, split into ~180-token chunks and appended to a JSONL file. Answers are retrieved with BM25 plus boosts for exact identifiers and phrases, then accepted only if the chunk covers most of the question's terms - so it quotes what you said instead of guessing. | `eddi/memory.py` |
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
* verbatim memory grows without a fixed budget: chunks are appended to disk and only the
  index lives in RAM;
* `--prune N` drops n-grams seen fewer than N times to keep saved brains small.

Eddi does **not** download anything on its own and never calls a cloud API.

---

## CLI reference

```
eddi                        status: brain, size, C++ core, held-out perplexity
eddi --demo                 capability tour (memory, analogy, chain, math, generation)
eddi --train [--pack P]... [--corpus DIR]... [--epochs N] [--order N] [--prune N]
eddi --bench [--json]       held-out benchmark with miss explanations
eddi --ask "..." [--explain] [--save]
eddi --chat [--save-on-exit]
eddi --ui [--host H] [--port 8000]      browser chat UI (talk to Eddi in a tab)
eddi --serve [--host H] [--port 8765]   JSON API: GET /health /stats, POST /ask
eddi-gui                    Tkinter desktop app (train, chat, ingest, save)
```

`--brain PATH` points at a different brain file (default: `eddi_brain.pkl` in the OS temp dir;
use `--brain ./eddi_brain.pkl` to keep it next to your project).

### Curriculum

Eddi is taught from a registry of topic packs (`eddi/curriculum.py`), each of which is a
list of `(subject, relation, object)` facts plus prose the language model learns from.
Subject packs live in `eddi/curriculum_packs/`, one module per subject:

| Pack | Covers |
|---|---|
| `core` | Eddi's own design, HDC, the memory hierarchy |
| `mathematics` | arithmetic, algebra, geometry, number theory, calculus, probability |
| `science` | physics, astronomy, biology, chemistry |
| `computing` | languages, systems, data structures, protocols |
| `reasoning` | logic, method, statistics |
| `geography` | capitals, rivers, mountains, units |

Every pack is checked before it is taught:

```bash
python3 -m eddi.curriculum_packs.validate
```

The validator tokenises each subject, relation and object with the *real* tokeniser and
rejects any value that splits into more than one token, because such a fact can be stored
but never retrieved. It also rejects duplicate `(subject, relation)` pairs, where the
second fact silently overwrites the first. Both checks have caught real defects:
`19.3_g_per_cm3` split at the decimal point, and `9.81_m_per_s2` did too.

## Capability suite

`python -m eddi.eval_questions` runs **104 questions across 70 distinct
abilities** — lookup, reverse lookup, comparison, counting, analogy, chaining,
arithmetic, percentages, powers, roots, unit conversion, user memory,
derivation, robustness (case/typos/punctuation), honesty, small talk, self
knowledge and teaching.

Every question names the ability it exercises, so a failure points at a
mechanism rather than just "wrong answer". Exit code is 1 if anything fails, so
it can gate CI. `--verbose` shows every answer; `--json` emits rows.

The suite is mutation-checked: deliberately breaking the inverse-fact
direction (104 -> 101) or removing the out-of-vocabulary guard (104 -> 102)
both make it fail, so it is not passing by accident.

### Browser UI

```bash
python -m eddi.cli --ui              # or: python -m eddi.webserver --port 8000
```

Opens a chat page on `http://localhost:8000` (bind `0.0.0.0` so a preview pane or
another device can reach it). No build step, no framework, no CDN — one page
served from `eddi/webserver.py`, styled with the same design tokens as the
generated pages.

What it does:

| | |
|---|---|
| **Chat** | Type or click a suggestion; each answer is tagged with the capability that produced it (`about`, `math`, `ask`, `generate`) and its confidence. |
| **Honest refusals** | If Eddi does not know something it says so and offers a form to teach it, instead of continuing the language model into a confident-sounding sentence. |
| **Teach inline** | When Eddi says it does not know a subject, an inline form writes the fact — and it is saved to the brain immediately, so it survives a restart. |
| **Live status** | Header shows fact count, trained tokens and liveness. |

Endpoints (also usable on their own):

```
GET  /          the chat page
GET  /health    {"status":"ok","version":...}
GET  /stats     fact count, vocab, trained tokens
POST /ask       {"text":"..."} -> {"text","kind","confidence"}
POST /teach     {"subject","relation","object"}   writes one fact
POST /ingest    {"text":"..."}                    learns from a paragraph
```

The server is stateless with respect to conversation: the transcript lives in the
page, so two tabs cannot corrupt each other. Requests are serialised through a
lock because the agent itself is not thread-safe.

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
python -m unittest discover -s tests -v     # 70 tests, cross-platform
```

The suite covers hypervector algebra (bind/unbind, bundles, permutation), fuzzy knowledge
lookup, analogy and multi-hop chaining, **writable rules / aliases / provenance with zero
trained tokens**, language-model normalisation/generalisation/persistence, question routing,
math sandboxing, training + brain round-trip, and Windows-safe paths.

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
        /ask API    │        └──► memory.ChunkMemory           (verbatim text, BM25)    │
                    │ training ──► curriculum / your corpus / evaluation ──► brain.pkl  │
                    └───────────────────────────────────────────────────────────────┘
                                   optional accelerator: src/eddi_core.cpp
```

* **Memory and inference are written, not trained.** One `teach` is one fact; one `rule` is
  one inference rule; neither touches the language model (`trained_tokens` stays 0).
* **Language is trained, not prompted.** No cloud, no API key, no GPU.
* **Answers are traceable.** Every response carries a `kind` (`ask`, `about`, `chain`,
  `analogy`, `math`, `generate`, `unknown`) and optional reasoning details (`--explain`).

## Limitations (honest ones)

* The language model is an n-gram model: it writes fluent short passages from its training
  distribution, but it does not follow complex instructions or understand long documents.
* Fact accuracy is measured on the built-in curriculum's held-out questions; questions far
  outside that vocabulary return `unknown` rather than a guess.
* Verbatim memory retrieves; it does not reason. A recalled sentence is quoted as-is, and
  the model is not asked to combine several recalled passages into a new answer.
* `context.TieredContext1M` is still in the tree but is *not* memory: it records token
  hashes in a ring buffer, stores no text, and nothing in the answer path reads it back.
  It is legacy and scheduled for removal.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
