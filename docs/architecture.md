# Can Eddi reason?

An answer, and the measurements behind it.

The question was: can a system reach frontier-quality engineering ability on a
consumer laptop, fast, without a frontier model. The research says the question
is mis-framed, and the experiments here say what the right frame is.

---

## 1. What was measured

Four experiments, all reproducible in this repository, none of them requiring a
GPU or a network.

| Experiment | Command | Result |
|---|---|---|
| Engineering loop, hidden tests | `python3 -m eddi.swe bench` | **7/7** |
| Same, loop only, no proposer | `python3 -m eddi.swe bench --ablate` | **2/7** |
| Size of the decision the model faces | `python3 tools/measure_decision_space.py` | **162 candidates** |
| Bugs with no structural signature | `python3 tools/coverage_boundary.py` | **0/2 solved** |
| Same, with a model in the loop | `python3 tools/hybrid_boundary.py` | **2/2 solved** |
| Same, with a *wrong* model | `python3 tools/hybrid_boundary.py` | **2/2 safely refused** |

The benchmark hides the graded tests, as SWE-bench does. A task counts only if
the hidden test passes; the loop's own opinion is not evidence.

---

## 2. What the numbers mean

### Competence is mostly not in the weights

The loop alone resolves 2 of 7. Adding a proposer that can only recognise six
structural patterns resolves 7 of 7. Fourteen hundred lines of loop honour the
findings from mini-swe-agent: explore, reproduce, localise, read, edit, verify,
re-localise. That part is deterministic and contains no model at all.

This is the load-bearing result. It means the difference between a 2B model and
a frontier model is not spent on planning or tool use -- those are code. It is
spent on a much smaller thing.

### That smaller thing is small

The decision the proposer makes, instrumented directly:

- 25 functions and 84 editable lines across all seven tasks
- six recognisable bug families
- worst case **27 lines × 6 strategies = 162 candidates**

A 162-way choice over a narrow, structured input is not a frontier-model task.
It is the kind of classification a small model can be trained to do well, and
the loop can verify its answer in a second.

### The gap is real, adjacent, and closable

Two bugs were written that are one step outside the strategy set: two function
bodies transposed, and a precondition that does not exist yet. Both are ordinary
bugs a person fixes without thinking.

Without a model: **0/2.** With a model in the loop: **2/2.** With a model that
answers confidently and wrongly: **0/2, and nothing breaks.**

That last row is the one that makes the architecture safe. Every proposed line
is validated against the bytes on disk before it is applied. A wrong answer
costs microseconds. The model is a participant, not an authority.

---

## 3. The verdict on the two hypotheses

### "Maybe it doesn't need neurons"

Tested honestly. It does not hold for language, and the reason is specific.

The vector-symbolic memory in `eddi/swe/vsa.py` was built and its capacity law
derived and confirmed: bundling *k* items into a *D*-bit vector stays separable
while `k < D·8/(π·σ²)`, i.e. about **14 bits per association**. Measured, with
the exact index bypassed so only algebra remained:

```
D=2048:  k=64 100% | k=128 69% | k=256 23% | k=512  0%
D=8192:  k=64 100% | k=128 100% | k=256 100% | k=512 46%
```

100 MB of D=10,000 vectors holds roughly **58 million exact associations** --
genuinely useful, and it is now part of the system. But it has **no
generalisation**: an unseen input returns nothing. Hyperdimensional computing
does well on compositional and analogical structure and poorly on raw
perception; the literature puts it near 95% on MNIST and 45% on CIFAR-10, and
proves that binary sparse codes cannot approximate an arbitrary similarity
matrix, so no Bayes-optimal classifier exists in that form.

So: symbols are excellent for *structure*, which is why they carry the loop and
the memory. They cannot replace the thing that reads an issue and knows what a
sentence means. The honest conclusion is not "no neurons" but "**few neurons,
in one place**".

### "The light signals in the machine are the model"

Reframed usefully. Fast local inference is real and measured elsewhere: ternary
weights decode in 29 ms for 2B parameters at 0.4 GB, 2.4-6.2× faster than
llama.cpp given the right kernels. A 0.9B model already reaches 44.6 on
Terminal-Bench 2.1. Frontier SWE capability is 76.8.

The gap between 44.6 and 76.8 is not closed by cheaper arithmetic. It is closed
by not asking the small model to do the part that is free to do in code. That
is the entire design.

---

## 4. What to build, in order

Each step is measurable, and each is worth doing even if the next one fails.

**Step 1 -- done.** The loop and the seven-task benchmark. `2/7 → 7/7`,
ablation recorded.

**Step 2 -- done.** `LLMProposer`, a narrow interface with strict validation.
Proven capable of carrying a correct judgement (`2/2`) and surviving a wrong
one (`0/2`, no damage).

**Step 3 -- next.** Train a small model to fill that interface. The training
data is generated, not annotated: take real repositories, inject a bug, record
the diff, and keep only the cases the loop cannot solve alone. That filter is
what makes the corpus small enough to train on a laptop and targeted exactly at
the 162-way decision.

Acceptance: `hybrid_boundary` and `bench` both stay green with the trained model
in place, at zero network calls.

**Step 4.** Scale the benchmark. Seven tasks proves the mechanism; it does not
prove SWE-bench. The same harness points at real issues once there are enough
tasks to be honest about a percentage.

**Step 5.** Grow the loop's strategy set from failures. Every bug family that
turns out to be structurally recognisable after the fact moves out of the model
and into code, which makes the system faster and the model's job smaller. This
is the flywheel: the model's job should shrink over time, not grow.

---

## 5. What this does not claim

- **7/7 is seven tasks.** It is a mechanism demonstration, not a SWE-bench
  score. Anyone quoting it as one is misreading it.
- **The boundary tasks use a stand-in, not a neural model.** It proves the
  interface, the validation and the loop can carry a model's judgement end to
  end. It does not prove any particular small model is good enough. That is
  step 3, and it is not done.
- **The VSA memory does not reason.** It stores and retrieves structure. The
  reasoning is the loop.
- **"Blazingly fast" is measured for the deterministic parts and projected for
  the model.** The loop runs in milliseconds. The model, when used, is called
  once per unsolved decision, not once per step.

---

## 6. The short version

Frontier engineering ability on low-end hardware is not achieved by making a
small model think like a big one. It is achieved by arranging for the small
model to be asked a small question.

Measured here: **2/7 without judgement, 7/7 with it, and the judgement is a
162-way choice over 84 lines of code.** The model is called only when structure
runs out, its answer is checked before it is trusted, and a wrong answer is free.

That is a design a laptop can run, and it is the only one in this research that
got better the more it was measured.
