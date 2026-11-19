"""Tests for Eddi.  All paths are cross-platform: no hard-coded /tmp."""

import os
import pickle
import tempfile
import unittest
from pathlib import Path

from eddi.agent import EddiAgent
from eddi.hdc import HyperVector, WritableKnowledgeBase
from eddi.language import NGramLanguageModel, tokenize
from eddi.paths import core_binary_name, default_context_path, find_core_binary
from eddi.training import TrainingConfig, load_brain, save_brain, train


class TestHyperVectors(unittest.TestCase):
    def test_hdc_bind_unbind(self):
        a = HyperVector.from_symbol("subject")
        b = HyperVector.from_symbol("object")
        bound = a.bind(b)
        recovered = bound.unbind(a)
        self.assertAlmostEqual(recovered.similarity(b), 1.0)

    def test_bind_is_self_inverse(self):
        a = HyperVector.from_symbol("x")
        self.assertEqual(a.bind(a).value, 0)

    def test_bundle_is_majority_vote(self):
        majority = HyperVector.from_symbol("majority")
        minority = HyperVector.from_symbol("minority")
        bundle = HyperVector.bundle([majority] * 7 + [minority] * 3)
        self.assertGreater(bundle.similarity(majority), 0.75)
        self.assertLess(bundle.similarity(minority), 0.75)

    def test_permute_encodes_order(self):
        a = HyperVector.from_symbol("token")
        self.assertGreater(a.similarity(a.permute(0)), 0.99)
        self.assertLess(a.similarity(a.permute(5)), 0.6)

    def test_similarity_is_symmetric_and_bounded(self):
        a = HyperVector.from_symbol("one")
        b = HyperVector.from_symbol("two")
        s = a.similarity(b)
        self.assertEqual(s, b.similarity(a))
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)


class TestWritableKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = WritableKnowledgeBase()
        self.kb.import_facts(
            [
                ("RTX2060Super", "vram", "8GB"),
                ("Earth", "orbits", "Sun"),
                ("Moon", "orbits", "Earth"),
                ("Mars", "orbits", "Sun"),
                ("Berlin", "capital_of", "Germany"),
            ]
        )

    def test_writable_kb(self):
        hits = self.kb.query("RTX2060Super", "vram")
        self.assertTrue(len(hits) >= 1)
        self.assertEqual(hits[0][0], "8GB")

    def test_exact_lookup_is_o1_and_fuzzy(self):
        self.assertEqual(self.kb.query("Earth", "orbits")[0][0], "Sun")
        self.assertEqual(self.kb.query("earth", "orbits")[0][0], "Sun")
        self.assertEqual(self.kb.query("the Earth", "Orbits")[0][0], "Sun")

    def test_facts_reject_duplicates_but_track_recency(self):
        self.kb.write_fact("Earth", "type", "planet")
        self.kb.write_fact("Earth", "type", "planet")
        self.assertEqual(len(self.kb.query("Earth", "type", top_k=10)), 1)

    def test_facts_about(self):
        relations = {rel for rel, _obj, _conf in self.kb.facts_about("Earth")}
        self.assertIn("orbits", relations)

    def test_analogy(self):
        result = self.kb.analogy("Earth", "Sun", "Moon")
        self.assertTrue(result)
        relation, obj, score = result[0]
        self.assertEqual((relation, obj), ("orbits", "Earth"))
        self.assertGreater(score, 0.5)

    def test_chain(self):
        self.assertEqual(self.kb.chain("Moon", ["orbits", "orbits"]), ["Sun"])

    def test_probe_multi_valued_relation(self):
        kb = WritableKnowledgeBase()
        kb.write_fact("Eddi", "knows", "math")
        kb.write_fact("Eddi", "knows", "code")
        kb.write_fact("Eddi", "knows", "physics")
        top = kb.probe("Eddi", "knows", top_k=1)[0][0]
        self.assertIn(top, {"math", "code", "physics"})


class TestWritableInference(unittest.TestCase):
    """Smartness that comes from *writing*, not training."""

    def test_zero_training_still_answers(self):
        a = EddiAgent(persist_context=False)
        self.assertEqual(a.lm.trained_tokens, 0)
        a.teach("Phobos", "orbits", "Mars")
        self.assertEqual(a.answer("what does Phobos orbit").text, "Phobos --orbits--> Mars")
        self.assertEqual(a.lm.trained_tokens, 0, "answering must not require training")

    def test_rule_adds_inference_without_training(self):
        a = EddiAgent(persist_context=False)
        a.answer("rule is_a => warm_blooded")
        a.teach("cat", "is_a", "mammal")
        self.assertEqual(a.lm.trained_tokens, 0)
        self.assertIn("mammal", a.answer("what is the warm_blooded of cat").text)

    def test_transitive_closure_rule(self):
        kb = WritableKnowledgeBase()
        kb.import_facts(
            [
                ("cat", "is_a", "mammal"),
                ("mammal", "is_a", "animal"),
                ("animal", "is_a", "living_thing"),
            ]
        )
        kb.add_rule("is_a", "is_a")
        ancestors = sorted(o for o, _ in kb.query("cat", "is_a", top_k=10))
        self.assertEqual(ancestors, ["animal", "living_thing", "mammal"])

    def test_rules_are_idempotent(self):
        kb = WritableKnowledgeBase()
        kb.import_facts([("cat", "is_a", "mammal"), ("mammal", "is_a", "animal")])
        kb.add_rule("is_a", "is_a")
        total = len(kb)
        self.assertEqual(kb.apply_rules(), 0)
        self.assertEqual(len(kb), total)

    def test_explain_shows_provenance(self):
        kb = WritableKnowledgeBase()
        kb.import_facts([("cat", "is_a", "mammal"), ("mammal", "is_a", "animal")])
        kb.add_rule("is_a", "is_a")
        explanation = kb.explain("cat", "is_a", "animal")
        self.assertIsNotNone(explanation)
        self.assertIn("written rule", explanation)
        self.assertIsNone(kb.explain("cat", "is_a", "mammal"))

    def test_alias_needs_no_training(self):
        a = EddiAgent(persist_context=False)
        a.answer("teach Mars | type | planet")
        a.answer("alias the red planet = Mars")
        self.assertIn("planet", a.answer("what is the type of the red planet").text)
        self.assertEqual(a.lm.trained_tokens, 0)

    def test_written_intelligence_survives_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = os.path.join(tmp, "writes.pkl")
            a = EddiAgent(persist_context=False)
            a.answer("rule is_a => warm_blooded")
            a.teach("whale", "is_a", "mammal")
            a.answer("alias sea giant = whale")
            save_brain(a, brain)
            b = load_brain(brain, persist_context=False)
            try:
                self.assertEqual(b.lm.trained_tokens, 0)
                self.assertIn("mammal", b.answer("what is the warm_blooded of sea giant").text)
                self.assertEqual(b.kb.rules, [("is_a", "warm_blooded")])
            finally:
                b.close()


class TestHtmlComposition(unittest.TestCase):
    """Eddi composes pages from written components - the language model is not used."""

    def test_generated_pages_are_well_formed(self):
        from html.parser import HTMLParser

        from eddi.html_library import HtmlBuilder, spec_from_request

        class Balance(HTMLParser):
            VOID = {"meta", "br", "hr", "img", "input", "link"}

            def __init__(self):
                super().__init__()
                self.stack: list = []
                self.errors: list = []

            def handle_starttag(self, tag, attrs):
                if tag not in self.VOID:
                    self.stack.append(tag)

            def handle_endtag(self, tag):
                if not self.stack:
                    self.errors.append(f"stray </{tag}>")
                elif self.stack[-1] == tag:
                    self.stack.pop()
                else:
                    self.errors.append(f"mismatch {self.stack[-1]}/{tag}")
                    while self.stack and self.stack[-1] != tag:
                        self.stack.pop()
                    if self.stack:
                        self.stack.pop()

        builder = HtmlBuilder()
        for request in (
            "landing page for a coffee shop",
            "dashboard for server metrics",
            "article about hyperdimensional computing",
        ):
            with self.subTest(request=request):
                page = builder.build(spec_from_request(request))
                checker = Balance()
                checker.feed(page)
                self.assertEqual(checker.errors, [])
                self.assertEqual(checker.stack, [])
                for required in ("<!DOCTYPE html>", 'name="viewport"', "<style>", "</html>"):
                    self.assertIn(required, page)

    def test_components_are_writable(self):
        from eddi.html_library import HtmlBuilder, spec_from_request

        builder = HtmlBuilder()
        builder.teach_component("card", "<article class='custom'>{title}</article>")
        page = builder.build(spec_from_request("landing page for coffee"))
        self.assertIn("custom", page)
        self.assertNotIn('class="card"', page)

    def test_agent_routes_page_requests_without_training(self):
        a = EddiAgent(persist_context=False)
        answer = a.answer("make a landing page for a coffee shop")
        self.assertEqual(answer.kind, "webpage")
        self.assertIn("Coffee Shop", answer.text)
        self.assertEqual(a.lm.trained_tokens, 0)

    def test_html_is_escaped(self):
        from eddi.html_library import HtmlBuilder, PageSpec

        builder = HtmlBuilder()
        page = builder.build(PageSpec(title="<script>alert(1)</script>", brand="x", heading="x"))
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)


class TestLanguageModel(unittest.TestCase):
    def test_training_and_distribution_normalised(self):
        lm = NGramLanguageModel(order=3)
        lm.train_text("the cat sat on the mat. the cat sat again.")
        dist = lm.distribution(["the", "cat"])
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)
        self.assertEqual(lm.predict(["the", "cat"], 1)[0][0], "sat")

    def test_word_type_generalises(self):
        lm = NGramLanguageModel(order=3)
        lm.train_text("he runs fast. she runs daily. he runs often.")
        # 'walk' was never seen, but its word type context still predicts 'runs'-style words
        preds = [t for t, _ in lm.predict(["fast"], 6)]
        self.assertTrue(preds)

    def test_fast_generation_path_matches_full_distribution(self):
        lm = NGramLanguageModel(order=3)
        for sentence in ("the cat sat on the mat", "the cat sat again", "a dog sat on a log"):
            lm.train_text(sentence)
        full = sorted(lm.distribution(["the", "cat"]).items(), key=lambda kv: -kv[1])[:3]
        fast = lm.distribution_top(["the", "cat"], limit=64)
        fast_ids = sorted(fast, key=lambda tid: -fast[tid])[:3]
        self.assertEqual([tid for tid, _ in full], fast_ids)
        self.assertAlmostEqual(sum(fast.values()), 1.0, places=6)

    def test_greedy_decoding_is_deterministic(self):
        lm = NGramLanguageModel(order=3)
        for _ in range(3):
            lm.train_text("hello world this is a test of eddi context.")
        a = lm.generate("hello world", max_tokens=6, temperature=0.0)
        b = lm.generate("hello world", max_tokens=6, temperature=0.0)
        self.assertEqual(a, b)

    def test_generation_is_not_a_loop(self):
        lm = NGramLanguageModel(order=3)
        for _ in range(3):
            lm.train_text("a language model predicts the next token from the tokens before it.")
        text = lm.generate("a language model", max_tokens=20, temperature=0.5, seed=1)
        self.assertGreater(len(text.split()), 4)
        self.assertLess(text.count("the the"), 2)

    def test_persistence_roundtrip(self):
        lm = NGramLanguageModel(order=3)
        lm.train_text("eddi learns fast and remembers everything it is taught.")
        clone = NGramLanguageModel.from_payload(lm.to_payload())
        self.assertEqual(clone.vocab, lm.vocab)
        self.assertAlmostEqual(
            lm.perplexity(["eddi learns fast"]), clone.perplexity(["eddi learns fast"])
        )

    def test_detokenize_punctuation(self):
        from eddi.language import detokenize

        self.assertEqual(detokenize(tokenize("Hello, world!")), "Hello, world!")


class TestAgentAndReasoning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.agent = EddiAgent(
            context_path=os.path.join(self.tmp.name, "ctx.mmap"),
            persist_context=True,
        )
        self.agent.teach_many([("Earth", "orbits", "Sun"), ("Moon", "orbits", "Earth")])

    def tearDown(self):
        self.agent.close()
        self.tmp.cleanup()

    def test_ingest_and_teach(self):
        n = self.agent.ingest_text("hello world this is a test of eddi context")
        self.assertEqual(n, 9)
        # greedy decoding (temperature 0) is deterministic, so the assertion is stable
        gen = self.agent.generate("hello world", max_tokens=5, temperature=0.0)
        self.assertTrue(gen.startswith("hello world "))
        self.assertIn("this is a test of", gen)
        self.agent.teach("Eddi", "status", "working")
        ans = self.agent.ask("Eddi", "status")
        self.assertTrue(any("working" in a for a in ans))

    def test_agent_knows_its_own_name(self):
        a = EddiAgent(persist_context=False)
        a.teach("Eddi", "name", "Eddi")
        a.teach("Eddi", "type", "local_agent")
        for question in ("what is your name", "who are you", "tell me about yourself"):
            with self.subTest(question=question):
                answer = a.answer(question)
                self.assertEqual(answer.kind, "identity")
                self.assertIn("Eddi", answer.text)
                self.assertEqual(a.lm.trained_tokens, 0)

    def test_question_routing(self):
        cases = {
            "ask Earth | orbits": "Sun",
            "what does the Earth orbit": "Sun",
            "what is the moon of earth": "--orbits--> Earth",
            "tell me about Moon": "orbits: Earth",
            "Earth is to Sun as Moon is to ?": "--orbits--> Earth",
            "how is Earth related to Sun": "--orbits--> Sun",
            "chain Moon | orbits | orbits": "Sun",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertIn(expected, self.agent.answer(question).text)

    def test_math(self):
        self.assertEqual(self.agent.answer("2 + 2 * 10").text, "22")
        self.assertEqual(self.agent.answer("what is 17 * 3 + 4").text, "55")
        self.assertEqual(self.agent.answer("9 / 4").text, "2.25")
        self.assertEqual(self.agent.answer("1836 + 164").text, "2000")

    def test_math_reports_errors_instead_of_guessing(self):
        self.assertEqual(self.agent.answer("1/0").kind, "math_error")
        self.assertEqual(self.agent.answer("2 ** 10000").kind, "math_error")

    def test_help_for_incomplete_commands(self):
        self.assertEqual(self.agent.answer("tell me about").kind, "help")
        self.assertEqual(self.agent.answer("teach X").kind, "help")

    def test_math_blocks_code_execution(self):
        answer = self.agent.answer("__import__('os').system('echo hacked')")
        self.assertNotEqual(answer.kind, "math")

    def test_honest_unknown(self):
        answer = self.agent.answer("what is the favourite colour of the Moon")
        self.assertIn(answer.kind, {"unknown", "generate", "about"})
        self.assertNotIn("--orbits--> Sun", answer.text)

    def test_context_tokens_are_counted(self):
        self.agent.ingest_text("one two three")
        self.assertEqual(self.agent.ctx.count, 3)


class TestTraining(unittest.TestCase):
    def test_train_and_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = os.path.join(tmp, "brain.pkl")
            agent = EddiAgent(persist_context=False, context_path=os.path.join(tmp, "c.mmap"))
            try:
                report = train(agent, TrainingConfig(epochs=2, save_path=brain, seed=1))
                self.assertGreater(report.facts, 50)
                self.assertGreater(report.lm_vocab, 100)
                self.assertEqual(report.fact_accuracy, 1.0)
                self.assertGreaterEqual(report.reasoning_accuracy or 0, 0.5)
                self.assertTrue(Path(brain).exists())
            finally:
                agent.close()
            loaded = load_brain(brain, context_path=os.path.join(tmp, "c2.mmap"))
            try:
                self.assertEqual(len(loaded.kb), report.facts)
                self.assertIn("Sun", loaded.answer("what does the Earth orbit").text)
                self.assertTrue(loaded.generate("The compiler", max_tokens=10))
            finally:
                loaded.close()

    def test_brain_format_rejects_garbage(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.pkl")
            with open(bad, "wb") as f:
                pickle.dump({"nope": 1}, f)
            with self.assertRaises(ValueError):
                load_brain(bad)


class TestMemoryIsActuallyPersistent(unittest.TestCase):
    """Everything learned must land somewhere durable - not in a temp file."""

    def test_default_brain_lives_in_user_data_dir(self):
        from eddi.paths import default_brain_path, temp_dir

        path = default_brain_path()
        self.assertNotEqual(Path(path).parent, Path(temp_dir()))

    def test_everything_learned_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = os.path.join(tmp, "all.pkl")
            a = EddiAgent(persist_context=False)
            a.teach("kestrel", "is_a", "bird")
            a.teach_rule("is_a", "warm_blooded")
            a.teach_alias("the little hawk", "kestrel")
            a.ingest_text("The kestrel hovers over the field.")
            before = {
                "facts": len(a.kb),
                "derived": len(a.kb.derived),
                "rules": list(a.kb.rules),
                "aliases": len(a.kb.user_aliases),
                "tokens": a.lm.trained_tokens,
            }
            a.save(brain)
            a.close(save=False)

            b = load_brain(brain, persist_context=False)
            try:
                self.assertEqual(b.kb.rules, before["rules"])
                self.assertEqual(len(b.kb.user_aliases), before["aliases"])
                self.assertEqual(b.lm.trained_tokens, before["tokens"])
                self.assertEqual(len(b.kb.derived), before["derived"])
                self.assertIn("bird", b.answer("what is the warm_blooded of the little hawk").text)
            finally:
                b.close(save=False)

    def test_close_survives_when_autosave_is_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = os.path.join(tmp, "auto.pkl")
            a = EddiAgent(persist_context=False, brain_path=brain, autosave=True)
            a.teach("otter", "is_a", "mammal")
            a.close()
            self.assertTrue(Path(brain).exists())
            b = load_brain(brain, persist_context=False)
            try:
                self.assertEqual(len(b.kb), 1)
            finally:
                b.close(save=False)

    def test_context_survives_close_and_reopen(self):
        from eddi.context import TieredContext1M

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.mmap")
            store = TieredContext1M(path, max_tokens=128)
            for i in range(7):
                store.append(i, i * 3)
            store.close()
            self.assertTrue(Path(path).exists(), "context file must not be deleted on close")

            reopened = TieredContext1M(path, max_tokens=128)
            try:
                self.assertEqual(reopened.count, 7)
                reopened.append(99, 9)
                self.assertEqual(reopened.count, 8)
            finally:
                reopened.close()


class TestCrossPlatform(unittest.TestCase):
    def test_paths_are_cross_platform(self):
        path = default_context_path("unit_test")
        self.assertFalse(path.startswith("/tmp") and os.name == "nt")
        self.assertTrue(os.path.isabs(path))
        try:
            os.remove(path)
        except OSError:
            pass

    def test_context_handles_windows_size_access(self):
        from eddi.context import TieredContext1M

        with tempfile.TemporaryDirectory() as tmp:
            store = TieredContext1M(os.path.join(tmp, "ctx.mmap"), max_tokens=64)
            try:
                for i in range(10):
                    store.append(i, i * 7)
                store.flush()
                self.assertEqual(store.count, 10)
                self.assertEqual(store.read(3), (3, 21))
                hits = store.query_lsh(21, limit=4)
                self.assertTrue(all(hit[1] == 3 for hit in hits))
            finally:
                store.close()

    def test_core_binary_name(self):
        self.assertTrue(core_binary_name().endswith(("eddi_core", "eddi_core.exe")))
        self.assertIsNone(find_core_binary(Path(tempfile.gettempdir())))


if __name__ == "__main__":
    unittest.main()

class TestLanguageModelOrder(unittest.TestCase):
    """Deeper context is not cosmetic: a shallow model splices false facts."""

    @staticmethod
    def _agent(order: int):
        from eddi.curriculum import PACKS
        import tempfile

        agent = EddiAgent(
            lm_order=order,
            persist_context=False,
            memory_path=os.path.join(tempfile.mkdtemp(), "m.jsonl"),
            record_turns=False,
        )
        for pack in PACKS.values():
            for sentence in pack.get("prose", []):
                agent.train_corpus([sentence])
        return agent

    def test_order_four_does_not_splice_a_false_boiling_point(self):
        # With order 3 the model produced "water boils at one hundred eighty
        # degrees" by joining the water sentence to the triangle sentence. The
        # number is wrong, which is worse than saying nothing.
        agent = self._agent(order=4)
        text = agent.generate("water freezes", max_tokens=16, temperature=0.0)
        self.assertIn("one hundred degrees", text)
        self.assertNotIn("eighty", text)
        agent.close(save=False)

    def test_order_four_completes_a_catalyst_sentence(self):
        agent = self._agent(order=4)
        text = agent.generate("a catalyst", max_tokens=16, temperature=0.0)
        self.assertNotIn("largest city", text)
        agent.close(save=False)

    def test_suggest_order_scales_with_corpus(self):
        from eddi.training import suggest_order

        self.assertEqual(suggest_order(500), 3)
        self.assertEqual(suggest_order(2_500), 4)
        self.assertEqual(suggest_order(25_000), 5)
        self.assertEqual(suggest_order(250_000), 6)

class TestBrainCarriesCorpus(unittest.TestCase):
    """A saved brain must keep the text it learned from."""

    def test_trained_sentences_survive_a_save_and_load(self):
        import tempfile
        from eddi.training import load_brain, save_brain
        from eddi.curriculum import PACKS

        with tempfile.TemporaryDirectory() as tmp:
            brain = os.path.join(tmp, "b.pkl")
            agent = EddiAgent(lm_order=3, persist_context=False,
                              memory_path=os.path.join(tmp, "m.jsonl"))
            for sentence in PACKS["science"]["prose"]:
                agent.train_corpus([sentence])
            save_brain(agent, brain)
            agent.close(save=False)

            loaded = load_brain(brain)
            try:
                # Without this, a loaded agent cannot quote its training text and
                # the generation metric has nothing to score against, which is
                # how completion silently fell to 80% while the in-memory agent
                # scored 90%.
                self.assertEqual(
                    len(loaded.lm.trained_sentences), len(PACKS["science"]["prose"])
                )
            finally:
                loaded.close(save=False)

class TestStemming(unittest.TestCase):
    """Stems must join inflections without inventing shorter non-words."""

    def test_inflections_share_a_stem(self):
        from eddi.memory import _stem
        pairs = [
            ("catalyst", "catalysts"), ("reaction", "reactions"),
            ("freeze", "freezes"), ("string", "strings"), ("seed", "seeds"),
            ("stop", "stopped"), ("plan", "planned"), ("process", "processes"),
            ("study", "studies"), ("carry", "carried"), ("run", "running"),
            ("size", "sizes"), ("hundred", "hundreds"), ("need", "needs"),
            ("use", "used"), ("theory", "theories"), ("proton", "protons"),
        ]
        for singular, plural in pairs:
            self.assertEqual(
                _stem(singular), _stem(plural),
                f"{singular}/{plural} should share a stem",
            )

    def test_short_words_are_not_mangled(self):
        from eddi.memory import _stem
        # "speed" once stemmed to "spe", which matched nothing while displacing
        # real matches: a question about catalysts lost to one about light.
        for word in ("speed", "seed", "need", "bed", "red", "string", "hundred"):
            self.assertGreaterEqual(len(_stem(word)), len(word) - 1, word)
        self.assertEqual(_stem("speed"), "speed")
        self.assertNotEqual(_stem("speed"), _stem("spe"))

class TestRetrievalPrefersTheQuestion(unittest.TestCase):
    """The answer should be the chunk that addresses what was asked."""

    def _agent(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        agent = EddiAgent(lm_order=3, persist_context=False,
                          memory_path=os.path.join(tmp, "m.jsonl"))
        for s in (
            "Photosynthesis turns light, water and carbon dioxide into glucose and oxygen.",
            "Taking a logarithm answers the question of which exponent produces a given number.",
            "A catalyst lowers activation energy and is not consumed by the reaction.",
            "Enzymes are proteins that speed up reactions without being consumed.",
        ):
            agent.train_corpus([s])
        return agent

    def test_rare_subject_beats_common_verb(self):
        agent = self._agent()
        try:
            reply = agent.answer("what does photosynthesis take in")
            self.assertIn("Photosynthesis", reply.text)
        finally:
            agent.close(save=False)

    def test_the_specific_chunk_wins(self):
        agent = self._agent()
        try:
            reply = agent.answer("why does a catalyst speed up a reaction")
            self.assertIn("catalyst", reply.text.lower())
            self.assertNotIn("logarithm", reply.text)
        finally:
            agent.close(save=False)

class TestTrainingPersistsItsText(unittest.TestCase):
    """Training must leave behind the sentences it learned from."""

    def test_a_trained_brain_can_quote_its_corpus(self):
        import tempfile
        from eddi.training import train, TrainingConfig, load_brain

        with tempfile.TemporaryDirectory() as tmp:
            brain = os.path.join(tmp, "b.pkl")
            memory = os.path.join(tmp, "m.jsonl")
            agent = EddiAgent(lm_order=3, persist_context=False,
                              memory_path=memory)
            config = TrainingConfig(packs=("core",), epochs=1, order=3,
                                    evaluate=False, save_path=brain)
            train(agent, config)
            agent.close(save=False)

            # The memory file is the only place the sentences live outside the
            # process. Without it, a trained brain can generate a sentence but
            # cannot quote one -- and every question about the corpus falls
            # through to a guess.
            loaded = load_brain(brain)
            try:
                self.assertGreater(len(loaded.memory.chunks), 0)
                self.assertGreater(len(loaded.lm.trained_sentences), 0)
            finally:
                loaded.close(save=False)

class TestInterrogativeForm(unittest.TestCase):
    """'Which planet does the Moon orbit' must reach the fact base."""

    def test_which_category_question_resolves(self):
        import tempfile
        agent = EddiAgent(lm_order=2, persist_context=False,
                          memory_path=os.path.join(tempfile.mkdtemp(), "m.jsonl"))
        try:
            agent.teach("Moon", "orbits", "Earth")
            agent.teach("Moon", "distance_from_earth_km", "384400")
            reply = agent.answer("which planet does the Moon orbit")
            # The relation that answers the question is an action, not the
            # numerically-valued property that happens to sort first.
            self.assertIn("Earth", reply.text)
            self.assertNotIn("384400", reply.text)
        finally:
            agent.close(save=False)
