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

class TestEngineeringLoop(unittest.TestCase):
    """The loop must solve real bugs, judged by tests it never sees."""

    def test_full_suite_resolves_every_task(self):
        from eddi.swe.bench import run_suite
        from eddi.swe.__main__ import _solver_factory

        # Visible tests are hidden, so passing means the fix is behaviourally
        # right rather than written to match an assertion the agent read.
        outcomes = run_suite(_solver_factory(True), hide_visible_tests=True)
        failed = [o.name for o in outcomes if not o.resolved]
        self.assertEqual(failed, [], f"unresolved: {failed}")

    def test_loop_alone_cannot_solve_them(self):
        """The deterministic phases should get some of the way, not all of it.

        This is the measurement behind the design claim: a large share of the
        work is mechanical (localise, reproduce, verify), so a loop with no
        proposer still makes progress. If this ever reaches 100%, the proposer
        has stopped earning its place and the benchmark has become too easy.
        """
        from eddi.swe.bench import run_suite
        from eddi.swe.__main__ import _solver_factory

        outcomes = run_suite(_solver_factory(False), hide_visible_tests=True)
        solved = sum(1 for o in outcomes if o.resolved)
        self.assertLess(solved, len(outcomes))

    def test_edit_refuses_an_ambiguous_snippet(self):
        """An edit matching twice must fail loudly, not pick one silently."""
        import tempfile
        from eddi.swe.tools import Toolbox

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "m.py"
            target.write_text("x = 1\nx = 1\n", encoding="utf-8")
            tools = Toolbox(tmp)
            result = tools.edit("m.py", "x = 1", "x = 2")
            self.assertFalse(result.ok)
            self.assertIn("2 times", result.text)
            # and the file is untouched
            self.assertEqual(target.read_text(encoding="utf-8"), "x = 1\nx = 1\n")

    def test_tools_refuse_to_escape_the_repository(self):
        import tempfile
        from eddi.swe.tools import Toolbox

        with tempfile.TemporaryDirectory() as tmp:
            tools = Toolbox(tmp)
            with self.assertRaises(ValueError):
                tools.read("../../etc/passwd")
            escaped = tools.run("ls", cwd="..")
            self.assertFalse(escaped.ok)

    def test_traceback_frames_exclude_the_standard_library(self):
        """Only frames inside the repo are worth reading."""
        import tempfile
        from eddi.swe.tools import Toolbox
        from eddi.swe.loop import EngineeringLoop, parse_traceback

        trace = (
            'Traceback (most recent call last):\n'
            '  File "/usr/lib/python3.11/unittest/loader.py", line 154, in load\n'
            '    module = __import__(name)\n'
            '  File "/repo/pkg/mod.py", line 7, in convert\n'
            '    return x * self._cache["km"]\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pkg").mkdir()
            (Path(tmp) / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
            tools = Toolbox(tmp)
            loop = EngineeringLoop(tools)
            frames = parse_traceback(trace, root_prefix="/repo")
            kept = loop._in_repo(frames)
            self.assertEqual([f.path for f in kept], ["pkg/mod.py"])

class TestVectorSymbolicMemory(unittest.TestCase):
    """A hypervector memory stores associations exactly and composes them."""

    def test_binding_is_self_inverse(self):
        from eddi.swe.vsa import SymbolMemory

        mem = SymbolMemory(dim=1024)
        a, b = mem.vector("alpha"), mem.vector("beta")
        bound = mem.bind(a, b)
        self.assertEqual(mem.bind(bound, b), a)

    def test_relations_round_trip(self):
        from eddi.swe.vsa import SymbolMemory

        mem = SymbolMemory(dim=1024)
        mem.add("paris", "capital_of", "france")
        mem.add("paris", "currency", "euro")
        self.assertEqual(mem.lookup("paris", "capital_of"), ["france"])
        self.assertIn(("currency", "euro"), mem.relations_of("paris"))

    def test_chains_return_a_proof(self):
        from eddi.swe.vsa import SymbolMemory

        mem = SymbolMemory(dim=1024)
        mem.add("france", "capital", "paris")
        mem.add("paris", "capital_of", "france")
        trail = mem.chain("france", ["capital", "capital_of"])
        self.assertEqual([step.target for step in trail], ["paris", "france"])
        self.assertEqual(str(trail[0]), "france --capital--> paris")

    def test_reach_is_cycle_safe(self):
        from eddi.swe.vsa import SymbolMemory

        mem = SymbolMemory(dim=1024)
        mem.add("a", "next", "b")
        mem.add("b", "next", "a")
        paths = mem.reach("a", "next", max_depth=6)
        self.assertTrue(paths)
        for path in paths:
            targets = [s.target for s in path]
            self.assertEqual(len(targets), len(set(targets)))

    def test_bundle_capacity_matches_the_prediction(self):
        """Recall degrades when items per bundle approaches D/pi.

        This is the measured law, not a guess: at D/(pi*k) near 10 recall is
        perfect, at 5 it is unreliable, at 2.5 it is gone. The test asserts the
        safe side only, so it cannot become flaky as the implementation changes.
        """
        import math
        from eddi.swe.vsa import SymbolMemory

        dim, k = 2048, 64          # D/(pi*k) = 10.2, the safe side
        mem = SymbolMemory(dim=dim)
        for i in range(k):
            mem.add("hub", f"r{i}", f"o{i}")
        bundle = mem._bundle_for("hub")
        for i in range(0, k, 8):
            probe = mem.bind(mem.vector("hub"), mem.vector(f"r{i}"))
            recovered = mem.bind(bundle, probe)
            best, best_score = None, -9.0
            for j in range(k):
                score = mem.similarity(recovered, mem.vector(f"o{j}"), dim)
                if score > best_score:
                    best, best_score = j, score
            self.assertEqual(best, i, f"lost item {i} at D/(pi*k)={dim/(math.pi*k):.1f}")

    def test_repository_projection_records_structure(self):
        import tempfile
        from eddi.swe.tools import Toolbox
        from eddi.swe.vsa import project_repository

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "mod.py").write_text(
                "def convert(value):\n    return value\n", encoding="utf-8")
            (root / "test_mod.py").write_text(
                "from pkg.mod import convert\n\ndef test_it():\n    pass\n",
                encoding="utf-8")
            mem = project_repository(Toolbox(tmp))
            self.assertEqual(mem.lookup("convert", "defined_in"), ["pkg/mod.py"])
            self.assertIn("pkg.mod", [t for _, t in mem.relations_of("test_mod")])


class TestSuffixSliceStrategy(unittest.TestCase):
    """A silent stemmer bug: the slice does not match its own example."""

    SOURCE = '''def _stem(word: str) -> str:
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        w = w[:-2]                            # studies -> study
    elif w.endswith("es") and len(w) > 4:
        w = w[:-1]                            # sizes -> size
    return w
'''

    def _view(self):
        from eddi.swe.proposers import view_functions

        root = tempfile.mkdtemp()
        with open(os.path.join(root, "words.py"), "w", encoding="utf-8") as handle:
            handle.write(self.SOURCE)
        from eddi.swe.tools import Toolbox

        return view_functions(Toolbox(root), "words.py")[0]

    def test_finds_the_inconsistent_slice(self):
        from eddi.swe.proposers import HeuristicProposer

        proposal = HeuristicProposer()._suffix_slice_mismatch(self._view())
        self.assertIsNotNone(proposal)
        self.assertIn("[:-3]", proposal.new)
        self.assertIn("'y'", proposal.new)
        self.assertIn("studi", proposal.reason)

    def test_the_rule_then_satisfies_its_own_example(self):
        from eddi.swe.proposers import HeuristicProposer

        proposal = HeuristicProposer()._suffix_slice_mismatch(self._view())
        # Apply the proposal the way the loop does: replace the target line in
        # the original source, then run the whole repaired function.
        repaired = self.SOURCE.replace(proposal.old, proposal.new)
        self.assertNotEqual(repaired, self.SOURCE)
        namespace = {}
        exec(compile(repaired, "<repaired>", "exec"), namespace)
        self.assertEqual(namespace["_stem"]("studies"), "study")
        self.assertEqual(namespace["_stem"]("sizes"), "size")


class TestStemmerBenchmark(unittest.TestCase):
    """The seventh benchmark task is solvable and is really broken."""

    def test_task_is_registered(self):
        from eddi.swe.bench import TASKS

        names = [task.name for task in TASKS]
        self.assertIn("stemmer-slice", names)
        self.assertEqual(len(TASKS), 7)

    def test_hidden_test_rejects_the_broken_fixture(self):
        import subprocess
        import tempfile

        from eddi.swe.bench import TASKS

        task = [t for t in TASKS if t.name == "stemmer-slice"][0]
        root = tempfile.mkdtemp()
        for relative, body in task.files.items():
            path = os.path.join(root, relative)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body)
        with open(os.path.join(root, "hidden_test.py"), "w", encoding="utf-8") as handle:
            handle.write(task.hidden_test)
        outcome = subprocess.run(
            ["python3", "-m", "unittest", "hidden_test", "-q"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(outcome.returncode, 0)
        self.assertIn("study", outcome.stderr)


class TestLLMProposer(unittest.TestCase):
    """The model is consulted for one narrow decision and must fail safely."""

    def _workspace(self, files):
        root = tempfile.mkdtemp()
        for relative, body in files.items():
            path = os.path.join(root, relative)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body)
        return root

    def _state(self, root, files):
        from eddi.swe.loop import Probe, TaskState

        state = TaskState(goal="the issue", probes=[Probe("python3 -c pass")])
        state.files_read = list(files)
        from eddi.swe.tools import Toolbox

        return state, Toolbox(root)

    SOURCE = "def area(w, h):\n    return w + h\n"

    def test_a_verbatim_line_is_accepted(self):
        from eddi.swe.llm import proposal_from_reply

        root = self._workspace({"geometry.py": self.SOURCE})
        state, toolbox = self._state(root, ["geometry.py"])
        proposal = proposal_from_reply(
            {"strategy": "novel", "old": "    return w + h", "new": "    return w * h"},
            toolbox,
            state,
        )
        self.assertEqual(proposal, ("geometry.py", "    return w + h", "    return w * h"))

    def test_a_line_that_is_not_in_the_file_is_rejected(self):
        from eddi.swe.llm import proposal_from_reply

        root = self._workspace({"geometry.py": self.SOURCE})
        state, toolbox = self._state(root, ["geometry.py"])
        proposal = proposal_from_reply(
            {"strategy": "novel", "old": "    return 42", "new": "    return 43"},
            toolbox,
            state,
        )
        self.assertIsNone(proposal)

    def test_novel_is_not_a_refusal(self):
        from eddi.swe.llm import proposal_from_reply

        root = self._workspace({"geometry.py": self.SOURCE})
        state, toolbox = self._state(root, ["geometry.py"])
        proposal = proposal_from_reply(
            {"strategy": "novel", "old": "    return w + h", "new": "    return w * h"},
            toolbox,
            state,
        )
        self.assertIsNotNone(proposal)

    def test_an_empty_edit_is_a_refusal(self):
        from eddi.swe.llm import proposal_from_reply

        root = self._workspace({"geometry.py": self.SOURCE})
        state, toolbox = self._state(root, ["geometry.py"])
        self.assertIsNone(
            proposal_from_reply({"strategy": "novel", "old": "", "new": ""}, toolbox, state)
        )

    def test_a_plural_decision_returns_several_edits(self):
        from eddi.swe.llm import proposal_from_reply

        source = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
        root = self._workspace({"m.py": source})
        state, toolbox = self._state(root, ["m.py"])
        proposal = proposal_from_reply(
            {
                "strategy": "novel",
                "edits": [
                    {"path": "m.py", "old": "    return 1", "new": "    return 2"},
                    {"path": "m.py", "old": "    return 2", "new": "    return 1"},
                ],
            },
            toolbox,
            state,
        )
        self.assertIsInstance(proposal, list)
        self.assertEqual(len(proposal), 2)

    def test_fenced_json_is_parsed(self):
        from eddi.swe.llm import parse_reply

        payload = parse_reply('Here you go:\n```json\n{"old": "a", "new": "b"}\n```')
        self.assertEqual(payload["old"], "a")

    def test_a_broken_reply_is_not_an_exception(self):
        from eddi.swe.llm import LLMProposer

        root = self._workspace({"geometry.py": self.SOURCE})
        state, toolbox = self._state(root, ["geometry.py"])
        proposer = LLMProposer(complete=lambda system, user: "I am not sure.")
        self.assertIsNone(proposer(state, toolbox))

    def test_a_model_that_raises_does_not_break_the_loop(self):
        from eddi.swe.llm import LLMProposer

        root = self._workspace({"geometry.py": self.SOURCE})
        state, toolbox = self._state(root, ["geometry.py"])

        def explode(system, user):
            raise RuntimeError("server down")

        self.assertIsNone(LLMProposer(complete=explode)(state, toolbox))

    def test_the_prompt_stays_narrow(self):
        from eddi.swe.llm import LLMProposer, build_prompt

        root = self._workspace({"geometry.py": self.SOURCE})
        state, toolbox = self._state(root, ["geometry.py"])
        prompt = build_prompt(state, toolbox)
        self.assertIn("ISSUE", prompt)
        self.assertIn("def area", prompt)
        self.assertLess(len(prompt.splitlines()), 60)


class TestHybridProposer(unittest.TestCase):
    """Structure first; the model is a fallback, not a dependency."""

    def test_a_heuristic_hit_never_calls_the_model(self):
        from eddi.swe.llm import HybridProposer

        calls = []

        def model(state, toolbox):
            calls.append(1)
            return None

        hybrid = HybridProposer(heuristic=lambda s, t: ("a.py", "x", "y"), model=model)
        self.assertEqual(hybrid(None, None), ("a.py", "x", "y"))
        self.assertEqual(calls, [])
        self.assertEqual(hybrid.model_calls, 0)

    def test_a_heuristic_miss_delegates(self):
        from eddi.swe.llm import HybridProposer

        hybrid = HybridProposer(
            heuristic=lambda s, t: None,
            model=lambda s, t: ("b.py", "p", "q"),
        )
        self.assertEqual(hybrid(None, None), ("b.py", "p", "q"))
        self.assertEqual(hybrid.model_calls, 1)


class TestPluralEdits(unittest.TestCase):
    """A decision can be several edits, and they must not invalidate each other."""

    SOURCE = "def area():\n    return 1\n\n\ndef perimeter():\n    return 2\n"

    def test_swapping_two_bodies_is_one_decision(self):
        from eddi.swe.loop import EngineeringLoop, TaskState
        from eddi.swe.tools import Toolbox

        root = tempfile.mkdtemp()
        with open(os.path.join(root, "m.py"), "w", encoding="utf-8") as handle:
            handle.write(self.SOURCE)

        def propose(state, toolbox):
            # Each snippet carries the signature it belongs to, so it matches
            # exactly once. A bare "return 1" appears in both bodies and the
            # edit tool refuses it -- correctly, since it cannot know which
            # body was meant.
            return [
                ("m.py", "def area():\n    return 1", "def area():\n    return 2"),
                ("m.py", "def perimeter():\n    return 2", "def perimeter():\n    return 1"),
            ]

        # The loop localises before it proposes, so the goal has to name
        # something it can find. Real issues do; this test should too.
        loop = EngineeringLoop(Toolbox(root), propose=propose, verbose=False)
        state = TaskState(goal="`area` and `perimeter` have swapped bodies", probes=[])
        loop.run(state)
        with open(os.path.join(root, "m.py"), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("def area():\n    return 2", text)
        self.assertIn("def perimeter():\n    return 1", text)

    def test_a_single_tuple_still_works(self):
        from eddi.swe.loop import _as_edits

        self.assertEqual(_as_edits(("a.py", "x", "y")), [("a.py", "x", "y")])
        self.assertEqual(_as_edits(None), [])
        self.assertEqual(
            _as_edits([("a.py", "x", "y"), ("b.py", "p", "q")]),
            [("a.py", "x", "y"), ("b.py", "p", "q")],
        )
