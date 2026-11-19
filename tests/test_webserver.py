"""Tests for the browser chat UI (:mod:`eddi.webserver`).

The server is small but it is the only way a non-CLI user reaches Eddi, so the
parts worth testing are: the API contract, persistence of taught facts, thread
safety, and the honesty of the answers it serves.
"""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from eddi.agent import EddiAgent
from eddi.webserver import PAGE, EddiHandler, EddiServer, _tokens


def _start(agent, brain_path=None, save_on_change=False):
    """Run a server on an ephemeral port and return (server, thread, base)."""
    server = EddiServer(
        ("127.0.0.1", 0), EddiHandler, agent, brain_path, save_on_change=save_on_change
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, thread, f"http://{host}:{port}"


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as res:
        return res.status, json.loads(res.read())


def _post(base, path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestPage(unittest.TestCase):
    """The page is generated, so it can be checked without a browser."""

    def setUp(self):
        self.page = PAGE.replace("__TOKENS__", _tokens()).replace("__VERSION__", "0.0.0")

    def test_no_unsubstituted_placeholders(self):
        self.assertNotIn("__TOKENS__", self.page)
        self.assertNotIn("__VERSION__", self.page)

    def test_every_css_variable_resolves(self):
        # A token named accent_ink emitted as --accent_ink does not satisfy
        # var(--accent-ink): the variable silently fails and text loses its
        # colour. This caught exactly that bug.
        import re

        defined = set(re.findall(r"^\s*(--[a-z-]+):", self.page, re.M))
        used = set(re.findall(r"var\((--[a-z-]+)\)", self.page))
        self.assertEqual(sorted(used - defined), [], "CSS variables used but never defined")

    def test_page_does_not_block_framing(self):
        # The UI is served inside a preview iframe; these headers/headers-by-meta
        # would break it.
        lowered = self.page.lower()
        self.assertNotIn("x-frame-options", lowered)
        self.assertNotIn("frame-ancestors", lowered)

    def test_page_uses_relative_urls(self):
        # The browser is not the sandbox, so absolute localhost URLs would fail.
        self.assertNotIn("http://localhost", self.page)
        self.assertNotIn("http://127.0.0.1", self.page)

    def test_has_a_composer_and_transcript(self):
        self.assertIn('id="form"', self.page)
        self.assertIn('id="box"', self.page)
        self.assertIn('id="thread"', self.page)


class TestApi(unittest.TestCase):
    def setUp(self):
        self.agent = EddiAgent(persist_context=False)
        self.agent.teach("water", "chemical_formula", "H2O")
        self.server, self.thread, self.base = _start(self.agent)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.agent.close(save=False)

    def test_health(self):
        status, body = _get(self.base, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertIn("version", body)

    def test_stats(self):
        status, body = _get(self.base, "/stats")
        self.assertEqual(status, 200)
        self.assertIn("facts", body)

    def test_ask_returns_answer_with_kind_and_confidence(self):
        status, body = _post(self.base, "/ask", {"text": "tell me about water"})
        self.assertEqual(status, 200)
        self.assertIn("H2O", body["text"])
        self.assertIn("kind", body)
        self.assertIn("confidence", body)

    def test_ask_rejects_empty_text(self):
        status, body = _post(self.base, "/ask", {"text": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_ask_rejects_bad_json(self):
        req = urllib.request.Request(
            self.base + "/ask", data=b"not json", headers={"Content-Type": "application/json"}
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 400)

    def test_unknown_route_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/nope", timeout=10)
        self.assertEqual(ctx.exception.code, 404)

    def test_teach_then_ask(self):
        status, body = _post(
            self.base, "/teach", {"subject": "zorblax", "relation": "colour", "object": "ultraviolet"}
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        status, body = _post(self.base, "/ask", {"text": "what colour is zorblax"})
        self.assertIn("ultraviolet", body["text"])

    def test_teach_requires_all_three_fields(self):
        status, body = _post(self.base, "/teach", {"subject": "a", "relation": "b"})
        self.assertEqual(status, 400)

    def test_ingest_accepts_text(self):
        status, body = _post(self.base, "/ingest", {"text": "The sky is blue above the hills."})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])


class TestPersistence(unittest.TestCase):
    """A fact taught through the UI must survive a restart."""

    def test_taught_fact_survives_reload(self):
        import os

        from eddi.training import load_brain

        path = os.path.join(tempfile.mkdtemp(), "brain.pkl")
        agent = EddiAgent(persist_context=False, brain_path=path)
        server, thread, base = _start(agent, path, save_on_change=True)
        try:
            _post(
                base,
                "/teach",
                {"subject": "quux", "relation": "colour", "object": "puce"},
            )
        finally:
            server.shutdown()
            server.server_close()
            agent.close(save=False)

        self.assertTrue(os.path.exists(path))
        reloaded = load_brain(path, persist_context=False)
        try:
            self.assertTrue(reloaded.kb.facts_about("quux"))
            self.assertIn("puce", reloaded.answer("what colour is quux").text)
        finally:
            reloaded.close(save=False)


class TestConcurrency(unittest.TestCase):
    """Many requests in flight must not interleave inside the agent."""

    def test_parallel_asks_all_succeed(self):
        agent = EddiAgent(persist_context=False)
        agent.teach("water", "chemical_formula", "H2O")
        server, thread, base = _start(agent)
        results = []
        lock = threading.Lock()

        def worker():
            try:
                status, body = _post(base, "/ask", {"text": "what is the chemical formula of water"})
                with lock:
                    results.append((status, "H2O" in body.get("text", "")))
            except Exception as exc:  # pragma: no cover - only on failure
                with lock:
                    results.append((0, str(exc)))

        try:
            threads = [threading.Thread(target=worker) for _ in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)
        finally:
            server.shutdown()
            server.server_close()
            agent.close(save=False)

        self.assertEqual(len(results), 12)
        self.assertTrue(all(status == 200 for status, _ in results), results)
        self.assertTrue(all(ok is True for _, ok in results), results)


class TestHonesty(unittest.TestCase):
    """The UI must never present fabrication as knowledge."""

    def setUp(self):
        from eddi.curriculum import PACKS

        self.agent = EddiAgent(persist_context=False)
        for pack in ("core", "science", "computing"):
            for s, r, o in PACKS[pack]["facts"]:
                self.agent.teach(s, r, o)
        # The refusal path lives in _handle_generate, which only runs when a
        # language model exists. Without training, the honest answer is
        # "I have no model yet" - correct, but it does not test this code.
        for pack in ("core", "science", "computing"):
            for sentence in PACKS[pack].get("prose", []):
                self.agent.train_corpus([sentence])

    def tearDown(self):
        self.agent.close(save=False)

    def test_unknown_word_is_refused_not_continued(self):
        answer = self.agent.answer("what is zorblax")
        self.assertEqual(answer.kind, "unknown")
        self.assertEqual(answer.confidence, 0.0)
        self.assertIn("don't know", answer.text.lower())

    def test_refusal_names_the_unknown_term(self):
        answer = self.agent.answer("what is zorblax")
        self.assertIn("zorblax", answer.text)

    def test_known_subject_still_answers(self):
        answer = self.agent.answer("tell me about water")
        self.assertEqual(answer.kind, "about")
        self.assertIn("H2O", answer.text)

    def test_known_fact_not_blocked_by_out_of_vocabulary_guard(self):
        # 'zorblax' is not in the language model's vocabulary, but after teach
        # it IS knowledge. The guard must not shadow the knowledge base.
        self.agent.teach("zorblax", "colour", "ultraviolet")
        answer = self.agent.answer("what colour is zorblax")
        self.assertIn("ultraviolet", answer.text)

    def test_generated_answer_does_not_echo_the_question(self):
        answer = self.agent.answer("what is a database")
        if answer.kind == "generate":
            self.assertFalse(
                answer.text.lower().startswith("what is a database"),
                f"generated text echoed the prompt: {answer.text!r}",
            )

    def test_greeting_with_a_follow_up_is_a_greeting(self):
        # "Hi how are you?" is small talk, not a knowledge lookup. It used to
        # fall through to the n-gram model and answer about something else.
        answer = self.agent.answer("Hi how are you?")
        self.assertEqual(answer.kind, "greeting")

    def test_bare_pleasantries_are_recognised(self):
        for text in ("Hi", "hello", "Hello there", "hey eddi", "good morning",
                     "how are you?", "thanks", "thank you", "bye", "good afternoon"):
            with self.subTest(text=text):
                self.assertEqual(self.agent.answer(text).kind, "greeting")

    def test_greeting_does_not_swallow_a_real_question(self):
        # A pleasantry followed by a question must still answer the question.
        for text in ("hi, what is python?", "Hi! Tell me about water"):
            with self.subTest(text=text):
                answer = self.agent.answer(text)
                self.assertIn(answer.kind, {"about", "ask"})

    def test_refusal_does_not_prescribe_cli_syntax(self):
        # Browser users cannot type shell commands; the UI offers a form.
        answer = self.agent.answer("tell me about an unladen swallow")
        self.assertNotIn("python -m eddi", answer.text)
        self.assertNotIn("Try 'teach", answer.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
