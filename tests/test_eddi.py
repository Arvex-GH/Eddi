import os
import unittest
from eddi.agent import EddiAgent
from eddi.hdc import HyperVector, WritableKnowledgeBase


class TestEddi(unittest.TestCase):
    def test_hdc_bind_unbind(self):
        a = HyperVector.from_symbol("subject")
        b = HyperVector.from_symbol("object")
        bound = a.bind(b)
        recovered = bound.bind(a)
        self.assertAlmostEqual(recovered.similarity(b), 1.0)

    def test_writable_kb(self):
        kb = WritableKnowledgeBase()
        kb.write_fact("RTX2060Super", "vram", "8GB")
        hits = kb.query("RTX2060Super", "vram")
        self.assertTrue(len(hits) >= 1)
        self.assertEqual(hits[0][0], "8GB")

    def test_agent_ingest_and_teach(self):
        path = "/tmp/test_eddi_ctx.mmap"
        agent = EddiAgent(context_path=path)
        try:
            n = agent.ingest_text("hello world this is a test of eddi context")
            self.assertEqual(n, 9)
            gen = agent.generate("hello world", max_tokens=5)
            self.assertIn("this is a test of", gen)
            agent.teach("Eddi", "status", "working")
            ans = agent.ask("Eddi", "status")
            self.assertTrue(any("working" in a for a in ans))
        finally:
            agent.close()
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
