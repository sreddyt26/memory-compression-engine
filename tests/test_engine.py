"""
test_engine.py — unit tests for the Memory Compression Engine.

Uses only Python's built-in unittest (no pytest needed, nothing to install).

Run:
    python -m unittest discover -s tests
  or from the tests/ folder:
    python test_engine.py
"""
import os, sys, unittest

# make src/ importable regardless of where tests are run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import (
    MemoryStore, compress, extractive_compress, importance_score,
    keyword_recall, split_sentences, cosine, PureEmbedder,
)


class TestCompression(unittest.TestCase):
    def test_compression_reduces_tokens(self):
        """A multi-sentence passage should get shorter when compressed."""
        text = ("We decided the launch date is September 15th. "
                "The weather was nice. Someone had coffee. "
                "The budget is $12,000 and was approved by finance.")
        mem = compress(text, tier="heavy")
        self.assertLess(mem.tokens_compressed, mem.tokens_original)
        self.assertLessEqual(mem.compression_ratio, 1.0)

    def test_single_sentence_not_broken(self):
        """A one-sentence input should survive intact."""
        text = "The budget is $12,000."
        out = extractive_compress(text, keep_ratio=0.4)
        self.assertIn("12,000", out)

    def test_important_facts_survive(self):
        """Sentences with cue words / numbers should be preferentially kept."""
        text = ("The intern likes pizza. The team is nice. "
                "The critical deadline is August 20th and must not slip.")
        out = extractive_compress(text, keep_ratio=0.34)
        self.assertIn("August 20th", out)


class TestImportance(unittest.TestCase):
    def test_factual_scores_higher_than_chitchat(self):
        factual = ("We decided the budget is $12,000 and the deadline "
                   "is September 15th, which is important.")
        chitchat = "The weather is nice and the coffee was warm today."
        self.assertGreater(importance_score(factual),
                           importance_score(chitchat))


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(raw_window=2, light_window=2)
        for t in [
            "The launch date is September 15th and the budget is $12,000.",
            "The venue is the Downtown Convention Center holding 300 people.",
            "Marketing uses email newsletters and LinkedIn ads.",
        ]:
            self.store.add(t)

    def test_retrieval_finds_relevant_memory(self):
        hits = self.store.retrieve("where is the venue?", k=1)
        self.assertTrue(hits)
        top_text = hits[0][0].text.lower()
        self.assertIn("convention center", top_text)

    def test_retrieval_ranking(self):
        hits = self.store.retrieve("budget and launch date", k=3)
        # the budget/date memory should rank first
        self.assertIn("september 15th", hits[0][0].text.lower())

    def test_empty_store_returns_nothing(self):
        empty = MemoryStore()
        self.assertEqual(empty.retrieve("anything"), [])


class TestHierarchicalDecay(unittest.TestCase):
    def test_newest_stays_raw_oldest_decays(self):
        store = MemoryStore(raw_window=1, light_window=1)
        for i in range(4):
            store.add(f"This is memory number {i} with some filler content here. "
                      f"It has a second sentence to allow compression to happen.")
        tiers = [m.tier for m in store.memories]
        # newest (last) should be raw; oldest (first) should be heavy
        self.assertEqual(tiers[-1], "raw")
        self.assertEqual(tiers[0], "heavy")


class TestPrimitives(unittest.TestCase):
    def test_cosine_identical_is_one(self):
        emb = PureEmbedder(); emb.fit(["budget deadline launch"])
        v = emb.vector("budget deadline launch")
        self.assertAlmostEqual(cosine(v, v), 1.0, places=5)

    def test_cosine_orthogonal_is_zero(self):
        emb = PureEmbedder(); emb.fit(["apple banana", "rocket engine"])
        a = emb.vector("apple banana")
        b = emb.vector("rocket engine")
        self.assertAlmostEqual(cosine(a, b), 0.0, places=5)

    def test_split_sentences(self):
        s = split_sentences("First one. Second one! Third one?")
        self.assertEqual(len(s), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
