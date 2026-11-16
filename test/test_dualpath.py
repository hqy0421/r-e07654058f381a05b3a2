import unittest

import numpy as np

from datasketch import DualPathResult, DualPathRetriever
from datasketch.dualpath import TEXT_PATH, VECTOR_PATH


def _tokens(prefix, count):
    return ["%s%d" % (prefix, i) for i in range(count)]


def _text_batch():
    """Records with high pairwise Jaccard so the LSH Forest recalls all of
    them deterministically.
    """
    common = _tokens("t", 40)
    query_tokens = common + _tokens("q", 10)
    records = [
        # Jaccard with query: 50/50 = 1.0
        {"id": "a", "tokens": common + _tokens("q", 10)},
        # Jaccard with query: 40/60
        {"id": "b", "tokens": common + _tokens("b", 10)},
        # Jaccard with query: 40/70
        {"id": "c", "tokens": common + _tokens("c", 20)},
    ]
    return records, query_tokens


def _vector_batch():
    return [
        {"id": "x", "vector": [0.0, 0.0]},
        {"id": "y", "vector": [1.0, 0.0]},
        {"id": "z", "vector": [2.0, 0.0]},
    ]


def _both_path_batch():
    records, query_tokens = _text_batch()
    vectors = {"a": [0.0, 0.0], "b": [1.0, 0.0], "c": [2.0, 0.0]}
    for record in records:
        record["vector"] = vectors[record["id"]]
    query_vector = [0.1, 0.0]
    return records, query_tokens, query_vector


class TestDualPathRetriever(unittest.TestCase):
    def test_text_path_ranked_by_exact_jaccard(self):
        records, query_tokens = _text_batch()
        retriever = DualPathRetriever()
        retriever.load_batch(records)
        results = retriever.query(tokens=query_tokens, k=3)
        self.assertEqual([r.key for r in results], ["a", "b", "c"])
        for rank, result in enumerate(results, start=1):
            self.assertEqual(result.paths, (TEXT_PATH,))
            self.assertEqual(result.ranks, {TEXT_PATH: rank})
        self.assertAlmostEqual(results[0].raw_scores[TEXT_PATH], 1.0)
        self.assertAlmostEqual(results[1].raw_scores[TEXT_PATH], 40.0 / 60.0)
        self.assertAlmostEqual(results[2].raw_scores[TEXT_PATH], 40.0 / 70.0)

    def test_vector_path_ranked_by_native_distance(self):
        retriever = DualPathRetriever(hnsw_seed=42)
        retriever.load_batch(_vector_batch())
        results = retriever.query(vector=[0.1, 0.0], k=3)
        self.assertEqual([r.key for r in results], ["x", "y", "z"])
        for rank, result in enumerate(results, start=1):
            self.assertEqual(result.paths, (VECTOR_PATH,))
            self.assertEqual(result.ranks, {VECTOR_PATH: rank})
        self.assertAlmostEqual(results[0].raw_scores[VECTOR_PATH], 0.1)
        self.assertAlmostEqual(results[1].raw_scores[VECTOR_PATH], 0.9)
        self.assertAlmostEqual(results[2].raw_scores[VECTOR_PATH], 1.9)

    def test_fusion_score_and_hit_paths(self):
        records, query_tokens, query_vector = _both_path_batch()
        retriever = DualPathRetriever(text_weight=2.0, vector_weight=1.0, c=10.0, hnsw_seed=42)
        retriever.load_batch(records)
        results = retriever.query(tokens=query_tokens, vector=query_vector, k=3)
        self.assertEqual([r.key for r in results], ["a", "b", "c"])
        # "a" is rank 1 on both paths.
        a = results[0]
        self.assertEqual(a.paths, (TEXT_PATH, VECTOR_PATH))
        self.assertEqual(a.ranks, {TEXT_PATH: 1, VECTOR_PATH: 1})
        self.assertAlmostEqual(a.score, 2.0 / 11.0 + 1.0 / 11.0)
        self.assertAlmostEqual(a.raw_scores[TEXT_PATH], 1.0)
        self.assertAlmostEqual(a.raw_scores[VECTOR_PATH], 0.1)
        # "b" is rank 2 on both paths.
        b = results[1]
        self.assertEqual(b.ranks, {TEXT_PATH: 2, VECTOR_PATH: 2})
        self.assertAlmostEqual(b.score, 2.0 / 12.0 + 1.0 / 12.0)
        # Scores are positive and descending.
        scores = [r.score for r in results]
        self.assertTrue(all(s > 0 for s in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_fusion_adds_contributions_per_id(self):
        # A record hit by both paths scores higher than a record hit by one
        # path at a better rank.
        common = _tokens("t", 40)
        query_tokens = common + _tokens("q", 10)
        records = [
            # Text rank 1 (J=1.0), no vector.
            {"id": "text-only", "tokens": common + _tokens("q", 10)},
            # Text rank 2 (J=40/60) and vector rank 1 (distance 0.0).
            {"id": "both", "tokens": common + _tokens("b", 10), "vector": [0.0, 0.0]},
        ]
        retriever = DualPathRetriever(hnsw_seed=42)
        retriever.load_batch(records)
        results = retriever.query(tokens=query_tokens, vector=[0.0, 0.0], k=2)
        self.assertEqual([r.key for r in results], ["both", "text-only"])
        both, text_only = results
        self.assertEqual(both.paths, (TEXT_PATH, VECTOR_PATH))
        self.assertAlmostEqual(both.score, 1.0 / 62.0 + 1.0 / 61.0)
        self.assertEqual(text_only.paths, (TEXT_PATH,))
        self.assertAlmostEqual(text_only.score, 1.0 / 61.0)

    def test_query_with_one_path_only(self):
        records, query_tokens, query_vector = _both_path_batch()
        retriever = DualPathRetriever(hnsw_seed=42)
        retriever.load_batch(records)
        # Tokens only.
        results = retriever.query(tokens=query_tokens, k=3)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.paths, (TEXT_PATH,))
        # Vector only.
        results = retriever.query(vector=query_vector, k=3)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.paths, (VECTOR_PATH,))
        # Neither path.
        self.assertEqual(retriever.query(k=3), [])
        # Empty token set behaves like a missing token path.
        results = retriever.query(tokens=[], vector=query_vector, k=3)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.paths, (VECTOR_PATH,))

    def test_records_missing_a_path_do_not_participate(self):
        records = [
            {"id": "no-text", "vector": [0.0, 0.0]},
            {"id": "empty-text", "tokens": [], "vector": [1.0, 0.0]},
            {"id": "no-vector", "tokens": ["alpha", "beta"]},
            {"id": "both", "tokens": ["alpha", "beta"], "vector": [2.0, 0.0]},
        ]
        retriever = DualPathRetriever(hnsw_seed=42)
        retriever.load_batch(records)
        # Text query: records without tokens cannot be hit.
        results = retriever.query(tokens=["alpha", "beta"], k=10)
        keys = {r.key for r in results}
        self.assertIn("no-vector", keys)
        self.assertIn("both", keys)
        self.assertNotIn("no-text", keys)
        self.assertNotIn("empty-text", keys)
        # Vector query: records without a vector cannot be hit.
        results = retriever.query(vector=[0.0, 0.0], k=10)
        keys = {r.key for r in results}
        self.assertIn("no-text", keys)
        self.assertIn("empty-text", keys)
        self.assertIn("both", keys)
        self.assertNotIn("no-vector", keys)

    def test_original_record_returned(self):
        record = {"id": "a", "tokens": ["x"], "vector": [0.0], "payload": {"foo": 1}}
        retriever = DualPathRetriever(hnsw_seed=42)
        retriever.load_batch([record])
        results = retriever.query(tokens=["x"], vector=[0.0], k=1)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].record, record)
        self.assertEqual(results[0].record["payload"], {"foo": 1})
        self.assertIsInstance(results[0], DualPathResult)

    def test_reload_replaces_batch(self):
        retriever = DualPathRetriever(hnsw_seed=42)
        retriever.load_batch([{"id": "old", "tokens": ["hello"], "vector": [0.0]}])
        self.assertIn("old", retriever)
        self.assertEqual(len(retriever), 1)
        retriever.load_batch([{"id": "new", "tokens": ["world"], "vector": [1.0]}])
        self.assertNotIn("old", retriever)
        self.assertIn("new", retriever)
        self.assertEqual(len(retriever), 1)
        results = retriever.query(tokens=["world"], vector=[1.0], k=10)
        self.assertEqual([r.key for r in results], ["new"])
        # Old IDs are no longer retrievable through either path.
        self.assertEqual(retriever.query(tokens=["hello"], k=10), [])
        self.assertEqual([r.key for r in retriever.query(vector=[0.0], k=10)], ["new"])

    def test_failed_reload_keeps_old_batch(self):
        retriever = DualPathRetriever(hnsw_seed=42)
        old_records = [{"id": "old", "tokens": ["hello"], "vector": [0.0]}]
        retriever.load_batch(old_records)
        # Duplicate IDs in the new batch.
        self.assertRaises(
            ValueError,
            retriever.load_batch,
            [{"id": "dup", "tokens": ["a"]}, {"id": "dup", "tokens": ["b"]}],
        )
        # Inconsistent vector dimensions in the new batch.
        self.assertRaises(
            ValueError,
            retriever.load_batch,
            [{"id": "p", "vector": [1.0, 2.0]}, {"id": "q", "vector": [1.0, 2.0, 3.0]}],
        )
        # Record without an "id" key.
        self.assertRaises(ValueError, retriever.load_batch, [{"tokens": ["a"]}])
        # Non-mapping record.
        self.assertRaises(TypeError, retriever.load_batch, [("id", ["a"])])
        # The old batch is still fully queryable through both paths.
        self.assertEqual(len(retriever), 1)
        self.assertIn("old", retriever)
        results = retriever.query(tokens=["hello"], vector=[0.0], k=10)
        self.assertEqual([r.key for r in results], ["old"])
        self.assertIs(results[0].record, old_records[0])

    def test_index_and_records_are_same_batch(self):
        # After a failed reload the returned records must still come from the
        # same batch as the indexes.
        retriever = DualPathRetriever(hnsw_seed=42)
        batch1 = [{"id": 1, "tokens": ["alpha"], "vector": [0.0]}]
        retriever.load_batch(batch1)
        batch2 = [{"id": 2, "tokens": ["beta"], "vector": [1.0]}, {"id": 2, "tokens": ["x"]}]
        self.assertRaises(ValueError, retriever.load_batch, batch2)
        results = retriever.query(tokens=["alpha"], vector=[0.0], k=10)
        self.assertEqual([r.key for r in results], [1])
        self.assertIs(results[0].record, batch1[0])

    def test_validation(self):
        self.assertRaises(ValueError, DualPathRetriever, text_weight=-1.0)
        self.assertRaises(ValueError, DualPathRetriever, vector_weight=-0.5)
        self.assertRaises(ValueError, DualPathRetriever, c=0.0)
        self.assertRaises(ValueError, DualPathRetriever, c=-1.0)
        retriever = DualPathRetriever()
        # Query before any batch is loaded.
        self.assertRaises(ValueError, retriever.query, tokens=["a"], k=1)
        retriever.load_batch([{"id": "a", "tokens": ["a"]}])
        self.assertRaises(ValueError, retriever.query, tokens=["a"], k=0)
        self.assertRaises(ValueError, retriever.query, tokens=["a"], k=1, num_candidates=0)

    def test_zero_weight_path_contributes_nothing(self):
        records, query_tokens, query_vector = _both_path_batch()
        retriever = DualPathRetriever(text_weight=0.0, vector_weight=1.0, c=10.0, hnsw_seed=42)
        retriever.load_batch(records)
        results = retriever.query(tokens=query_tokens, vector=query_vector, k=3)
        by_key = {r.key: r for r in results}
        # Text hits are still reported with their ranks, but add nothing to
        # the fusion score.
        a = by_key["a"]
        self.assertEqual(a.paths, (TEXT_PATH, VECTOR_PATH))
        self.assertEqual(a.ranks[TEXT_PATH], 1)
        self.assertAlmostEqual(a.score, 1.0 / 11.0)

    def test_top_k_and_num_candidates(self):
        records, query_tokens, _ = _both_path_batch()
        retriever = DualPathRetriever()
        retriever.load_batch(records)
        self.assertEqual(len(retriever.query(tokens=query_tokens, k=2)), 2)
        # Only the top text candidate participates in fusion.
        results = retriever.query(tokens=query_tokens, k=3, num_candidates=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].key, "a")
        self.assertEqual(results[0].ranks, {TEXT_PATH: 1})

    def test_empty_batch(self):
        retriever = DualPathRetriever()
        self.assertTrue(retriever.is_empty())
        self.assertEqual(len(retriever), 0)
        retriever.load_batch([])
        self.assertTrue(retriever.is_empty())
        self.assertEqual(retriever.query(tokens=["a"], vector=[0.0], k=5), [])

    def test_custom_distance_func(self):
        retriever = DualPathRetriever(
            distance_func=lambda x, y: float(np.sum(np.abs(x - y))),
            hnsw_seed=42,
        )
        retriever.load_batch(_vector_batch())
        results = retriever.query(vector=[0.1, 0.0], k=1)
        self.assertEqual(results[0].key, "x")
        self.assertAlmostEqual(results[0].raw_scores[VECTOR_PATH], 0.1)

    def test_bytes_and_string_tokens_equivalent(self):
        retriever = DualPathRetriever()
        retriever.load_batch([{"id": "a", "tokens": [b"hello", b"world"]}])
        results = retriever.query(tokens=["hello", "world"], k=1)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].raw_scores[TEXT_PATH], 1.0)

    def test_unhashable_or_missing_id_rejected(self):
        retriever = DualPathRetriever()
        self.assertRaises(TypeError, retriever.load_batch, [{"id": ["unhashable"], "tokens": ["a"]}])
        self.assertTrue(retriever.is_empty())


if __name__ == "__main__":
    unittest.main()
