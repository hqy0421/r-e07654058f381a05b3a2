"""Dual-path similar case retrieval: token sets recalled by
MinHashLSHForest and numeric vectors recalled by HNSW are fused with
weighted reciprocal-rank scores w / (c + r)."""

from datasketch import DualPathRetriever

cases = [
    {"id": "case-1", "tokens": ["cardiac", "arrest", "chest", "pain"], "vector": [0.10, 0.90]},
    {"id": "case-2", "tokens": ["cardiac", "failure", "chest", "pain"], "vector": [0.20, 0.80]},
    {"id": "case-3", "tokens": ["broken", "arm", "fracture"], "vector": [0.95, 0.05]},
    {"id": "case-4", "tokens": ["cardiac", "arrest", "chest"]},  # no vector
    {"id": "case-5", "vector": [0.15, 0.85]},  # no tokens
]

retriever = DualPathRetriever(text_weight=1.0, vector_weight=1.0, c=60, hnsw_seed=42)
retriever.load_batch(cases)

print("Both paths:")
for r in retriever.query(tokens=["cardiac", "arrest", "chest"], vector=[0.12, 0.88], k=3):
    print("  %s score=%.4f paths=%s ranks=%s raw=%s" % (r.key, r.score, r.paths, r.ranks, r.raw_scores))

print("Tokens only:")
for r in retriever.query(tokens=["cardiac", "arrest", "chest"], k=3):
    print("  %s score=%.4f paths=%s" % (r.key, r.score, r.paths))

print("Vector only:")
for r in retriever.query(vector=[0.12, 0.88], k=3):
    print("  %s score=%.4f paths=%s" % (r.key, r.score, r.paths))

# Reloading a new batch is atomic: a failed load keeps the old batch
# queryable.
try:
    retriever.load_batch([{"id": "case-1"}, {"id": "case-1"}])
except ValueError as e:
    print("Reload failed as expected: %s" % e)
print("Old batch still queryable:", [r.key for r in retriever.query(tokens=["cardiac"], k=2)])
