from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

import numpy as np

from datasketch.hnsw import HNSW
from datasketch.lshforest import MinHashLSHForest
from datasketch.minhash import MinHash

#: Name of the token-set retrieval path (MinHashLSHForest).
TEXT_PATH = "text"
#: Name of the numeric-vector retrieval path (HNSW).
VECTOR_PATH = "vector"


def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _normalize_tokens(tokens: Iterable) -> frozenset:
    """Normalize a token collection into a frozenset, encoding strings as
    UTF-8 so that the exact Jaccard similarity is computed over the same
    values that the MinHash hashes.
    """
    return frozenset(token.encode("utf-8") if isinstance(token, str) else token for token in tokens)


@dataclass
class DualPathResult:
    """A single fused retrieval result.

    Attributes:
        key (Hashable): The ID of the record in the loaded batch.
        record: The original record mapping, as passed to
            :meth:`DualPathRetriever.load_batch`.
        score (float): The fusion score: the sum of ``w / (c + r)`` over
            the paths that hit this record, where ``r`` is the 1-based rank
            of the record on that path.
        paths (tuple): The paths that actually hit this record, a subset of
            ``("text", "vector")`` in that order.
        ranks (dict): Per-path 1-based rank, e.g. ``{"text": 1}``. Only
            hit paths are present.
        raw_scores (dict): Per-path raw score: the exact Jaccard similarity
            of the original token sets for ``"text"`` (higher is better) and
            the native distance for ``"vector"`` (lower is better). Only hit
            paths are present.

    """

    key: Hashable
    record: Any
    score: float
    paths: tuple
    ranks: dict
    raw_scores: dict


class _Batch(NamedTuple):
    """One consistent batch of records and their indexes. Instances are
    immutable and swapped atomically, so a query always sees indexes and
    records from the same batch.
    """

    records: dict  # key -> original record
    token_sets: dict  # key -> frozenset of tokens (text path participants)
    forest: MinHashLSHForest
    hnsw: HNSW


class DualPathRetriever:
    """Dual-path similar case retrieval over one batch of ID-aligned records.

    The token path recalls candidates with :class:`MinHashLSHForest` and
    ranks them by the exact Jaccard similarity of the original token sets
    (descending); the vector path recalls candidates with :class:`HNSW` and
    ranks them by the native distance (ascending). Both rankings are
    numbered from 1, and a record ranked ``r`` on a path contributes
    ``w / (c + r)`` to its fusion score, where ``w`` is the non-negative
    weight of that path and ``c`` a positive constant. Contributions are
    added per record ID and the top-k records are returned.

    Records are loaded as one batch. Every record is a mapping with:

    - ``"id"``: the unique, hashable record ID (required). The same ID
      aligns the record across both paths.
    - ``"tokens"``: an iterable of hashables for the token path (optional).
      Records with a missing or empty token set do not participate in the
      token path. Strings are encoded as UTF-8, and the exact Jaccard
      similarity is computed over the encoded sets, i.e. over the same
      values that the MinHash hashes.
    - ``"vector"``: a numeric vector for the vector path (optional).
      Records with a missing vector do not participate in the vector path.

    Any additional keys are kept as part of the record and returned
    untouched in query results.

    Loading is atomic: the new indexes and records are built on the side and
    swapped in as one consistent batch, so a failed load leaves the previous
    batch fully queryable, and a concurrent query never mixes indexes and
    records from different batches.

    Args:
        num_perm (int): The number of permutation functions for the MinHash
            used by the token path.
        l (int): The number of prefix trees of the LSH Forest.
        distance_func (Optional[Callable]): The distance function of the
            HNSW index. Defaults to Euclidean distance.
        text_weight (float): The non-negative weight ``w`` of the token
            path in the fusion score.
        vector_weight (float): The non-negative weight ``w`` of the vector
            path in the fusion score.
        c (float): The positive constant ``c`` of the fusion score
            ``w / (c + r)``.
        seed (int): The random seed of the MinHash used by the token path.
        hashfunc (Optional[Callable]): The hash function of the MinHash used
            by the token path. If None, the MinHash default is used.
        scheme (Optional[str]): The permutation scheme of the MinHash used
            by the token path. If None, the MinHash default is used.
        hnsw_m (int): The ``m`` parameter of the HNSW index.
        hnsw_ef_construction (int): The ``ef_construction`` parameter of the
            HNSW index.
        hnsw_m0 (Optional[int]): The ``m0`` parameter of the HNSW index.
        hnsw_seed (Optional[int]): The random seed of the HNSW index.

    Examples:
        .. code-block:: python

            from datasketch import DualPathRetriever

            retriever = DualPathRetriever(text_weight=1.0, vector_weight=1.0, c=60)
            retriever.load_batch(
                [
                    {"id": "case-1", "tokens": ["cardiac", "arrest"], "vector": [0.1, 0.9]},
                    {"id": "case-2", "tokens": ["cardiac", "failure"], "vector": [0.2, 0.8]},
                    {"id": "case-3", "tokens": ["broken", "arm"]},  # no vector
                ]
            )
            for r in retriever.query(tokens=["cardiac", "arrest"], vector=[0.1, 0.9], k=2):
                print(r.key, r.score, r.paths, r.ranks, r.raw_scores)

    """

    def __init__(
        self,
        num_perm: int = 128,
        l: int = 8,
        distance_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        text_weight: float = 1.0,
        vector_weight: float = 1.0,
        c: float = 60.0,
        seed: int = 1,
        hashfunc: Optional[Callable] = None,
        scheme: Optional[str] = None,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_m0: Optional[int] = None,
        hnsw_seed: Optional[int] = None,
    ) -> None:
        if text_weight < 0 or vector_weight < 0:
            raise ValueError("Path weights must be non-negative")
        if c <= 0:
            raise ValueError("c must be positive")
        self._num_perm = num_perm
        self._l = l
        self._distance_func = distance_func if distance_func is not None else _euclidean_distance
        self._text_weight = float(text_weight)
        self._vector_weight = float(vector_weight)
        self._c = float(c)
        self._seed = seed
        self._hashfunc = hashfunc
        self._scheme = scheme
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_m0 = hnsw_m0
        self._hnsw_seed = hnsw_seed
        self._state: Optional[_Batch] = None

    def _new_minhash(self) -> MinHash:
        return MinHash(
            num_perm=self._num_perm,
            seed=self._seed,
            hashfunc=self._hashfunc,
            scheme=self._scheme,
        )

    def _minhash_of(self, token_set: frozenset) -> MinHash:
        minhash = self._new_minhash()
        for token in token_set:
            minhash.update(token)
        return minhash

    def load_batch(self, records: Iterable[Mapping]) -> None:
        """Load a new batch of records, replacing the current batch.

        The indexes and the record store are built on the side and swapped
        in atomically as one consistent batch. If building the batch fails
        (e.g. a duplicate ID or an invalid vector), the exception propagates
        and the previous batch remains fully queryable.

        Args:
            records (Iterable[Mapping]): The records to load. See the class
                documentation for the expected mapping keys.

        Raises:
            TypeError: If a record is not a mapping.
            ValueError: If a record has no ``"id"`` key or the same ID
                appears twice in the batch.

        """
        new_records: dict = {}
        new_token_sets: dict = {}
        forest = MinHashLSHForest(num_perm=self._num_perm, l=self._l)
        hnsw = HNSW(
            distance_func=self._distance_func,
            m=self._hnsw_m,
            ef_construction=self._hnsw_ef_construction,
            m0=self._hnsw_m0,
            seed=self._hnsw_seed,
        )
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError(
                    "Each record must be a mapping with keys 'id', 'tokens' and 'vector', got %r" % (record,)
                )
            if "id" not in record:
                raise ValueError("Each record must have an 'id' key")
            key = record["id"]
            if key in new_records:
                raise ValueError("Duplicate record id in batch: %r" % (key,))
            new_records[key] = record
            tokens = record.get("tokens")
            if tokens is not None:
                token_set = _normalize_tokens(tokens)
                # Records with an empty token set do not participate in the
                # token path.
                if token_set:
                    forest.add(key, self._minhash_of(token_set))
                    new_token_sets[key] = token_set
            vector = record.get("vector")
            # Records with a missing vector do not participate in the vector
            # path.
            if vector is not None:
                hnsw.insert(key, np.array(vector))
        forest.index()
        # Atomic swap: a query always sees the indexes and the records of
        # one consistent batch, and a failure above leaves the previous
        # batch untouched.
        self._state = _Batch(new_records, new_token_sets, forest, hnsw)

    def query(
        self,
        tokens: Optional[Iterable] = None,
        vector: Optional[Iterable] = None,
        k: int = 10,
        num_candidates: Optional[int] = None,
        ef: Optional[int] = None,
    ) -> list[DualPathResult]:
        """Retrieve the top-k records by fused dual-path score.

        A query may omit one path: with no (or empty) ``tokens`` only the
        vector path is used, and with no ``vector`` only the token path is
        used. With neither, an empty list is returned.

        Args:
            tokens (Optional[Iterable]): The query token set for the token
                path. Strings are encoded as UTF-8 before hashing.
            vector (Optional[Iterable]): The query vector for the vector
                path.
            k (int): The maximum number of fused results to return.
            num_candidates (Optional[int]): The number of candidates
                recalled per path before fusion. Defaults to ``k``.
            ef (Optional[int]): The ``ef`` parameter of the HNSW query. If
                None, the construction ef is used.

        Returns:
            list[DualPathResult]: At most ``k`` results ordered by
            descending fusion score.

        Raises:
            ValueError: If no batch has been loaded, or ``k`` or
                ``num_candidates`` is not positive.

        """
        state = self._state
        if state is None:
            raise ValueError("No batch has been loaded; call load_batch before querying")
        if k <= 0:
            raise ValueError("k must be positive")
        if num_candidates is None:
            num_candidates = k
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")

        # Token path: approximate recall from the LSH Forest, then rank the
        # candidates by the exact Jaccard similarity of the original token
        # sets, descending. Ties are broken by repr(key) for determinism.
        text_ranked: list = []
        if tokens is not None and state.token_sets:
            query_tokens = _normalize_tokens(tokens)
            if query_tokens:
                minhash = self._minhash_of(query_tokens)
                candidates = state.forest.query(minhash, num_candidates)
                text_ranked = sorted(
                    ((key, _jaccard(query_tokens, state.token_sets[key])) for key in candidates),
                    key=lambda item: (-item[1], repr(item[0])),
                )

        # Vector path: approximate recall from HNSW, natively ordered by
        # ascending distance.
        vector_ranked: list = []
        if vector is not None and len(state.hnsw) > 0:
            vector_ranked = state.hnsw.query(np.array(vector), k=num_candidates, ef=ef)

        # Weighted reciprocal-rank fusion: a record ranked r (1-based) on a
        # path contributes w / (c + r); contributions are added per record
        # ID.
        fused: dict = {}
        for rank, (key, jaccard) in enumerate(text_ranked, start=1):
            entry = fused.setdefault(key, [0.0, None, None, None, None])
            entry[0] += self._text_weight / (self._c + rank)
            entry[1] = rank
            entry[2] = jaccard
        for rank, (key, distance) in enumerate(vector_ranked, start=1):
            entry = fused.setdefault(key, [0.0, None, None, None, None])
            entry[0] += self._vector_weight / (self._c + rank)
            entry[3] = rank
            entry[4] = distance

        ordered = sorted(fused.items(), key=lambda item: (-item[1][0], repr(item[0])))
        results = []
        for key, (score, text_rank, jaccard, vector_rank, distance) in ordered[:k]:
            paths = []
            ranks = {}
            raw_scores = {}
            if text_rank is not None:
                paths.append(TEXT_PATH)
                ranks[TEXT_PATH] = text_rank
                raw_scores[TEXT_PATH] = jaccard
            if vector_rank is not None:
                paths.append(VECTOR_PATH)
                ranks[VECTOR_PATH] = vector_rank
                raw_scores[VECTOR_PATH] = distance
            results.append(
                DualPathResult(
                    key=key,
                    record=state.records[key],
                    score=score,
                    paths=tuple(paths),
                    ranks=ranks,
                    raw_scores=raw_scores,
                )
            )
        return results

    def is_empty(self) -> bool:
        """Return True if no batch has been loaded or the current batch has
        no records.
        """
        return self._state is None or not self._state.records

    def __len__(self) -> int:
        """Return the number of records in the current batch."""
        return 0 if self._state is None else len(self._state.records)

    def __contains__(self, key: Hashable) -> bool:
        """Return True if the current batch contains a record with the given
        ID.
        """
        return self._state is not None and key in self._state.records
