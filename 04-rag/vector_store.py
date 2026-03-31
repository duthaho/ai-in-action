"""
FAISS-based vector store with metadata storage and persistence.

This wraps FAISS (Facebook AI Similarity Search) to provide:
- Add vectors with metadata
- Search by cosine similarity
- Save/load index to disk
- Basic statistics

For production, you'd replace this with pgvector, Qdrant, or Pinecone.
FAISS is excellent for learning and for systems under ~1M vectors.
"""

import json
import os
from dataclasses import dataclass, field

import faiss
import numpy as np


@dataclass
class SearchResult:
    """A single search result with score and metadata."""
    score: float           # cosine similarity (0 to 1, higher = more similar)
    metadata: dict         # stored metadata (text, source, etc.)
    vector_id: int         # internal FAISS ID


class FaissVectorStore:
    """
    Vector store using FAISS IndexFlatIP (inner product = cosine similarity
    when vectors are L2-normalized).

    Architecture note: FAISS IndexFlatIP does brute-force search — it compares
    the query against every vector. This is fine up to ~100K vectors. For larger
    scales, you'd switch to IndexIVFFlat (IVF) or IndexHNSWFlat (HNSW).
    """

    def __init__(self, dimension: int):
        """
        Initialize an empty vector store.

        Args:
            dimension: Vector dimensionality (must match your embedding model).
                       e.g., 1024 for voyage-3-large, 384 for all-MiniLM-L6-v2
        """
        self.dimension = dimension
        # IndexFlatIP = brute-force inner product search
        # With L2-normalized vectors, inner product = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store: dict[int, dict] = {}
        self._next_id = 0

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        return self.index.ntotal

    def add(self, embedding: list[float], metadata: dict) -> int:
        """
        Add a single vector with metadata.

        Args:
            embedding: The embedding vector (will be L2-normalized)
            metadata: Arbitrary metadata dict (stored alongside the vector)

        Returns:
            The ID assigned to this vector
        """
        vector = np.array([embedding], dtype=np.float32)
        # L2 normalize so inner product = cosine similarity
        faiss.normalize_L2(vector)
        self.index.add(vector)

        vector_id = self._next_id
        self.metadata_store[vector_id] = metadata
        self._next_id += 1

        return vector_id

    def add_batch(self, embeddings: list[list[float]], metadata_list: list[dict]) -> list[int]:
        """
        Add multiple vectors at once (more efficient than one-by-one).

        Args:
            embeddings: List of embedding vectors
            metadata_list: Corresponding metadata for each vector

        Returns:
            List of assigned IDs
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("embeddings and metadata_list must have same length")

        if not embeddings:
            return []

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        ids = []
        for meta in metadata_list:
            vector_id = self._next_id
            self.metadata_store[vector_id] = meta
            self._next_id += 1
            ids.append(vector_id)

        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Find the most similar vectors to the query.

        Args:
            query_embedding: The query vector
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1). Results below this are filtered out.

        Returns:
            List of SearchResult, sorted by descending similarity
        """
        if self.size == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        # FAISS search returns (distances, indices) arrays
        scores, indices = self.index.search(query, min(top_k, self.size))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            if score < min_score:
                continue
            results.append(SearchResult(
                score=float(score),
                metadata=self.metadata_store.get(int(idx), {}),
                vector_id=int(idx),
            ))

        return results

    def save(self, directory: str) -> None:
        """
        Persist the index and metadata to disk.

        Saves two files:
        - index.faiss: The FAISS index (vectors)
        - metadata.json: The metadata store
        """
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        meta = {
            "dimension": self.dimension,
            "next_id": self._next_id,
            "metadata": {str(k): v for k, v in self.metadata_store.items()},
        }
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "FaissVectorStore":
        """Load a previously saved index from disk."""
        index = faiss.read_index(os.path.join(directory, "index.faiss"))

        with open(os.path.join(directory, "metadata.json")) as f:
            meta = json.load(f)

        store = cls(dimension=meta["dimension"])
        store.index = index
        store._next_id = meta["next_id"]
        store.metadata_store = {int(k): v for k, v in meta["metadata"].items()}

        return store

    def clear(self) -> None:
        """Remove all vectors and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store.clear()
        self._next_id = 0
