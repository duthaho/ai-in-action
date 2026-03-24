"""
Tests for the Embeddings & Vector Search module.

Tests chunking strategies, vector store operations, and the API endpoints.
Run: pytest test_embeddings.py -v
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker, Chunk
from vector_store import FaissVectorStore


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        text = "a" * 250
        chunks = chunker.chunk(text, source="test")

        assert len(chunks) >= 3
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.source == "test" for c in chunks)

    def test_empty_text(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_text_smaller_than_chunk(self):
        chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=100)
        chunks = chunker.chunk("Short text")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"

    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_indices_are_sequential(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk("word " * 100)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))


class TestSentenceChunker:
    def test_respects_sentence_boundaries(self):
        chunker = SentenceChunker(chunk_size=100, chunk_overlap=20)
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = chunker.chunk(text)

        # No chunk should end mid-word (roughly)
        for chunk in chunks:
            assert chunk.text[-1] in ".!? " or chunk.text.endswith(text.split(".")[-1].strip())

    def test_empty_text(self):
        chunker = SentenceChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk("") == []


class TestRecursiveChunker:
    def test_respects_paragraph_boundaries(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three content here."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_handles_long_paragraphs(self):
        """Long paragraphs should be split at finer-grained separators."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        text = "word " * 200  # 1000 chars in one paragraph
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # No chunk should exceed chunk_size by much (some tolerance for edge cases)
        for chunk in chunks:
            assert len(chunk.text) <= 150  # generous tolerance

    def test_source_preserved(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("Some text content", source="docs/readme.md")
        assert all(c.source == "docs/readme.md" for c in chunks)


# ---------------------------------------------------------------------------
# Vector store tests
# ---------------------------------------------------------------------------

class TestFaissVectorStore:
    def _random_embedding(self, dim: int = 384) -> list[float]:
        vec = np.random.randn(dim).astype(np.float32)
        return vec.tolist()

    def test_add_and_search(self):
        store = FaissVectorStore(dimension=384)

        # Add 3 vectors
        for i in range(3):
            store.add(self._random_embedding(), {"id": i, "text": f"doc {i}"})

        assert store.size == 3

        # Search should return results
        results = store.search(self._random_embedding(), top_k=2)
        assert len(results) == 2
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "metadata") for r in results)

    def test_add_batch(self):
        store = FaissVectorStore(dimension=384)
        embeddings = [self._random_embedding() for _ in range(10)]
        metadata = [{"id": i} for i in range(10)]

        ids = store.add_batch(embeddings, metadata)
        assert len(ids) == 10
        assert store.size == 10

    def test_batch_mismatch_raises(self):
        store = FaissVectorStore(dimension=384)
        with pytest.raises(ValueError):
            store.add_batch(
                [self._random_embedding()],
                [{"id": 0}, {"id": 1}],
            )

    def test_search_empty_store(self):
        store = FaissVectorStore(dimension=384)
        results = store.search(self._random_embedding())
        assert results == []

    def test_min_score_filter(self):
        store = FaissVectorStore(dimension=384)
        # Add a known vector
        known = [1.0] + [0.0] * 383
        store.add(known, {"text": "known"})

        # Search with a very different vector — should be filtered by high min_score
        different = [0.0] * 383 + [1.0]
        results = store.search(different, min_score=0.99)
        assert len(results) == 0

    def test_save_and_load(self, tmp_path):
        store = FaissVectorStore(dimension=384)
        embeddings = [self._random_embedding() for _ in range(5)]
        metadata = [{"text": f"doc {i}"} for i in range(5)]
        store.add_batch(embeddings, metadata)

        # Save
        save_dir = str(tmp_path / "test_index")
        store.save(save_dir)

        # Load
        loaded = FaissVectorStore.load(save_dir)
        assert loaded.size == 5
        assert loaded.dimension == 384

        # Search should work on loaded store
        results = loaded.search(self._random_embedding(), top_k=3)
        assert len(results) == 3

    def test_clear(self):
        store = FaissVectorStore(dimension=384)
        store.add(self._random_embedding(), {"text": "test"})
        assert store.size == 1

        store.clear()
        assert store.size == 0

    def test_similar_vectors_score_higher(self):
        """Vectors close in direction should have higher similarity."""
        store = FaissVectorStore(dimension=384)

        # Create a base vector and a similar one
        base = np.random.randn(384).astype(np.float32)
        similar = base + np.random.randn(384).astype(np.float32) * 0.1  # small perturbation
        different = np.random.randn(384).astype(np.float32)

        store.add(similar.tolist(), {"text": "similar"})
        store.add(different.tolist(), {"text": "different"})

        results = store.search(base.tolist(), top_k=2)
        # The similar vector should rank first
        assert results[0].metadata["text"] == "similar"
        assert results[0].score > results[1].score


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

class TestAPI:
    """Test the FastAPI endpoints using TestClient."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test client with local embedding backend."""
        import os
        os.environ["EMBEDDING_BACKEND"] = "local"

        from app import app
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "embedding_backend" in data

    def test_ingest_and_search(self):
        # Ingest a document
        doc = {
            "content": (
                "Password Reset Instructions\n\n"
                "To reset your password, go to the login page and click 'Forgot Password'. "
                "Enter your email address and we'll send you a reset link. "
                "The link expires in 24 hours.\n\n"
                "Account Security\n\n"
                "We recommend using a strong password with at least 12 characters. "
                "Enable two-factor authentication for additional security."
            ),
            "source": "help-center/password.md",
        }
        response = self.client.post("/documents", json=doc)
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_created"] > 0

        # Search for it
        search = {"query": "How do I change my login credentials?", "top_k": 3}
        response = self.client.post("/search", json=search)
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) > 0
        # Top result should be about passwords
        assert "password" in data["results"][0]["text"].lower()

    def test_search_empty_index(self):
        """Search on fresh app with no documents should fail gracefully."""
        # This test may pass or fail depending on whether other tests ran first
        # The key assertion is that it doesn't crash
        response = self.client.post("/search", json={"query": "test"})
        assert response.status_code in (200, 400)

    def test_compare_similarity(self):
        texts = [
            "How do I reset my password?",
            "I forgot my login credentials",
            "The weather is nice today",
        ]
        response = self.client.post("/compare-similarity", json=texts)
        assert response.status_code == 200
        data = response.json()

        # Password-related texts should be more similar to each other than to weather
        pairs = data["pairs"]
        password_pair = next(p for p in pairs if "password" in p["text_a"] and "login" in p["text_b"])
        weather_pairs = [p for p in pairs if "weather" in p["text_a"] or "weather" in p["text_b"]]

        assert password_pair["similarity"] > max(p["similarity"] for p in weather_pairs)

    def test_list_documents(self):
        response = self.client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data

    def test_ingest_empty_content(self):
        response = self.client.post("/documents", json={"content": ""})
        assert response.status_code == 422  # Pydantic validation
