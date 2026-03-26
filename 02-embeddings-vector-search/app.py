"""
Embeddings & Vector Search — Standalone Demo

Walks through the core concepts interactively:
1. Embed text and inspect vectors
2. Compare semantic similarity between texts
3. Chunk a document, embed chunks, store in FAISS
4. Semantic search (retrieval)
5. RAG: retrieve + generate with Claude

Supports two embedding backends:
- Voyage AI (higher quality) — requires VOYAGE_API_KEY
- Local sentence-transformers (free, no API key) — good for learning

Run: python app.py
"""

import os
import time

import anthropic
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from chunking import RecursiveChunker
from vector_store import FaissVectorStore

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")

VOYAGE_MODEL = "voyage-3-large"
VOYAGE_DIMENSION = 1024

LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_DIMENSION = 384

CLAUDE_MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class EmbeddingBackend:
    """Base class for embedding backends."""
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class VoyageBackend(EmbeddingBackend):
    """Voyage AI embedding backend — high quality, requires API key."""

    def __init__(self):
        import voyageai
        self.client = voyageai.Client()
        self.dimension = VOYAGE_DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), 128):
            batch = texts[i:i + 128]
            response = self.client.embed(texts=batch, model=VOYAGE_MODEL)
            all_embeddings.extend(response.embeddings)
        return all_embeddings


class LocalBackend(EmbeddingBackend):
    """Local sentence-transformers backend — free, no API key needed."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print(f"Loading local model: {LOCAL_MODEL_NAME} ...")
        self.model = SentenceTransformer(LOCAL_MODEL_NAME)
        self.dimension = LOCAL_DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


def create_embedding_backend() -> EmbeddingBackend:
    if EMBEDDING_BACKEND == "voyage":
        return VoyageBackend()
    return LocalBackend()


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_separator():
    print(f"\n{'-' * 60}\n")


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------

def demo_embed(embedder: EmbeddingBackend):
    """Show what an embedding vector looks like."""
    print_header("1. What Does an Embedding Look Like?")

    text = "The quick brown fox jumps over the lazy dog."
    print(f"Text: \"{text}\"")

    embedding = embedder.embed([text])[0]
    print(f"Dimension: {len(embedding)}")
    print(f"First 10 values: {[round(v, 4) for v in embedding[:10]]}")
    print(f"Min: {min(embedding):.4f}, Max: {max(embedding):.4f}")

    vec = np.array(embedding)
    print(f"L2 norm: {np.linalg.norm(vec):.4f} (≈1.0 means normalized)")


def demo_similarity(embedder: EmbeddingBackend):
    """Compare semantic similarity between texts."""
    print_header("2. Semantic Similarity")

    texts = [
        "How do I reset my password?",
        "I forgot my login credentials",
        "The weather is nice today",
        "It's sunny outside",
    ]

    print("Texts:")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t}")

    embeddings = embedder.embed(texts)
    vectors = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms
    similarity_matrix = normalized @ normalized.T

    print("\nPairwise cosine similarity:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = similarity_matrix[i][j]
            bar = "█" * int(score * 30)
            print(f"  [{i}] vs [{j}]: {score:.4f}  {bar}")

    print("\nNotice: semantically related texts score higher, even with")
    print("different words ('password' vs 'login credentials').")


def demo_chunking():
    """Demonstrate text chunking strategies."""
    print_header("3. Chunking a Document")

    document = (
        "Password Reset Instructions\n\n"
        "To reset your password, go to the login page and click 'Forgot Password'. "
        "Enter your email address and we'll send you a reset link. "
        "The link expires in 24 hours.\n\n"
        "Account Security\n\n"
        "We recommend using a strong password with at least 12 characters. "
        "Enable two-factor authentication for additional security. "
        "Never share your password with anyone.\n\n"
        "Contact Support\n\n"
        "If you're still having trouble, email support@example.com or call 1-800-HELP. "
        "Our support team is available 24/7."
    )

    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk(document, source="help-center/password.md")

    print(f"Document length: {len(document)} chars")
    print(f"Chunks created: {len(chunks)}")
    print()

    for chunk in chunks:
        print(f"  Chunk {chunk.index} ({len(chunk.text)} chars):")
        preview = chunk.text[:80].replace("\n", " ")
        print(f"    \"{preview}...\"")

    return document, chunker


def demo_search(embedder: EmbeddingBackend, document: str, chunker):
    """Chunk, embed, store, and search."""
    print_header("4. Semantic Search (Retrieval)")

    # Build the index
    chunks = chunker.chunk(document, source="help-center/password.md")
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)

    store = FaissVectorStore(dimension=embedder.dimension)
    metadata_list = [
        {"text": chunk.text, "source": chunk.source, "chunk_index": chunk.index}
        for chunk in chunks
    ]
    store.add_batch(embeddings, metadata_list)
    print(f"Indexed {store.size} chunks into FAISS\n")

    # Search
    queries = [
        "How do I change my login credentials?",
        "What phone number can I call for help?",
    ]

    for query in queries:
        print(f"Query: \"{query}\"")
        start = time.perf_counter()
        query_embedding = embedder.embed([query])[0]
        results = store.search(query_embedding, top_k=2, min_score=0.0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        for r in results:
            text_preview = r.metadata["text"][:80].replace("\n", " ")
            print(f"  Score: {r.score:.4f} | \"{text_preview}...\"")
        print(f"  ({elapsed_ms:.1f}ms)\n")

    return store


def demo_rag(embedder: EmbeddingBackend, store: FaissVectorStore):
    """Full RAG: retrieve context, then generate answer with Claude."""
    print_header("5. RAG — Retrieve & Generate")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Skipping RAG demo — ANTHROPIC_API_KEY not set.")
        print("Set it in .env to see the full RAG pipeline in action.")
        return

    question = "How do I reset my password and how long is the link valid?"
    print(f"Question: \"{question}\"\n")

    # Retrieve
    query_embedding = embedder.embed([question])[0]
    results = store.search(query_embedding, top_k=3, min_score=0.3)

    print(f"Retrieved {len(results)} chunks:")
    for r in results:
        preview = r.metadata["text"][:60].replace("\n", " ")
        print(f"  [{r.score:.4f}] \"{preview}...\"")

    # Build context
    context_parts = []
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "unknown")
        text = r.metadata.get("text", "")
        context_parts.append(f"[Source {i}: {source} (relevance: {r.score:.2f})]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    # Generate
    print("\nGenerating answer with Claude...\n")
    client = anthropic.Anthropic()
    start = time.perf_counter()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=(
            "You are a helpful assistant that answers questions based on provided documentation. "
            "Only answer based on the provided context. If the context doesn't contain enough "
            "information to answer, say so clearly. Cite which source(s) you used."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Context from documentation:\n\n{context}\n\n---\n\nQuestion: {question}",
            }
        ],
    )
    gen_ms = (time.perf_counter() - start) * 1000

    print(f"Answer ({gen_ms:.0f}ms):")
    print(f"  {response.content[0].text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("Embeddings & Vector Search Demo")
    print(f"Backend: {EMBEDDING_BACKEND}")

    embedder = create_embedding_backend()
    print(f"Dimension: {embedder.dimension}")

    demo_embed(embedder)
    print_separator()

    demo_similarity(embedder)
    print_separator()

    document, chunker = demo_chunking()
    print_separator()

    store = demo_search(embedder, document, chunker)
    print_separator()

    demo_rag(embedder, store)

    print_header("Done!")


if __name__ == "__main__":
    main()
