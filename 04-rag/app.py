"""
RAG (Retrieval-Augmented Generation) — Standalone Demo
=======================================================

Walks through the complete RAG pipeline step by step:
1. Load documents from a directory
2. Chunk documents into retrieval-friendly pieces
3. Embed chunks and build a FAISS index
4. Retrieve relevant chunks for a query
5. Generate a grounded answer with Claude using retrieved context

Reuses chunking.py and vector_store.py from module 02.

Supports two embedding backends:
- Voyage AI (higher quality) — requires VOYAGE_API_KEY
- Local sentence-transformers (free, no API key) — good for learning

Run: python app.py
"""

import os
import time
from pathlib import Path

import anthropic
import numpy as np
from dotenv import load_dotenv

from chunking import RecursiveChunker, Chunk
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

DOCS_DIR = Path(__file__).parent / "docs"

# RAG tuning knobs
CHUNK_SIZE = 400          # chars — smaller chunks = more precise retrieval
CHUNK_OVERLAP = 60        # overlap to preserve context at boundaries
TOP_K = 5                 # number of chunks to retrieve
MIN_SIMILARITY = 0.3      # filter out low-relevance results


# ---------------------------------------------------------------------------
# Embedding backends (same pattern as module 02)
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
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_separator():
    print(f"\n{'-' * 60}\n")


# ---------------------------------------------------------------------------
# RAG Pipeline Components
# ---------------------------------------------------------------------------

def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """
    Load all markdown files from a directory.

    Returns list of (filename, content) tuples.

    In production you'd have loaders for PDF, HTML, Notion API,
    Confluence API, database rows, Slack exports, etc.
    """
    docs = []
    for md_file in sorted(docs_dir.glob("**/*.md")):
        content = md_file.read_text(encoding="utf-8")
        docs.append((md_file.name, content))
    return docs


def build_index(
    docs: list[tuple[str, str]],
    embedder: EmbeddingBackend,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[FaissVectorStore, list[Chunk]]:
    """
    The INDEXING phase of RAG (offline pipeline):
    Documents → Chunk → Embed → Store in FAISS

    Returns the vector store and the list of chunks (for inspection).
    """
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: list[Chunk] = []

    for filename, content in docs:
        chunks = chunker.chunk(content, source=filename)
        all_chunks.extend(chunks)

    # Embed all chunks in one batch
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed(texts)

    # Build FAISS index
    store = FaissVectorStore(dimension=embedder.dimension)
    metadata_list = [
        {"text": chunk.text, "source": chunk.source, "chunk_index": chunk.index}
        for chunk in all_chunks
    ]
    store.add_batch(embeddings, metadata_list)

    return store, all_chunks


def retrieve(
    query: str,
    store: FaissVectorStore,
    embedder: EmbeddingBackend,
    top_k: int = TOP_K,
    min_score: float = MIN_SIMILARITY,
):
    """
    The RETRIEVAL phase of RAG (online pipeline):
    Query → Embed → Vector Search → Top-K chunks

    Returns search results sorted by similarity.
    """
    query_embedding = embedder.embed([query])[0]
    results = store.search(query_embedding, top_k=top_k, min_score=min_score)
    return results


def generate_answer(query: str, results: list, client: anthropic.Anthropic) -> str:
    """
    The GENERATION phase of RAG (online pipeline):
    Retrieved chunks + Query → Prompt → Claude → Grounded answer

    The system prompt is the most critical piece — it tells the LLM
    how to use the retrieved context and when to say "I don't know."
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "unknown")
        text = r.metadata.get("text", "")
        context_parts.append(
            f"[Source {i}: {source} (relevance: {r.score:.2f})]\n{text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant that answers questions based on provided documentation. "
        "RULES:\n"
        "1. Answer ONLY based on the provided context. Do not use prior knowledge.\n"
        "2. If the context doesn't contain enough information, say: "
        "\"I don't have enough information in the available documents to answer this.\"\n"
        "3. Cite which source document(s) you used.\n"
        "4. Be concise and direct."
    )

    user_message = (
        f"Context from documentation:\n\n{context}\n\n---\n\n"
        f"Question: {query}"
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------

def demo_load_and_chunk():
    """Step 1 & 2: Load documents and chunk them."""
    print_header("1. Load & Chunk Documents")

    docs = load_documents(DOCS_DIR)
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}/:")
    for filename, content in docs:
        print(f"  {filename} ({len(content)} chars)")

    chunker = RecursiveChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    total_chunks = 0
    for filename, content in docs:
        chunks = chunker.chunk(content, source=filename)
        total_chunks += len(chunks)
        print(f"\n  {filename} → {len(chunks)} chunks:")
        for chunk in chunks[:3]:  # Show first 3
            preview = chunk.text[:80].replace("\n", " ")
            print(f"    Chunk {chunk.index}: \"{preview}...\"")
        if len(chunks) > 3:
            print(f"    ... and {len(chunks) - 3} more")

    print(f"\nTotal: {len(docs)} docs → {total_chunks} chunks")
    print(f"Settings: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    return docs


def demo_build_index(docs, embedder):
    """Step 3: Embed chunks and build the FAISS index."""
    print_header("2. Embed & Index")

    start = time.perf_counter()
    store, all_chunks = build_index(docs, embedder)
    elapsed = time.perf_counter() - start

    print(f"Indexed {store.size} chunks into FAISS ({elapsed:.1f}s)")
    print(f"Vector dimension: {embedder.dimension}")
    print(f"Index type: Flat (brute-force inner product)")

    # Show a sample embedding
    sample_text = all_chunks[0].text[:60].replace("\n", " ")
    sample_embedding = embedder.embed([all_chunks[0].text])[0]
    print(f"\nSample chunk: \"{sample_text}...\"")
    print(f"Embedding (first 5 values): {[round(v, 4) for v in sample_embedding[:5]]}")

    return store, all_chunks


def demo_retrieval(store, embedder):
    """Step 4: Semantic retrieval — search the index."""
    print_header("3. Retrieval — Finding Relevant Chunks")

    queries = [
        "How should I handle database connection pooling?",
        "What's the best way to deploy a FastAPI application?",
        "How do I set up health checks for my service?",
    ]

    for query in queries:
        print(f"Query: \"{query}\"")
        start = time.perf_counter()
        results = retrieve(query, store, embedder, top_k=3)
        elapsed_ms = (time.perf_counter() - start) * 1000

        for r in results:
            source = r.metadata["source"]
            text_preview = r.metadata["text"][:70].replace("\n", " ")
            print(f"  [{r.score:.4f}] ({source}) \"{text_preview}...\"")

        if not results:
            print("  No results above similarity threshold.")
        print(f"  ({elapsed_ms:.1f}ms)\n")


def demo_rag(store, embedder):
    """Step 5: Full RAG — retrieve + generate with Claude."""
    print_header("4. Full RAG — Retrieve & Generate")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Skipping RAG generation — ANTHROPIC_API_KEY not set.")
        print("Set it in .env to see the full pipeline in action.")
        return

    client = anthropic.Anthropic()

    questions = [
        "How do I configure database connection pooling for a high-traffic service?",
        "What logging practices should I follow in production?",
    ]

    for question in questions:
        print(f"Question: \"{question}\"\n")

        # Retrieve
        start = time.perf_counter()
        results = retrieve(question, store, embedder, top_k=TOP_K)
        retrieve_ms = (time.perf_counter() - start) * 1000

        print(f"Retrieved {len(results)} chunks ({retrieve_ms:.0f}ms):")
        for r in results:
            source = r.metadata["source"]
            preview = r.metadata["text"][:60].replace("\n", " ")
            print(f"  [{r.score:.4f}] ({source}) \"{preview}...\"")

        if not results:
            print("  No relevant context found. Skipping generation.")
            print()
            continue

        # Generate
        print("\nGenerating answer with Claude...")
        gen_start = time.perf_counter()
        answer = generate_answer(question, results, client)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        print(f"\nAnswer ({gen_ms:.0f}ms):")
        print(f"  {answer}")
        print_separator()


def demo_comparison(docs, embedder):
    """Bonus: Compare retrieval quality with different chunk sizes."""
    print_header("5. Chunk Size Comparison")

    query = "How do I handle database migrations safely?"
    print(f"Query: \"{query}\"\n")

    for size in [200, 400, 800]:
        store, _ = build_index(docs, embedder, chunk_size=size, chunk_overlap=size // 8)
        results = retrieve(query, store, embedder, top_k=3)

        print(f"chunk_size={size}:")
        for r in results:
            source = r.metadata["source"]
            text_preview = r.metadata["text"][:60].replace("\n", " ")
            print(f"  [{r.score:.4f}] ({source}) \"{text_preview}...\"")
        print()

    print("Notice how chunk size affects which content gets retrieved and")
    print("at what similarity score. Smaller chunks are more precise but")
    print("may lose surrounding context. This is the #1 tuning knob in RAG.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header("RAG (Retrieval-Augmented Generation) Demo")
    print(f"Backend: {EMBEDDING_BACKEND}")
    print(f"Docs directory: {DOCS_DIR}")

    if not DOCS_DIR.exists() or not list(DOCS_DIR.glob("*.md")):
        print(f"\nNo markdown files found in {DOCS_DIR}/.")
        print("Add .md files to that directory and re-run.")
        return

    embedder = create_embedding_backend()
    print(f"Dimension: {embedder.dimension}")

    docs = demo_load_and_chunk()
    print_separator()

    store, all_chunks = demo_build_index(docs, embedder)
    print_separator()

    demo_retrieval(store, embedder)
    print_separator()

    demo_rag(store, embedder)

    demo_comparison(docs, embedder)

    print_header("Done!")


if __name__ == "__main__":
    main()
