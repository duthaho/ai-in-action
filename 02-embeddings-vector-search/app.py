"""
Embeddings & Vector Search — FastAPI Application

A complete RAG pipeline:
1. Ingest documents → chunk → embed → store in FAISS
2. Search by semantic similarity
3. Ask questions with RAG (retrieve + generate)

Supports two embedding backends:
- Voyage AI (recommended, higher quality) — requires VOYAGE_API_KEY
- Local sentence-transformers (free, no API key) — good for learning

Run: uvicorn app:app --reload --port 8001
"""

import os
import time
from contextlib import asynccontextmanager

import anthropic
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chunking import RecursiveChunker, Chunk
from vector_store import FaissVectorStore

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Embedding backend: "voyage" or "local"
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")

# Voyage AI settings
VOYAGE_MODEL = "voyage-3-large"
VOYAGE_DIMENSION = 1024

# Local model settings (sentence-transformers)
LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_DIMENSION = 384

# Chunking defaults
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# RAG settings
TOP_K_DEFAULT = 5
MIN_SIMILARITY_THRESHOLD = 0.3
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
        # Voyage API supports batches of up to 128 texts
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
# Global state
# ---------------------------------------------------------------------------

embedder: EmbeddingBackend = None
store: FaissVectorStore = None
chunker: RecursiveChunker = None
claude_client: anthropic.Anthropic = None

INDEX_DIR = "./index_data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global embedder, store, chunker, claude_client

    print(f"Initializing embedding backend: {EMBEDDING_BACKEND}")
    embedder = create_embedding_backend()
    print(f"Embedding dimension: {embedder.dimension}")

    # Load existing index or create new
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("Loading existing index...")
        store = FaissVectorStore.load(INDEX_DIR)
        print(f"Loaded {store.size} vectors")
    else:
        store = FaissVectorStore(dimension=embedder.dimension)
        print("Created new empty index")

    chunker = RecursiveChunker(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    claude_client = anthropic.Anthropic()

    yield

    # Save index on shutdown
    if store.size > 0:
        store.save(INDEX_DIR)
        print(f"Saved {store.size} vectors to {INDEX_DIR}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Embeddings & Vector Search API",
    description="RAG pipeline: ingest documents, search semantically, ask questions",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DocumentInput(BaseModel):
    """A document to ingest into the vector store."""
    content: str = Field(..., min_length=1, description="The document text")
    source: str = Field(default="", description="Document identifier (filename, URL, etc.)")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=50)
    min_score: float = Field(default=MIN_SIMILARITY_THRESHOLD, ge=0.0, le=1.0)


class AskQuery(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=20)


class SearchResultResponse(BaseModel):
    text: str
    source: str
    score: float
    chunk_index: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultResponse]
    search_time_ms: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SearchResultResponse]
    search_time_ms: float
    generation_time_ms: float


class IngestResponse(BaseModel):
    source: str
    chunks_created: int
    total_vectors: int


class EmbedResponse(BaseModel):
    """Response for the /embed endpoint — useful for learning."""
    text: str
    embedding: list[float]
    dimension: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "embedding_backend": EMBEDDING_BACKEND,
        "dimension": embedder.dimension,
        "vectors_indexed": store.size,
    }


@app.get("/stats")
async def stats():
    """Index statistics."""
    return {
        "total_vectors": store.size,
        "dimension": embedder.dimension,
        "embedding_backend": EMBEDDING_BACKEND,
        "index_persisted": os.path.exists(os.path.join(INDEX_DIR, "index.faiss")),
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(text: str):
    """
    Embed a single text and return the vector.

    Useful for understanding what embeddings look like and for debugging.
    In production, you wouldn't expose this directly.
    """
    embeddings = embedder.embed([text])
    return EmbedResponse(
        text=text,
        embedding=embeddings[0],
        dimension=len(embeddings[0]),
    )


@app.post("/documents", response_model=IngestResponse)
async def ingest_document(doc: DocumentInput):
    """
    Ingest a document: chunk it, embed each chunk, store in the vector index.

    This is the "indexing" side of the RAG pipeline.
    """
    # Step 1: Chunk the document
    chunks = chunker.chunk(doc.content, source=doc.source)

    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no chunks")

    # Step 2: Embed all chunks in one batch (efficient)
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)

    # Step 3: Store in vector index with metadata
    metadata_list = [
        {
            "text": chunk.text,
            "source": doc.source or "unknown",
            "chunk_index": chunk.index,
            **doc.metadata,
        }
        for chunk in chunks
    ]
    store.add_batch(embeddings, metadata_list)

    return IngestResponse(
        source=doc.source,
        chunks_created=len(chunks),
        total_vectors=store.size,
    )


@app.get("/documents")
async def list_documents():
    """List all unique document sources in the index."""
    sources = set()
    for meta in store.metadata_store.values():
        sources.add(meta.get("source", "unknown"))
    return {"sources": sorted(sources), "total_vectors": store.size}


@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """
    Semantic search: find the most similar chunks to the query.

    This is the "retrieval" step of RAG — no LLM involved yet.
    """
    if store.size == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest documents first.")

    start = time.perf_counter()

    # Embed the query
    query_embedding = embedder.embed([query.query])[0]

    # Search the vector store
    results = store.search(
        query_embedding=query_embedding,
        top_k=query.top_k,
        min_score=query.min_score,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return SearchResponse(
        query=query.query,
        results=[
            SearchResultResponse(
                text=r.metadata.get("text", ""),
                source=r.metadata.get("source", ""),
                score=round(r.score, 4),
                chunk_index=r.metadata.get("chunk_index", -1),
            )
            for r in results
        ],
        search_time_ms=round(elapsed_ms, 2),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(query: AskQuery):
    """
    RAG endpoint: retrieve relevant chunks, then ask Claude to answer
    based on the retrieved context.

    This is the full RAG pipeline: embed query → search → build prompt → generate.
    """
    if store.size == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest documents first.")

    # Step 1: Retrieve
    search_start = time.perf_counter()
    query_embedding = embedder.embed([query.question])[0]
    results = store.search(
        query_embedding=query_embedding,
        top_k=query.top_k,
        min_score=MIN_SIMILARITY_THRESHOLD,
    )
    search_ms = (time.perf_counter() - search_start) * 1000

    if not results:
        return AskResponse(
            question=query.question,
            answer="I couldn't find any relevant information in the indexed documents.",
            sources=[],
            search_time_ms=round(search_ms, 2),
            generation_time_ms=0,
        )

    # Step 2: Build context from retrieved chunks
    context_parts = []
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "unknown")
        text = r.metadata.get("text", "")
        context_parts.append(f"[Source {i}: {source} (relevance: {r.score:.2f})]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Generate answer with Claude
    gen_start = time.perf_counter()
    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=(
            "You are a helpful assistant that answers questions based on provided documentation. "
            "Only answer based on the provided context. If the context doesn't contain enough "
            "information to answer, say so clearly. Cite which source(s) you used."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Context from documentation:\n\n{context}\n\n"
                    f"---\n\nQuestion: {query.question}"
                ),
            }
        ],
    )
    gen_ms = (time.perf_counter() - gen_start) * 1000

    answer = response.content[0].text

    return AskResponse(
        question=query.question,
        answer=answer,
        sources=[
            SearchResultResponse(
                text=r.metadata.get("text", ""),
                source=r.metadata.get("source", ""),
                score=round(r.score, 4),
                chunk_index=r.metadata.get("chunk_index", -1),
            )
            for r in results
        ],
        search_time_ms=round(search_ms, 2),
        generation_time_ms=round(gen_ms, 2),
    )


@app.post("/compare-similarity")
async def compare_similarity(texts: list[str]):
    """
    Compare similarity between multiple texts.

    Educational endpoint: embed multiple texts and show their pairwise
    cosine similarities. Great for building intuition about embeddings.
    """
    if len(texts) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 texts")
    if len(texts) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 texts")

    embeddings = embedder.embed(texts)
    vectors = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms

    # Pairwise cosine similarity matrix
    similarity_matrix = (normalized @ normalized.T).tolist()

    # Format as pairs
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs.append({
                "text_a": texts[i],
                "text_b": texts[j],
                "similarity": round(similarity_matrix[i][j], 4),
            })

    # Sort by similarity descending
    pairs.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "pairs": pairs,
        "similarity_matrix": [
            [round(v, 4) for v in row] for row in similarity_matrix
        ],
    }
