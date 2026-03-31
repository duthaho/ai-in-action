# Module 04: RAG (Retrieval-Augmented Generation)

A deep engineering guide for senior backend developers learning AI Engineering.

---

## Table of Contents

1. [Concept Explanation](#1-concept-explanation)
2. [Why It Matters in Real Systems](#2-why-it-matters-in-real-systems)
3. [Internal Mechanics](#3-internal-mechanics)
4. [Practical Example](#4-practical-example)
5. [Hands-on Implementation](#5-hands-on-implementation)
6. [System Design Perspective](#6-system-design-perspective)
7. [Common Pitfalls](#7-common-pitfalls)
8. [Advanced Topics](#8-advanced-topics)
9. [Exercises](#9-exercises)
10. [Interview / Architect Questions](#10-interview--architect-questions)

---

## 1. Concept Explanation

RAG is a pattern that **augments an LLM's generation with external knowledge retrieved at query time**. Instead of relying solely on what the model memorized during training, you fetch relevant documents from your own data store and inject them into the prompt as context.

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Retriever   │────▶│  Retrieved Docs  │────▶│     LLM     │
│ (vector DB)  │     │  (top-k chunks)  │     │ (generation)│
└─────────────┘     └──────────────────┘     └─────────────┘
                                                    │
                                                    ▼
                                              Final Answer
                                          (grounded in your data)
```

Think of it like a developer who, instead of answering from memory, first runs a search across the docs, reads the relevant sections, then gives you an informed answer.

### The Key Insight

RAG separates **knowledge** (what the system knows) from **reasoning** (how it thinks). The LLM provides reasoning; your data store provides knowledge. This means:
- Update knowledge without retraining the model
- Ground answers in verifiable sources
- Control exactly what information the model can access

---

## 2. Why It Matters in Real Systems

| Problem | How RAG Solves It |
|---|---|
| **Knowledge cutoff** — LLMs don't know your internal docs, recent data, or proprietary info | Retrieves current, private data at query time |
| **Hallucination** — LLMs confidently fabricate facts | Grounds answers in actual source documents |
| **Cost of fine-tuning** — retraining is expensive and slow | No retraining needed; just update the document store |
| **Attribution** — users need to verify claims | You can cite the exact source chunks |
| **Data freshness** — information changes daily | Re-index documents; the LLM always sees latest data |

### Where Companies Use This

| Use Case | Company Example | Why RAG |
|---|---|---|
| Customer support bots | Zendesk, Intercom | Search help center by user intent, not keywords |
| Enterprise search | Glean, Notion AI | "Find the design doc about auth migration" |
| Code search & assistance | Cursor, GitHub Copilot | RAG over your codebase for context-aware help |
| Legal / compliance | Harvey AI | Query case law databases with natural language |
| Developer tools | Internal platforms | Query runbooks, incident history, API docs |
| Healthcare | Clinical systems | Query drug interactions, clinical guidelines |

**Rule of thumb:** If your LLM needs to answer questions about data it wasn't trained on, you need RAG.

---

## 3. Internal Mechanics

### 3.1 The Two Phases

**Offline: Indexing Pipeline**
```
Raw Documents → Load → Chunk → Embed → Store in Vector DB
```

**Online: Query Pipeline**
```
User Query → Embed → Vector Search → Top-K Chunks → Build Prompt → LLM → Answer
```

### 3.2 Component Deep Dive

**Document Loading** — Ingest from various sources (files, APIs, databases). In production, this is often a scheduled pipeline (Airflow, cron) that detects changes and re-indexes only modified documents.

**Chunking** — Split documents into retrieval-friendly pieces. This is the single most impactful design decision in a RAG system.

```
Strategy: Recursive Character Splitting (most common in production)
Separator hierarchy: ["\n\n", "\n", ". ", " "]

Document: "# Auth Guide\n\nUse OAuth2...\n\nFor API keys..."
                  ↓ split on \n\n first
Chunk 1: "# Auth Guide"
Chunk 2: "Use OAuth2..."
Chunk 3: "For API keys..."
```

Why recursive works: it tries the largest separator first (paragraphs), falls back to sentences, then words. This preserves semantic coherence within chunks.

**Embedding** — Convert text to dense vectors. The query and document embeddings must use the same model and vector space.

```
similarity(q, d) = cos(θ) = (q · d) / (||q|| × ||d||)
```

Two semantically similar texts have vectors pointing in similar directions, even with zero keyword overlap.

**Vector Search** — Find the most similar document vectors to the query vector. HNSW (the dominant algorithm) achieves O(log n) search time.

**Prompt Construction** — Inject retrieved chunks into the LLM prompt with clear instructions about how to use them and when to say "I don't know."

**Generation** — The LLM reads the context and produces a grounded answer. Low temperature (0.1-0.3) reduces hallucination risk.

### 3.3 Retrieval Strategies

```
Sparse Retrieval (BM25/TF-IDF)
  → Keyword matching. Good for exact terms, error codes, names.

Dense Retrieval (Embeddings)
  → Semantic matching. Good for intent, paraphrases, concepts.

Hybrid Retrieval (Combine both)
  → Best of both worlds. Use Reciprocal Rank Fusion to merge.

Reranking (Cross-encoder)
  → Rescore top-K candidates with a more accurate model.
```

---

## 4. Practical Example

### Scenario: Internal Engineering Documentation Chatbot

Your company has 2,000+ pages of internal docs across Confluence, Notion, and GitHub wikis. Engineers waste hours searching.

**Architecture:**

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Docs (MD,  │────→│  Chunking +  │────→│  Vector DB   │
│  PDF, HTML) │     │  Embedding   │     │  (FAISS)     │
└─────────────┘     └──────────────┘     └──────────────┘
                                               │
┌─────────────┐     ┌──────────────┐           │
│  User Query │────→│  Embed Query │──── search ┘
│  via API    │     └──────────────┘
│             │            │
│             │     ┌──────────────┐     ┌──────────────┐
│             │     │  Top-K Chunks│────→│  Claude LLM  │
│             │     └──────────────┘     │  + Context   │
│             │                          └──────┬───────┘
│  Response   │←────────────────────────────────┘
└─────────────┘
```

**What happens at query time:**
1. Engineer asks: "How do I set up database connection pooling?"
2. Query is embedded into a vector
3. FAISS finds the top 5 most similar chunks (from the DB patterns doc)
4. Chunks are injected into a prompt with instructions
5. Claude synthesizes a step-by-step answer, citing sources
6. Engineer gets an accurate, sourced answer in seconds

---

## 5. Hands-on Implementation

See the accompanying code files:

- **[app.py](app.py)** — Standalone demo: full RAG pipeline (load, chunk, embed, retrieve, generate)
- **[chunking.py](chunking.py)** — Text chunking strategies (cloned from module 02)
- **[vector_store.py](vector_store.py)** — FAISS vector store abstraction (cloned from module 02)
- Sample documents in **[docs/](docs/)** — markdown files to index

### Quick Start

```bash
# From the repo root
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the demo
cd 04-rag
python app.py
```

### Pipeline Walkthrough

#### Step 1: Load Documents

```python
def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    docs = []
    for md_file in sorted(docs_dir.glob("**/*.md")):
        content = md_file.read_text(encoding="utf-8")
        docs.append((md_file.name, content))
    return docs
```

In production, you'd have loaders for PDF (PyPDF), HTML (BeautifulSoup), Notion API, Confluence API, Slack exports, database rows, etc.

#### Step 2: Chunk Documents

```python
from chunking import RecursiveChunker

chunker = RecursiveChunker(chunk_size=400, chunk_overlap=60)
chunks = chunker.chunk(document_text, source="fastapi_guide.md")
```

We reuse the `RecursiveChunker` from module 02. Key parameters:
- `chunk_size=400` — smaller than module 02's default because RAG benefits from precise, focused chunks
- `chunk_overlap=60` — prevents losing context at chunk boundaries

#### Step 3: Embed & Index

```python
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
```

#### Step 4: Retrieve

```python
query_embedding = embedder.embed([query])[0]
results = store.search(query_embedding, top_k=5, min_score=0.3)
```

The `min_score` threshold is critical — without it, the system always returns results, even when nothing is relevant.

#### Step 5: Generate with Claude

```python
# Build context from retrieved chunks
context_parts = []
for i, r in enumerate(results, 1):
    source = r.metadata.get("source", "unknown")
    text = r.metadata.get("text", "")
    context_parts.append(f"[Source {i}: {source} (relevance: {r.score:.2f})]\n{text}")
context = "\n\n---\n\n".join(context_parts)

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="Answer ONLY based on the provided context. Cite your sources.",
    messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
)
```

---

## 6. System Design Perspective

### Production Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        API Gateway                              │
│                  (Auth, Rate Limit, Routing)                    │
└──────────────┬──────────────────────────────────┬──────────────┘
               │                                  │
    ┌──────────▼──────────┐           ┌───────────▼───────────┐
    │   Ingestion Service  │           │    Query Service       │
    │                      │           │                        │
    │  • Receive documents │           │  • Embed user query    │
    │  • Detect changes    │           │  • Vector search       │
    │  • Chunk text        │           │  • Re-rank results     │
    │  • Generate embeds   │           │  • Build LLM prompt    │
    │  • Store in vector DB│           │  • Stream response     │
    │  • Track doc→chunk   │           │  • Return citations    │
    │    mappings           │           │                        │
    └──────────┬───────────┘           └───────────┬────────────┘
               │                                   │
    ┌──────────▼───────────────────────────────────▼────────────┐
    │                     Vector Database                        │
    │  (pgvector / Pinecone / Qdrant / Weaviate)                │
    │  Stores: vector + metadata + chunk text                    │
    └───────────────────────────────────────────────────────────┘
```

### Key Production Concerns

| Concern | Solution |
|---|---|
| **Latency** | Cache frequent queries in Redis; self-host embedding model to cut API latency |
| **Cost** | Use smaller embedding models; cache LLM responses; batch embedding calls |
| **Freshness** | Incremental ingestion with change detection (hash docs, only re-embed changed ones) |
| **Scale** | Managed vector DBs (Pinecone, Qdrant Cloud) handle billions of vectors |
| **Auth/Access Control** | Store permissions as chunk metadata; filter at retrieval time, not generation time |
| **Observability** | Log every query + retrieved chunks + LLM response. Track retrieval precision. |
| **Evaluation** | Build eval dataset: (question, expected_answer, expected_sources). Run on every change. |

### Scaling Decisions

| Scale | Vector Store | Why |
|---|---|---|
| < 100K vectors | FAISS in-memory | Simple, fast, no infra |
| 100K – 10M | pgvector (PostgreSQL) | SQL filtering + vectors, one DB to manage |
| 10M – 100M | Qdrant / Weaviate | Purpose-built, better performance at scale |
| 100M+ | Pinecone / custom sharding | Managed, distributed, billion-scale |

---

## 7. Common Pitfalls

### Pitfall 1: Bad Chunking (the #1 mistake)

Too large (>1000 tokens): meaning gets diluted. A chunk about "auth AND caching AND deployment" won't strongly match any topic.

Too small (<100 tokens): loses context. "Use the `--force` flag" means nothing without knowing which command.

**Fix:** 256–512 tokens with 10-15% overlap. Test with your actual queries.

### Pitfall 2: Not Evaluating Retrieval Separately

Most engineers only evaluate the final answer. But if retrieval returns the wrong chunks, the LLM **cannot** give the right answer. Always check: "Are the right chunks being retrieved?" before blaming the LLM.

**Fix:** Build a retrieval evaluation set: 50+ (query, expected_document) pairs. Measure recall@5.

### Pitfall 3: Stuffing Too Many Chunks

Retrieving 20 chunks and dumping them all in the prompt drowns the signal in noise. The "lost in the middle" problem: LLMs pay less attention to content in the middle of long contexts.

**Fix:** Start with 3-5 chunks and measure. Quality over quantity.

### Pitfall 4: No Fallback for Out-of-Scope Queries

Without explicit instructions, the LLM will try to answer everything — including questions your docs don't cover. This causes hallucination.

**Fix:** Set a similarity threshold. Instruct the model to say "I don't know" when context is insufficient.

### Pitfall 5: Embedding Model Mismatch

Using different embedding models for indexing vs. querying produces garbage results. The query and document embeddings **must** be in the same vector space.

**Fix:** Version your indexes with the model name. Re-embed everything when switching models.

### Pitfall 6: Not Handling Document Updates

Deleting a source document without removing its chunks from the vector store creates "zombie" chunks — the system cites deleted documents.

**Fix:** Track document → chunk mappings. When a document changes, delete its old chunks and re-index.

### Pitfall 7: Ignoring Metadata Filtering

Vector similarity alone isn't enough. If a user asks about "deployment in staging," you should filter chunks by `environment=staging` metadata before running vector search.

**Fix:** Always store and filter on metadata (source, date, department, access level).

---

## 8. Advanced Topics

Explore these next, in recommended order:

### 8.1 Hybrid Search (BM25 + Vector)
Combine keyword search (BM25) with vector search using Reciprocal Rank Fusion. Production RAG systems almost always use hybrid — pure vector search misses exact matches (error codes, function names), pure keyword misses semantic matches.

### 8.2 Reranking with Cross-Encoders
A bi-encoder (embedding model) encodes query and document separately — fast but less accurate. A cross-encoder processes query+document together — slow but much more accurate. Use bi-encoder for top-100, cross-encoder to re-rank to top-5.

### 8.3 Query Transformation
Before searching, rewrite the query:
- **HyDE:** Generate a hypothetical answer, embed that instead of the question
- **Multi-query:** Generate 3-5 reformulations, retrieve for each, merge results
- **Step-back prompting:** Abstract the question first

### 8.4 Agentic RAG
Give the LLM a "search" tool and let it decide when and what to retrieve. Instead of always doing one retrieval, the agent can search multiple times, refine queries, and combine results. This is how modern AI assistants work.

### 8.5 Graph RAG
Build a knowledge graph from documents, then traverse relationships during retrieval. Powerful for multi-hop questions ("Who manages the team that owns the billing service?").

### 8.6 Evaluation Frameworks
- **RAGAS:** Measures faithfulness, relevance, context precision/recall
- **Custom eval:** Build query → expected_docs test sets from real user questions
- **A/B testing:** Compare retrieval strategies with real traffic

### 8.7 Fine-tuned Embeddings
Train your embedding model on your domain's data to improve retrieval quality for specialized vocabulary.

---

## 9. Exercises

### Exercise 1: Chunking Strategy Comparison

Build a script that takes the sample documents and chunks them using 3 different strategies (fixed-size, sentence-aware, recursive from module 02's `chunking.py`). For a set of 5 test queries, compare which strategy retrieves the most relevant chunks. Output a comparison table with similarity scores.

**Success criteria:** Produce a table showing which chunking strategy performs best for your specific queries and data.

### Exercise 2: Hybrid Search Implementation

Extend the retriever to combine BM25 (keyword search) with vector search:
1. Add BM25 scoring using the `rank_bm25` library (pip install rank-bm25)
2. Implement Reciprocal Rank Fusion to merge both result sets
3. Compare hybrid vs. vector-only results for queries containing specific terms (e.g., "Alembic migration commands")

**Success criteria:** Query "alembic downgrade" returns results about database migrations from both keyword match AND semantic similarity.

### Exercise 3: RAG Evaluation Pipeline

Build an evaluation system:
1. Create 10 question-answer pairs from the sample documents (ground truth)
2. Run each question through your RAG pipeline
3. Measure: (a) Was the correct chunk retrieved? (b) Does the generated answer match?
4. Output precision, recall, and a per-question breakdown

**Success criteria:** A JSON report showing retrieval and generation quality metrics.

---

## 10. Interview / Architect Questions

### Q1: Your RAG system retrieves relevant chunks but the LLM still gives wrong answers. How do you debug this?

**What this tests:** Systematic debugging of the generation phase when retrieval is working correctly.

**Key points:** Examine the prompt template — is the LLM being instructed clearly? Check for "lost in the middle" effect — reorder chunks by relevance. Check if the answer requires synthesizing across multiple chunks (harder for LLMs). Try reducing chunk count. Try a more capable model. Check temperature setting.

### Q2: How would you implement access control in a multi-tenant RAG system where users should only see documents they're authorized to access?

**What this tests:** Security architecture in AI systems.

**Key points:** Store tenant/permission metadata on every chunk at ingestion time. Apply metadata filter before vector search (`WHERE tenant_id = ? AND access_level <= ?`). Never rely on the LLM to filter — it's a security boundary that must be enforced at the retrieval layer. Consider row-level security if using pgvector.

### Q3: Your document corpus is 10M documents and growing 50K/day. Queries must return in under 500ms. Design the retrieval architecture.

**What this tests:** System design under scale and latency constraints.

**Key points:** Two-stage retrieval: fast ANN index (HNSW) for top-100 in ~50ms, cross-encoder reranker for top-5. Async ingestion via queue (Kafka/SQS). Incremental indexing — only embed new/changed docs. Self-hosted embedding model to cut API latency. Cache hot queries. Monitor p99 latency.

### Q4: When would you choose fine-tuning over RAG, and when would you use both together?

**What this tests:** Understanding the trade-offs between knowledge injection approaches.

**Key points:** RAG for factual grounding in specific, changing data. Fine-tuning for teaching the model a new style, format, or domain vocabulary. Use both together when you need domain-specific reasoning AND current data. Example: a legal AI fine-tuned on legal reasoning + RAG over case law. RAG alone fails when the model can't reason about domain concepts; fine-tuning alone fails when facts change.

### Q5: How do you evaluate whether your RAG system is actually working well in production?

**What this tests:** Production observability and evaluation maturity.

**Key points:** Automated metrics: retrieval precision@k, context recall, faithfulness, answer relevancy. Operational metrics: latency p50/p95/p99, token cost per query, cache hit rate. Human-in-the-loop: thumbs up/down, track reformulated queries (signal of poor first answer). Build a golden dataset of 100+ Q&A pairs and run regression tests on every pipeline change. A/B test against baselines.
