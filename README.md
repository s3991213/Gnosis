# 🧠 Gnosis

> A local, private second brain powered by RAG and MCP. Ask your notes anything.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io/)

---

## What is Gnosis?

Gnosis turns your local files — Markdown notes, PDFs, web bookmarks — into a queryable knowledge base that runs entirely on your machine. No cloud. No subscriptions. Your data never leaves.

It exposes a **Model Context Protocol (MCP) server** so any MCP-compatible client (Claude Desktop, OpenWebUI, etc.) can query your knowledge base using natural language tools.

---

## Features

- 📥 **Ingest anything** — Markdown, PDF, plain text, web URLs
- 🔍 **Hybrid search** — vector similarity (ChromaDB) + BM25 keyword search
- 🧩 **Advanced RAG** — parent-child chunking, HyDE queries, cross-encoder reranking
- 🔌 **MCP server** — expose your brain as tools any LLM agent can use
- 🏠 **Fully local** — runs on Ollama or LM Studio (Qwen3, Llama 3, etc.)
- ⚡ **Incremental ingestion** — only re-embeds files that have changed
- 🖥️ **CLI** — `gnosis add`, `gnosis search`, `gnosis serve`

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    MCP Clients                      │
│         (Claude Desktop · OpenWebUI · agents)       │
└──────────────────────┬──────────────────────────────┘
                       │ MCP Protocol
┌──────────────────────▼──────────────────────────────┐
│                  MCP Server (FastMCP)                │
│   search_memory · add_note · summarize_topic        │
│              find_connections                        │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 Retrieval Pipeline                   │
│   Query → HyDE → Hybrid Search → Reranker → LLM    │
└────────┬─────────────────────────┬──────────────────┘
         │                         │
┌────────▼──────────┐   ┌──────────▼──────────────────┐
│    ChromaDB        │   │      BM25 Index              │
│  (vector store)    │   │   (keyword search)           │
└────────┬──────────┘   └─────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────┐
│               Ingestion Pipeline                   │
│   Loader → Chunker → Embedder → Deduplication     │
│        (Markdown · PDF · URL · plain text)        │
└───────────────────────────────────────────────────┘
```

---

## Project Structure

```
gnosis/
├── ingestion/
│   ├── loader.py          # File/URL loaders
│   ├── chunker.py         # Parent-child chunking strategy
│   ├── embedder.py        # sentence-transformers wrapper
│   └── deduplicator.py    # Hash-based change detection
├── store/
│   ├── chroma_store.py    # ChromaDB wrapper
│   └── bm25_store.py      # BM25 index
├── retrieval/
│   ├── searcher.py        # Hybrid search (vector + BM25)
│   ├── hyde.py            # HyDE query expansion
│   └── reranker.py        # Cross-encoder reranking
├── mcp_server/
│   ├── server.py          # FastMCP server entry point
│   └── tools.py           # Tool definitions + schemas
├── llm/
│   └── client.py          # Ollama / LM Studio adapter
├── cli/
│   └── main.py            # Typer CLI
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/yourusername/gnosis.git
cd gnosis
pip install -e ".[dev]"
```

### 2. Start a local LLM

```bash
# Option A: Ollama
ollama pull qwen3:14b

# Option B: LM Studio
# Load any GGUF model and start the local server
```

### 3. Ingest your notes

```bash
# Add a single file
gnosis add ~/notes/meeting.md

# Add a whole folder
gnosis add ~/notes/

# Add a PDF
gnosis add ~/docs/paper.pdf

# Add a URL
gnosis add https://example.com/article
```

### 4. Search your brain

```bash
gnosis search "what did I learn about transformer attention?"
```

### 5. Start the MCP server

```bash
gnosis serve
# MCP server running on http://localhost:8000
```

---

## MCP Tools

Once the server is running, any MCP-compatible client can use these tools:

| Tool | Description |
|------|-------------|
| `search_memory` | Semantic search across all ingested content |
| `add_note` | Add a new note directly via the MCP client |
| `summarize_topic` | Summarize everything Gnosis knows about a topic |
| `find_connections` | Surface related notes you may have forgotten |

### Example (Claude Desktop)

```
You: What do I know about RAG reranking?

Claude: [calls search_memory("RAG reranking")]
        Based on your notes from 3 sources...
```

---

## RAG Design Decisions

### Why parent-child chunking?
Standard fixed-size chunking loses context at boundaries. Gnosis indexes small child chunks for precision retrieval, but returns the larger parent chunk to the LLM for richer context.

### Why HyDE?
Direct embedding of short queries often mismatches document embeddings. HyDE generates a hypothetical answer first, embeds that, and uses it as the search vector — significantly improving recall on vague or short queries.

### Why hybrid search?
Pure vector search misses exact keyword matches (names, acronyms, code snippets). BM25 handles these well. Gnosis uses Reciprocal Rank Fusion (RRF) to merge both result sets.

### Why a cross-encoder reranker?
Bi-encoder retrieval is fast but imprecise. A cross-encoder (`ms-marco-MiniLM`) re-scores the top-k candidates with full attention over (query, document) pairs — much more accurate for final ranking.

---

## Docker

```bash
docker compose up
```

Starts ChromaDB + Gnosis MCP server. Point your MCP client at `http://localhost:8000`.

---

## Roadmap

- [ ] Web UI (optional, lightweight)
- [ ] Memory decay / recency weighting
- [ ] Auto-tagging on ingestion via LLM
- [ ] GraphRAG mode (Neo4j backend)
- [ ] Obsidian vault watcher plugin

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Ollama / LM Studio (local) |
| MCP server | FastMCP |
| CLI | Typer |
| PDF parsing | PyMuPDF |

---

## License

MIT