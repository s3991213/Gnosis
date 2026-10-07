# Gnosis

> A private second brain powered by RAG and MCP. Ask your notes anything.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io/)

---

## What is Gnosis?

Gnosis turns your local files — Markdown notes, PDFs, web bookmarks — into a queryable knowledge base. Everything runs on your machine: embedding, reranking, the LLM, and the Postgres + pgvector database that holds your notes. Point it at a hosted Postgres (e.g. Supabase) instead and the same knowledge base is reachable from any machine.

It exposes a **Model Context Protocol (MCP) server** so any MCP-compatible client (Claude Desktop, OpenWebUI, etc.) can query your knowledge base using natural language tools.

---

## Features

- **Ingest anything** — Markdown, PDF, plain text, web URLs
- **Hybrid search** — vector similarity + full-text keyword search, both in Postgres/pgvector
- **Advanced RAG** — parent-child chunking, HyDE queries, cross-encoder reranking
- **MCP server** — expose your brain as tools any LLM agent can use
- **Local models** — embeddings and reranking run on-device; LLM runs on Ollama or LM Studio (Qwen3, Llama 3, etc.)
- **Incremental ingestion** — only re-embeds chunks that have changed, and removes chunks from earlier versions of a re-ingested file
- **Terminal menu + CLI** — run `gnosis` for an interactive menu, or `gnosis add`, `gnosis search`, `gnosis ask`, `gnosis serve`

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    MCP Clients                      │
│         (Claude Desktop · OpenWebUI · agents)       │
└──────────────────────┬──────────────────────────────┘
                       │ MCP Protocol
┌──────────────────────▼──────────────────────────────┐
│                  MCP Server (FastMCP)               │
│   search_memory · add_note · summarize_topic        │
│         ask_question · find_connections             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 Retrieval Pipeline                  │
│   Query → HyDE → Hybrid Search → Reranker → LLM     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│     Postgres + pgvector (local Docker / Supabase)   │
│   vector similarity  ·  full-text keyword search    │
│   gnosis_chunks (search)  ·  gnosis_notes (edits)   │
└──────────────────────▲──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│                 Ingestion Pipeline                  │
│      Loader → Chunker → Embedder → Upsert           │
│        (Markdown · PDF · URL · plain text)          │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── gnosis/
│   ├── ingestion/
│   │   ├── loader.py          # File/URL loaders
│   │   ├── chunker.py         # Parent-child chunking strategy
│   │   └── embedder.py        # sentence-transformers wrapper
│   ├── store/
│   │   └── pgvector_store.py  # Postgres/pgvector wrapper (vector + full-text + dedup)
│   ├── retrieval/
│   │   ├── searcher.py        # Hybrid search (vector + full-text)
│   │   ├── hyde.py            # HyDE query expansion
│   │   └── reranker.py        # Cross-encoder reranking
│   ├── mcp_server/
│   │   ├── server.py          # FastMCP server entry point
│   │   └── tools.py           # Tool definitions + schemas
│   ├── llm/
│   │   └── client.py          # Ollama / LM Studio adapter
│   ├── cli/
│   │   ├── main.py            # Typer CLI
│   │   ├── menu.py            # Interactive menu (`gnosis` with no command)
│   │   └── formatting.py      # Note rendering: Markdown + ==highlight== / ++underline++
│   └── config.py              # Env config + lazy component factories
├── tests/
├── examples/notes/            # Sample notes, auto-ingested by Docker
├── docker/entrypoint.sh       # Seeds sample notes, then starts the server
├── inbox/                     # Drop files here to import them from the menu (Docker)
├── gn                         # One-command launcher: starts Docker, opens the menu
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Quickstart

### Option 1: Docker (zero setup)

The only requirement is [Docker](https://docs.docker.com/get-docker/). You don't install Postgres, Ollama, or Python — each runs in its own container, and `docker compose down -v` removes all of it.

From the project folder:

```bash
./gn
```

That's it. `./gn` starts everything in the background (the first time, it builds the image, which takes a few minutes) and opens the Gnosis menu:

```
╭───────────────────────────── Gnosis ──────────────────────────────╮
│                                                                   │
│  5 notes in your knowledge base   ● LLM online                    │
│                                                                   │
╰──────────────────────── your second brain ────────────────────────╯
  (↑/↓ to move, Enter to choose)
 » Ask a question              answers from your notes, with sources
   Search notes                find anything by meaning
   Browse all notes            see everything you've saved
   Write a new note            type it right here
   Add a file, folder or URL   Markdown, text, PDF, web pages
   Summarize a topic           uses the local LLM
   Change AI model             pick or download a different LLM
   ---------------
   Quit
```

Move with the arrow keys and press Enter. **Ask a question** answers from your notes using the local LLM, streaming the answer as it's written and citing the passages it used as `[1]`, `[2]` (with page numbers for PDFs). Follow-up questions keep the conversation's context; press Enter on an empty line to finish. If nothing relevant is found it says so rather than guessing.

Search results and the note list are navigable the same way: Enter opens a result or note in full, any key goes back, and Ctrl+C backs out one level. Each screen starts at the top of the window, and earlier output stays in your scrollback. Browsing shows how many notes you have, pages through them to fit your window, and can filter by name.

Notes you write in Gnosis can be edited or deleted from **Browse all notes** (open a note, then **Edit note**). The editor is a full multi-line editor: arrow keys move the cursor, Enter starts a new line, and **Ctrl+S** saves (or Esc then Enter). Notes are Markdown, with two extras for the terminal:

| Type | Renders as |
|------|-----------|
| `**bold**`, `*italic*`, `~~strike~~` | **bold**, *italic*, ~~strike~~ |
| `++underline++` | underlined |
| `==highlight==` | highlighted (black on yellow) |
| `# Heading`, `- bullet`, `1. numbered`, `` `code` `` | headings, lists, code |

Notes imported from files or URLs are read-only in Gnosis — edit the original and import it again.

Importing shows a progress screen (reading, splitting, embedding) with time remaining. Each batch is saved as soon as it's embedded, so an interrupted import picks up where it stopped when you run it again. As a rough guide, a 60-page PDF takes about 10 seconds on CPU. PDFs need selectable text; scanned pages are images and have nothing to import.

**Change AI model** lists the models you've downloaded plus recommended ones with their download sizes. Picking a new one downloads it with a progress bar. The choice is saved in the database, so it survives restarts and the MCP server uses it too. The default `qwen2.5:1.5b` keeps the first download small; `qwen2.5:7b` gives noticeably better summaries if Docker has ~6 GB of memory to spare.

Try searching for `tips for concentrating`: it finds the Deep Work book notes by meaning, even though they never use the words "tips" or "concentrating."

To add your own files, drop them in the `inbox/` folder and choose **Add a file, folder or URL**, then press Enter. `./gn` also runs any CLI command, e.g. `./gn search "sourdough"`. On Windows without a shell, use `docker compose up -d` then `docker compose exec gnosis gnosis`.

The stack it starts:

| Service | What it does |
|---------|--------------|
| `db` | Postgres 17 + pgvector |
| `ollama` | Local LLM server (pulls `qwen2.5:1.5b`, ~1 GB, on first run) |
| `gnosis` | MCP server on `http://localhost:8000/sse`, pre-loaded with the [sample notes](examples/notes/) |

No accounts, no API keys, no `.env`. The embedding and reranker models are baked into the image, so later starts take seconds.

Stop with `docker compose down` (add `-v` to wipe the database and downloaded model).

### Option 2: Run locally with Python

Use Docker just for the database, and run Gnosis itself from source:

```bash
pip install -e ".[dev]"
docker compose up -d db        # Postgres + pgvector on localhost:5433
```

Postgres runs inside that container, so there's nothing to install. No configuration needed either: `DATABASE_URL` defaults to it. (No Docker at all? Use a hosted database, Option 3.) For the LLM features (`--hyde`, `ask`, `summarize_topic`), run [Ollama](https://ollama.com) or LM Studio on your machine:

```bash
ollama pull qwen3:14b          # or any model; set LLM_MODEL to match
```

Search and ingestion work without an LLM.

```bash
# Ingest: a file, a folder, a PDF, or a URL
gnosis add examples/notes/
gnosis add ~/docs/paper.pdf
gnosis add https://example.com/article

# Search
gnosis search "what did I learn about transformer attention?"

# Ask a question (answered by the local LLM, with cited sources)
gnosis ask "what is a rational agent?"

# Or just open the menu
gnosis

# MCP server
gnosis serve                   # stdio — for Claude Desktop, which launches the process itself
gnosis serve --transport sse   # SSE/HTTP on http://localhost:8000 — for network clients
```

### Option 3: Hosted database (Supabase)

To share one knowledge base across machines, point Gnosis at any Postgres with pgvector, e.g. a free [Supabase](https://supabase.com) project. Copy `.env.example` to `.env` and set:

```
DATABASE_URL=postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres
```

Use the **direct connection** or the **session pooler** (both port 5432), not the transaction pooler (port 6543), since Gnosis holds a long-lived connection. The direct connection is IPv6-only on Supabase's free tier, and Docker containers often lack IPv6, so from Docker use the session pooler string and set it as `DOCKER_DATABASE_URL`.

In every case the `vector` extension, the `gnosis_chunks` table, and its HNSW and full-text (GIN) indexes are created automatically on first connect. On Supabase's free tier, projects pause after 7 days of inactivity; the first query after a pause waits for it to wake.

### Configuration

All settings are optional.

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `postgresql://postgres:gnosis@localhost:5433/postgres` | Postgres + pgvector connection |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama / LM Studio endpoint |
| `LLM_MODEL` | `qwen3:14b` | Model for HyDE, questions and `summarize_topic` (a model picked in the menu overrides this) |

The Docker stack reads `DOCKER_DATABASE_URL`, `DOCKER_LLM_BASE_URL`, `DOCKER_LLM_MODEL`, `GNOSIS_PORT`, `GNOSIS_DB_PORT` and `GNOSIS_SEED_DIR` (set it empty to skip the sample notes) instead, so a `.env` meant for local runs can't point the containers at `localhost`.

---

## MCP Tools

Once the server is running, any MCP-compatible client can use these tools:

| Tool | Description |
|------|-------------|
| `search_memory` | Semantic search across all ingested content |
| `add_note` | Add a new note directly via the MCP client |
| `summarize_topic` | Summarize everything Gnosis knows about a topic |
| `ask_question` | Answer a question from your notes with the local LLM, citing sources |
| `find_connections` | Surface related notes you may have forgotten |

### Connect an AI client

**Claude Desktop, with Docker running** (`docker compose up`): Claude Desktop launches MCP servers as local commands, so use the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) bridge to reach the server in the container. This needs [Node.js](https://nodejs.org) for `npx`.

Open **Settings → Developer → Edit Config** and add:

```json
{
  "mcpServers": {
    "gnosis": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/sse"]
    }
  }
}
```

Restart Claude Desktop. Gnosis's tools appear in the chat's tools menu, and Claude calls them when your question is about your notes.

**Claude Desktop, running from source** (Option 2): skip Docker for the server and let Claude Desktop start Gnosis itself over stdio. Use the full path to the `gnosis` command (`which gnosis`), since Claude Desktop doesn't inherit your shell's `PATH`:

```json
{
  "mcpServers": {
    "gnosis": {
      "command": "/full/path/to/gnosis",
      "args": ["serve"]
    }
  }
}
```

**Other clients** (OpenWebUI, agent frameworks): point them at the SSE endpoint `http://localhost:8000/sse`.

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
Pure vector search misses exact keyword matches (names, acronyms, code snippets). Postgres full-text search (`tsvector`/`ts_rank_cd`) handles these well and lives in the same Postgres table as the embeddings — no separate index to keep in sync. Gnosis uses Reciprocal Rank Fusion (RRF) to merge both result sets.

### Why a cross-encoder reranker?
Bi-encoder retrieval is fast but imprecise. A cross-encoder (`ms-marco-MiniLM`) re-scores the top-k candidates with full attention over (query, document) pairs — much more accurate for final ranking.

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
| Vector store | Postgres + pgvector (Docker locally, or Supabase) |
| Keyword search | Postgres full-text search (`tsvector` + GIN) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Ollama / LM Studio (local) |
| MCP server | FastMCP |
| CLI | Typer |
| PDF parsing | PyMuPDF |
