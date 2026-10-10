# Architecture

Everything runs locally: two small models in-process, an LLM over HTTP, and one Postgres database that holds both the vector index and the full-text index.

```mermaid
flowchart TB
    subgraph entry["Entry points"]
        files["Your files<br/>Markdown · PDF · text · URLs<br/>inbox/ · examples/notes"]
        cli["gnosis CLI<br/>menu · add · search · ask · serve"]
        clients["MCP clients<br/>Claude Desktop · OpenWebUI · agents"]
        mcp["MCP server (FastMCP)<br/>search_memory · ask_question<br/>summarize_topic · find_connections<br/>add_note"]
        clients -- "stdio or SSE :8000" --> mcp
    end

    subgraph ingest["Ingestion · write path"]
        direction LR
        loader["Loader<br/>md · txt · url<br/>PDF page by page"]
        chunker["Chunker<br/>parent ≈1500 ch<br/>child ≈300 ch, 50 overlap"]
        embedW["Embed<br/>new chunks only"]
        upsert["Upsert<br/>delete stale chunks per file"]
        loader --> chunker --> embedW --> upsert
    end

    subgraph retrieve["Retrieval · read path"]
        direction TB
        query["Query<br/>follow-ups include the last question"]
        hyde["HyDE (optional)<br/>gnosis search --hyde only"]
        subgraph hybrid["Hybrid search · 20 candidates per leg, child chunks"]
            kw["Keyword leg<br/>tsvector · GIN"]
            vec["Vector leg<br/>cosine · HNSW"]
            rrf["RRF k=60<br/>swap child → parent passage"]
            kw --> rrf
            vec --> rrf
        end
        rerank["Rerank<br/>top n · drop scores below −8"]
        answer["Answer<br/>numbered passages → LLM → cites [1] [2]"]
        query --> kw
        query --> vec
        query -.-> hyde -.-> vec
        rrf --> rerank --> answer
    end

    subgraph pg["Postgres 17 + pgvector (Docker :5433 or Supabase)"]
        chunks[("gnosis_chunks<br/>embedding vector(384) HNSW<br/>text_search tsvector GIN")]
        notes[("gnosis_notes<br/>editable note text")]
        settings[("gnosis_settings<br/>chosen LLM model")]
    end

    subgraph models["Local models"]
        emb["Embedder<br/>all-MiniLM-L6-v2 · in-process"]
        ce["Cross-encoder<br/>ms-marco-MiniLM-L-6-v2 · in-process"]
        llm["LLM server (HTTP)<br/>Ollama :11434 · LM Studio :1234<br/>/v1/chat/completions"]
    end

    files --> loader
    cli -- add --> loader
    mcp -- add_note --> chunker
    cli -- "search · ask" --> query
    mcp -- "search · ask · summarize" --> query

    upsert --> chunks
    mcp -- add_note --> notes
    kw --> chunks
    vec --> chunks
    embedW --- emb
    vec --- emb
    rerank --- ce
    answer --- llm
    hyde -.- llm

    classDef write fill:#fbf1e9,stroke:#b4581c,color:#1b2124
    classDef read fill:#e9f4f1,stroke:#0e7468,color:#1b2124
    classDef store fill:#eaf0f7,stroke:#2d5786,color:#1b2124
    classDef model fill:#f1edf8,stroke:#6b4aa8,color:#1b2124
    class loader,chunker,embedW,upsert write
    class query,hyde,kw,vec,rrf,rerank,answer read
    class chunks,notes,settings store
    class emb,ce,llm model
```

## Asking a question

1. The question (plus the previous one, for follow-ups) is embedded with MiniLM.
2. Postgres runs a cosine search and a full-text search over child chunks, 20 hits each.
3. Reciprocal Rank Fusion merges the two lists, and each child is replaced by its parent passage.
4. The cross-encoder rescores the passages, keeps the top 5 and drops anything below −8.
5. The LLM answers only from the numbered passages and cites them inline.

HyDE is used by `gnosis search --hyde` only. `find_connections` uses the vector search alone, with no keyword search or reranking.

## Adding notes

1. A loader turns a file or URL into documents; PDFs become one document per page.
2. Each document splits into ~1500-character parents and ~300-character children.
3. IDs already in `gnosis_chunks` are skipped; only new chunks get embedded.
4. New chunks are upserted, and chunks from an older version of the same file are deleted.

Notes written in the menu or via `add_note` also keep their full text in `gnosis_notes` so they can be edited later.

## Docker stack

| Service       | Role                                                        |
| ------------- | ----------------------------------------------------------- |
| `db`          | `pgvector/pgvector:pg17`, host port `127.0.0.1:5433`        |
| `ollama`      | LLM server on the internal network                          |
| `ollama-pull` | One-shot download of `qwen2.5:1.5b`                         |
| `gnosis`      | Seeds `examples/notes`, then serves MCP over SSE on `:8000` |

Outside Docker, set `DATABASE_URL` and `LLM_BASE_URL` in `.env`.
