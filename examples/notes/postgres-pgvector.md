# Postgres as a vector database (pgvector)

`pgvector` adds a `vector(n)` column type and distance operators to Postgres:

- `<->` Euclidean (L2) distance
- `<=>` cosine distance
- `<#>` negative inner product

## Indexes
- **HNSW**: better recall/speed trade-off, slower to build, works on an empty table.
- **IVFFlat**: faster to build, but needs data in the table first to pick its lists.

## Why bother instead of a dedicated vector DB?
Keeping embeddings next to the text means one transaction, one backup, and you can mix vector
search with ordinary SQL filters and full-text search (`tsvector` + GIN index) in the same
query. Fewer moving parts to keep in sync.

Gotcha: the vector dimension is fixed by the column type, so switching embedding models means
a migration and re-embedding everything.
