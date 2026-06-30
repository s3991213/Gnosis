"""
Supabase + pgvector store for Gnosis.

Manages a PostgreSQL table with vector embeddings via the pgvector extension.
Handles Supabase free-tier cold-starts: when the project is paused after
7 days of inactivity, the first connection will fail and then succeed once
the instance wakes up. A visible spinner is shown during that wait.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console

from gnosis.ingestion.chunker import Chunk

_console = Console(stderr=True)

# Errors that indicate Supabase is paused / cold-starting, not a config problem.
_WAKEUP_ERRORS = (
    "Connection refused",
    "could not connect to server",
    "connection to server",
    "SSL connection has been closed",
    "server closed the connection",
    "timeout expired",
    "Connection timed out",
)

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS gnosis_chunks (
    chunk_id   TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    source     TEXT NOT NULL,
    parent_id  TEXT,
    metadata   JSONB,
    embedding  vector({dim})
);

CREATE INDEX IF NOT EXISTS gnosis_chunks_embedding_idx
    ON gnosis_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


def _is_wakeup_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(hint in msg for hint in _WAKEUP_ERRORS)


def _connect(dsn: str, wake_timeout: int = 90) -> Any:
    """Return a live psycopg2 connection, waiting for Supabase to wake if needed."""
    import psycopg2

    deadline = time.monotonic() + wake_timeout
    waking = False

    while True:
        try:
            conn = psycopg2.connect(dsn, connect_timeout=5)
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as exc:
            if not _is_wakeup_error(exc):
                raise
            if not waking:
                waking = True
                _console.print(
                    "[bold yellow]⏳ Supabase is waking up — hang tight...[/bold yellow]"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Supabase did not respond within {wake_timeout}s. "
                    "Check your connection string or Supabase dashboard."
                ) from exc
            time.sleep(3)
        else:
            break

    if waking:
        _console.print("[bold green]✓ Supabase is ready.[/bold green]")


class SupabaseStore:
    """
    pgvector-backed vector store using Supabase (PostgreSQL).

    Parameters
    ----------
    dsn:
        PostgreSQL connection string, e.g.
        ``postgresql://user:password@host:5432/postgres``
    embedding_dim:
        Dimensionality of the embedding vectors (must match your embedder).
    wake_timeout:
        Seconds to wait for Supabase to wake from a paused state (default 90).
    """

    def __init__(
        self,
        dsn: str,
        embedding_dim: int = 384,
        wake_timeout: int = 90,
    ) -> None:
        self._dsn = dsn
        self._dim = embedding_dim
        self._wake_timeout = wake_timeout
        self._conn = _connect(dsn, wake_timeout)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_DDL.format(dim=self._dim))
        self._conn.commit()

    def _reconnect_if_needed(self) -> None:
        """Reconnect transparently if the connection was dropped."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            self._conn = _connect(self._dsn, self._wake_timeout)

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with their precomputed embeddings."""
        if not chunks:
            return
        self._reconnect_if_needed()

        import json

        rows = [
            (
                c.chunk_id,
                c.text,
                c.source,
                c.parent_id or "",
                json.dumps(c.metadata),
                embedding,
            )
            for c, embedding in zip(chunks, embeddings)
        ]

        sql = """
            INSERT INTO gnosis_chunks
                (chunk_id, text, source, parent_id, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text      = EXCLUDED.text,
                source    = EXCLUDED.source,
                parent_id = EXCLUDED.parent_id,
                metadata  = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
        """
        with self._conn.cursor() as cur:
            cur.executemany(sql, rows)
        self._conn.commit()

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Return up to *n_results* chunks nearest to *query_embedding* by cosine distance.

        Each result contains ``id``, ``text``, ``distance``, and ``metadata``.
        """
        self._reconnect_if_needed()

        sql = """
            SELECT chunk_id, text, source, metadata,
                   embedding <=> %s::vector AS distance
            FROM gnosis_chunks
            {where_clause}
            ORDER BY distance
            LIMIT %s
        """
        where_clause = ""
        params: list[Any] = [query_embedding, n_results]

        if where:
            conditions = " AND ".join(
                f"metadata->>{repr(k)} = %s" for k in where
            )
            where_clause = f"WHERE {conditions}"
            params = list(where.values()) + params

        with self._conn.cursor() as cur:
            cur.execute(sql.format(where_clause=where_clause), params)
            rows = cur.fetchall()

        import json

        return [
            {
                "id": row[0],
                "text": row[1],
                "distance": float(row[4]),
                "metadata": {**(json.loads(row[3]) if row[3] else {}), "source": row[2]},
            }
            for row in rows
        ]

    def get(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch chunks by exact ID."""
        if not chunk_ids:
            return []
        self._reconnect_if_needed()

        import json

        sql = "SELECT chunk_id, text, source, metadata FROM gnosis_chunks WHERE chunk_id = ANY(%s)"
        with self._conn.cursor() as cur:
            cur.execute(sql, (chunk_ids,))
            rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "text": row[1],
                "metadata": {**(json.loads(row[3]) if row[3] else {}), "source": row[2]},
            }
            for row in rows
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID."""
        if not chunk_ids:
            return
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM gnosis_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
        self._conn.commit()

    def count(self) -> int:
        """Return the total number of stored chunks."""
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM gnosis_chunks")
            return cur.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass
