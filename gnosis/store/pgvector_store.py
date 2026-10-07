"""
Postgres + pgvector store for Gnosis.

Manages a PostgreSQL table with vector embeddings via the pgvector extension.
Works with any Postgres that has pgvector: the local Docker container or a
hosted one such as Supabase. Connections are retried while the database is
unreachable, which covers a container that is still starting and a Supabase
free-tier project waking from a pause. A visible spinner is shown during that wait.
"""

from __future__ import annotations

import atexit
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console

from gnosis.ingestion.chunker import Chunk

_console = Console(stderr=True)

# Errors that indicate the database is still starting / waking, not a config problem.
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

ALTER TABLE gnosis_chunks
    ADD COLUMN IF NOT EXISTS text_search tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

CREATE INDEX IF NOT EXISTS gnosis_chunks_embedding_idx
    ON gnosis_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS gnosis_chunks_text_search_idx
    ON gnosis_chunks USING GIN (text_search);

-- Full original text of notes written in Gnosis (not files), so they can be
-- edited exactly; their chunks above are derived from this.
CREATE TABLE IF NOT EXISTS gnosis_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gnosis_notes (
    source     TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _is_wakeup_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(hint in msg for hint in _WAKEUP_ERRORS)


def _connect(dsn: str, wake_timeout: int = 90) -> Any:
    """Return a live psycopg2 connection, waiting for the database to come up if needed."""
    import psycopg2

    deadline = time.monotonic() + wake_timeout
    waking = False

    while True:
        try:
            conn = psycopg2.connect(dsn, connect_timeout=5)
            # Autocommit so reads don't leave the session "idle in transaction"
            # (holding locks and blocking VACUUM). Multi-statement writes opt
            # into a transaction via PgVectorStore._transaction().
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        except psycopg2.OperationalError as exc:
            if not _is_wakeup_error(exc):
                raise
            if not waking:
                waking = True
                _console.print(
                    "[bold yellow]⏳ Waiting for the database — hang tight...[/bold yellow]"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Database did not respond within {wake_timeout}s. "
                    "Is it running? (`docker compose up db`) Otherwise check DATABASE_URL."
                ) from exc
            time.sleep(3)
        else:
            if waking:
                _console.print("[bold green]✓ Database is ready.[/bold green]")
            return conn


class PgVectorStore:
    """
    pgvector-backed vector store on PostgreSQL.

    Parameters
    ----------
    dsn:
        PostgreSQL connection string, e.g.
        ``postgresql://user:password@host:5432/postgres``
    embedding_dim:
        Dimensionality of the embedding vectors (must match your embedder).
    wake_timeout:
        Seconds to wait for the database to become reachable (default 90).
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
        atexit.register(self.close)

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_DDL.format(dim=self._dim))

    def _reconnect_if_needed(self) -> None:
        """Reconnect transparently if the connection was dropped."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            self._conn = _connect(self._dsn, self._wake_timeout)

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        """Run the block's statements atomically: commit on success, roll back on error."""
        self._conn.autocommit = False
        try:
            with self._conn.cursor() as cur:
                yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._conn.autocommit = True

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float] | None]) -> None:
        """Insert or update chunks with their precomputed embeddings.

        An embedding may be None (parent chunks, which are never vector-searched).
        """
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
        with self._transaction() as cur:
            cur.executemany(sql, rows)

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

        where_clause = ""
        where_params: list[Any] = []

        if where:
            conditions = []
            for k, v in where.items():
                conditions.append("metadata->>%s = %s")
                where_params.extend([k, v])
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT chunk_id, text, source, metadata, parent_id,
                   embedding <=> %s::vector AS distance
            FROM gnosis_chunks
            {where_clause}
            ORDER BY distance
            LIMIT %s
        """
        # Param order matches placeholder order: embedding first, where values, then limit
        params: list[Any] = [query_embedding] + where_params + [n_results]

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "text": row[1],
                "distance": float(row[5]),
                "metadata": {**(row[3] or {}), "source": row[2], "parent_id": row[4] or None},
            }
            for row in rows
        ]

    def keyword_search(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Return up to *n_results* chunks ranked by Postgres full-text search.

        Serves as the keyword leg of hybrid search, backed by the
        ``text_search`` tsvector column so no local index is needed.
        Each result contains ``id``, ``text``, ``score``, and ``metadata``.
        """
        self._reconnect_if_needed()

        where_clause = ""
        where_params: list[Any] = []

        if where:
            conditions = []
            for k, v in where.items():
                conditions.append("metadata->>%s = %s")
                where_params.extend([k, v])
            where_clause = "AND " + " AND ".join(conditions)

        sql = f"""
            SELECT chunk_id, text, source, metadata, parent_id,
                   ts_rank_cd(text_search, query) AS rank
            FROM gnosis_chunks, plainto_tsquery('english', %s) query
            WHERE text_search @@ query
            {where_clause}
            ORDER BY rank DESC
            LIMIT %s
        """
        params: list[Any] = [query] + where_params + [n_results]

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "text": row[1],
                "score": float(row[5]),
                "metadata": {**(row[3] or {}), "source": row[2], "parent_id": row[4] or None},
            }
            for row in rows
        ]

    def existing_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return the subset of *chunk_ids* already present in the store."""
        if not chunk_ids:
            return set()
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM gnosis_chunks WHERE chunk_id = ANY(%s)",
                (chunk_ids,),
            )
            return {row[0] for row in cur.fetchall()}

    def get(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch chunks by exact ID."""
        if not chunk_ids:
            return []
        self._reconnect_if_needed()

        sql = "SELECT chunk_id, text, source, metadata FROM gnosis_chunks WHERE chunk_id = ANY(%s)"
        with self._conn.cursor() as cur:
            cur.execute(sql, (chunk_ids,))
            rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "text": row[1],
                "metadata": {**(row[3] or {}), "source": row[2]},
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

    def delete_stale(self, source: str, keep_ids: list[str]) -> int:
        """
        Remove chunks from *source* whose IDs are not in *keep_ids*.

        Called after re-ingesting a source so edited content doesn't leave
        old chunks behind. Returns the number of chunks deleted.
        """
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM gnosis_chunks WHERE source = %s AND NOT (chunk_id = ANY(%s))",
                (source, keep_ids),
            )
            deleted = cur.rowcount
        return deleted

    def count(self) -> int:
        """Return the total number of stored chunks."""
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM gnosis_chunks")
            return cur.fetchone()[0]

    def count_sources(self) -> int:
        """Return the number of distinct sources (files, URLs, notes) stored."""
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT source) FROM gnosis_chunks")
            return cur.fetchone()[0]

    def list_sources(self) -> list[dict]:
        """Return one row per source — ``source``, ``type``, ``chunks`` — sorted by name."""
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, MAX(metadata->>'type'), COUNT(*)
                FROM gnosis_chunks
                GROUP BY source
                ORDER BY lower(source)
                """
            )
            rows = cur.fetchall()
        return [{"source": r[0], "type": r[1] or "", "chunks": r[2]} for r in rows]

    def source_passages(self, source: str) -> list[str]:
        """Return the parent passages of *source* in document order."""
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT text FROM gnosis_chunks
                WHERE source = %s AND metadata->>'chunk_type' = 'parent'
                ORDER BY COALESCE((metadata->>'page')::int, 0),
                         COALESCE((metadata->>'position')::int, 0)
                """,
                (source,),
            )
            return [row[0] for row in cur.fetchall()]

    def get_setting(self, key: str) -> str | None:
        """Return a saved setting (e.g. the chosen LLM model), or None."""
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute("SELECT value FROM gnosis_settings WHERE key = %s", (key,))
            row = cur.fetchone()
        return row[0] if row else None

    def set_setting(self, key: str, value: str) -> None:
        """Save a setting, replacing any previous value."""
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO gnosis_settings (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (key, value),
            )

    def save_note(self, source: str, content: str) -> None:
        """Store the full text of a note written in Gnosis (insert or replace)."""
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO gnosis_notes (source, content) VALUES (%s, %s)
                ON CONFLICT (source) DO UPDATE
                    SET content = EXCLUDED.content, updated_at = now()
                """,
                (source, content),
            )

    def get_note(self, source: str) -> str | None:
        """Return the stored full text of a note, or None if it isn't one."""
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute("SELECT content FROM gnosis_notes WHERE source = %s", (source,))
            row = cur.fetchone()
        return row[0] if row else None

    def source_type(self, source: str) -> str | None:
        """Return the ``type`` metadata of *source* (``note``, ``markdown``, ``pdf``, …)."""
        self._reconnect_if_needed()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT metadata->>'type' FROM gnosis_chunks WHERE source = %s LIMIT 1",
                (source,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def delete_source(self, source: str) -> int:
        """Remove every chunk of *source* and its note text. Returns chunks deleted."""
        self._reconnect_if_needed()
        with self._transaction() as cur:
            cur.execute("DELETE FROM gnosis_chunks WHERE source = %s", (source,))
            deleted = cur.rowcount
            cur.execute("DELETE FROM gnosis_notes WHERE source = %s", (source,))
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass
