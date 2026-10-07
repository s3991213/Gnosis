"""
Application-level configuration and lazy singletons for Gnosis.

Reads environment variables (from .env if present) and provides
cached factory functions for all major components.  Importing this
module is side-effect-free; components are initialised on first access.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIM = 384

# Matches the Postgres started by `docker compose up db`; override for Supabase.
DEFAULT_DATABASE_URL = "postgresql://postgres:gnosis@localhost:5433/postgres"


@lru_cache(maxsize=1)
def get_embedder():
    from gnosis.ingestion.embedder import Embedder

    return Embedder()


@lru_cache(maxsize=1)
def get_vector_store():
    from gnosis.store.pgvector_store import PgVectorStore

    return PgVectorStore(
        dsn=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        embedding_dim=EMBEDDING_DIM,
    )


@lru_cache(maxsize=1)
def get_searcher():
    from gnosis.retrieval.searcher import HybridSearcher

    return HybridSearcher(
        vector_store=get_vector_store(),
        embedder=get_embedder(),
    )


@lru_cache(maxsize=1)
def get_reranker():
    from gnosis.retrieval.reranker import Reranker

    return Reranker()


LLM_MODEL_SETTING = "llm_model"


def get_llm_model() -> str:
    """The model picked in the menu (saved in the database), else ``LLM_MODEL``."""
    from gnosis.llm.client import LLMClient

    try:
        saved = get_vector_store().get_setting(LLM_MODEL_SETTING)
    except Exception:
        saved = None
    return saved or os.getenv("LLM_MODEL", LLMClient.DEFAULT_MODEL)


def get_llm_client():
    """Not cached: the model can be changed from the menu while the server runs."""
    from gnosis.llm.client import LLMClient

    return LLMClient(
        base_url=os.getenv("LLM_BASE_URL", LLMClient.DEFAULT_BASE_URL),
        model=get_llm_model(),
    )
