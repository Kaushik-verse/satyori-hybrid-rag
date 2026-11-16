"""Settings, read once from environment variables, an optional .env file, or Streamlit secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "eval" / "golden_set.json"

load_dotenv(PROJECT_ROOT / ".env")


def _secret(name: str) -> str | None:
    """Environment first, then Streamlit Cloud secrets (without crashing when no secrets file exists)."""
    if os.getenv(name):
        return os.getenv(name)
    try:
        import streamlit as st

        return st.secrets.get(name) or None
    except Exception:  # no secrets.toml, or not running under Streamlit
        return None


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    answer_model: str
    council_models: tuple[str, ...]
    embedding_model: str
    reranker_model: str
    splade_model: str | None  # learned-sparse channel; None turns it off (saves ~0.5 GB of RAM)
    chunk_size: int
    chunk_overlap: int

    @property
    def llm_enabled(self) -> bool:
        return bool(self.groq_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    council = os.getenv("SATYORI_COUNCIL_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b")
    return Settings(
        groq_api_key=_secret("GROQ_API_KEY"),
        answer_model=os.getenv("SATYORI_MODEL", "openai/gpt-oss-120b"),
        council_models=tuple(m.strip() for m in council.split(",") if m.strip()),
        embedding_model=os.getenv("SATYORI_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker_model=os.getenv("SATYORI_RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        splade_model=os.getenv("SATYORI_SPLADE_MODEL", "prithivida/Splade_PP_en_v1") or None,
        chunk_size=int(os.getenv("SATYORI_CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("SATYORI_CHUNK_OVERLAP", "1")),
    )
