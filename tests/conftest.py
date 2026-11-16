"""Offline stand-ins for the embedding model, reranker and LLM, so tests need no network or API key."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from satyori.index import Index, tokenize
from satyori.ingest import Document


class HashEmbedder:
    """Bag-of-words hashed into 256 dims: crude, deterministic, good enough to test the plumbing."""

    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1  # noqa: S324 - not security
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class OverlapReranker:
    """Scores by shared tokens; returns -10 when nothing overlaps, like a real cross-encoder on noise."""

    def score(self, query, texts):
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) * 2.0 - 10 + (10 if q & set(tokenize(t)) else 0) for t in texts])


class ScriptedLLM:
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[tuple[str, str, str | None]] = []

    def complete(self, system, user, *, model=None, temperature=0.1, max_tokens=900):
        self.calls.append((system, user, model))
        return self.replies.pop(0) if self.replies else "ok"


DOCS = {
    "Agreement": "# Agreement\n\n## 1. Possession\n\nThe Promoter shall hand over possession on or before "
    "30 June 2027. "
    "Delay interest is SBI MCLR plus two percent.\n\n## 2. Cancellation\n\nOn cancellation the Promoter may forfeit "
    "10 percent of the consideration as earnest money.",
    "Brochure": "# Brochure\n\n## 1. Possession\n\nMove in by December 2026 at our lakeside towers.\n\n## 2. Pets\n\n"
    "A large pet park for dogs and cats.",
    "Title report": "# Title report\n\n## 1. Litigation\n\nAn active civil suit, Case Ref CS-2023-8891, claims a 15% "
    "undivided share in the land.\n\n## 2. Easement\n\nThe neighbours have perpetual access to a shared borewell.",
}


def make_doc(name: str, text: str) -> Document:
    return Document(name, [text], hashlib.sha256(text.encode()).hexdigest()[:16])


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def index(embedder):
    idx = Index(embedder)
    for name, text in DOCS.items():
        idx.add(make_doc(name, text), chunk_size=300)
    return idx


class TokenSplade:
    """SPLADE stand-in: each token's hash is a vocabulary id with weight 1."""

    @staticmethod
    def _vec(text):
        return {int(hashlib.md5(t.encode()).hexdigest(), 16) % 30000: 1.0 for t in tokenize(text)}  # noqa: S324

    def encode_passages(self, texts):
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return self._vec(text)


@pytest.fixture
def splade_index(embedder):
    idx = Index(embedder, TokenSplade())
    for name, text in DOCS.items():
        idx.add(make_doc(name, text), chunk_size=300)
    return idx
