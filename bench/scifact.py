"""Benchmark Satyori's retrieval on BEIR SciFact, a public dataset with published baselines.

    uv run python bench/scifact.py            # downloads the dataset once (~3 MB), then runs
    uv run python bench/scifact.py --quick    # first 100 test queries only

Every channel's scores are cached in bench/.cache, so an interrupted run resumes where it stopped.
Results are written to bench/results/scifact.json and printed as a table.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from satyori.config import get_settings  # noqa: E402
from satyori.index import BM25, FastEmbedder, SpladeEncoder, tokenize  # noqa: E402
from satyori.retrieval import FastReranker, route, zscore  # noqa: E402

DATA, CACHE, RESULTS = ROOT / "data" / "scifact", ROOT / ".cache", ROOT / "results"
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
RERANK_DEPTH = 100


def download() -> None:
    if (DATA / "corpus.jsonl").exists():
        return
    print("Downloading BEIR SciFact…")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    zipfile.ZipFile(io.BytesIO(resp.content)).extractall(DATA.parent)


def load(limit: int | None):
    corpus = [json.loads(line) for line in (DATA / "corpus.jsonl").read_text().splitlines()]
    ids = [c["_id"] for c in corpus]
    texts = [f"{c['title']}. {c['text']}".strip() for c in corpus]
    all_q = {(q := json.loads(line))["_id"]: q["text"] for line in (DATA / "queries.jsonl").read_text().splitlines()}
    qrels: dict[str, dict[str, int]] = {}
    for line in (DATA / "qrels" / "test.tsv").read_text().splitlines()[1:]:
        qid, doc, rel = line.split()
        if int(rel) > 0:
            qrels.setdefault(qid, {})[doc] = int(rel)
    queries = [(q, all_q[q]) for q in qrels][:limit]
    return ids, texts, queries, qrels


def cached(name: str, fn):
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path, allow_pickle=True)
    t = time.time()
    value = np.asarray(fn())
    np.save(path, value, allow_pickle=True)
    print(f"  {name}: {time.time() - t:.0f}s")
    return value


def sharded(name: str, texts: list[str], encode, size: int = 32) -> list:
    """Encode documents 32 at a time, saving each shard. Keeps memory low on an 8 GB laptop (fastembed's default
    batch of 256 long passages needs ~3 GB for attention alone) and lets an interrupted run resume."""
    out: list = []
    t = time.time()
    for start in range(0, len(texts), size):
        path = CACHE / f"{name}_{start:05d}.pkl"
        if path.exists():
            part = pickle.loads(path.read_bytes())  # noqa: S301 - our own cache
        else:
            part = encode(texts[start : start + size])
            path.write_bytes(pickle.dumps(part))
        out.extend(part)
        if start % (size * 20) == 0:
            print(f"  {name}: {start + len(part)}/{len(texts)} ({time.time() - t:.0f}s)", flush=True)
    return out


def rrf(mats: list[np.ndarray], k: int = 60, depth: int = 100) -> np.ndarray:
    out = np.zeros_like(mats[0])
    for m in mats:
        order = np.argsort(-m, axis=1)[:, :depth]
        for i in range(len(m)):
            out[i, order[i]] += 1 / (k + np.arange(1, depth + 1))
    return out


def evaluate(scores: np.ndarray, ids, queries, qrels) -> dict[str, float]:
    ndcg, mrr, recall = [], [], []
    for i, (qid, _) in enumerate(queries):
        rel = qrels[qid]
        ranked = [ids[j] for j in np.argsort(-scores[i], kind="stable")[:100]]
        dcg = sum(rel.get(d, 0) / math.log2(r + 2) for r, d in enumerate(ranked[:10]))
        ideal = sum(v / math.log2(r + 2) for r, v in enumerate(sorted(rel.values(), reverse=True)[:10]))
        ndcg.append(dcg / ideal)
        first = next((r for r, d in enumerate(ranked[:10]) if d in rel), None)
        mrr.append(0 if first is None else 1 / (first + 1))
        recall.append(len(set(ranked) & set(rel)) / len(rel))
    return {"nDCG@10": float(np.mean(ndcg)), "MRR@10": float(np.mean(mrr)), "Recall@100": float(np.mean(recall))}


def main() -> None:
    quick = "--quick" in sys.argv
    for d in (CACHE, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    download()
    ids, texts, queries, qrels = load(100 if quick else None)
    tag = "q100" if quick else "full"
    s = get_settings()
    qtexts = [q for _, q in queries]
    print(f"SciFact: {len(texts)} documents, {len(queries)} queries")

    def bm25():
        index = BM25([tokenize(t) for t in texts])
        return [index.scores(tokenize(q)) for q in qtexts]

    def dense():
        emb = FastEmbedder(s.embedding_model)
        docs = np.array(sharded("dense_docs", texts, lambda batch: list(emb.embed_passages(batch))))
        return np.array([emb.embed_query(q) for q in qtexts]) @ docs.T

    def splade():
        enc = SpladeEncoder(s.splade_model or "prithivida/Splade_PP_en_v1")
        inverted: dict[int, list[tuple[int, float]]] = {}
        for j, vec in enumerate(sharded("splade_docs", texts, enc.encode_passages)):
            for term, w in vec.items():
                inverted.setdefault(term, []).append((j, w))
        out = np.zeros((len(qtexts), len(texts)))
        for i, q in enumerate(qtexts):
            for term, w in enc.encode_query(q).items():
                for j, dw in inverted.get(term, []):
                    out[i, j] += w * dw
        return out

    B = cached(f"bm25_{tag}", bm25)
    D = cached(f"dense_{tag}", dense)
    S = cached(f"splade_{tag}", splade)
    alpha = np.array([route(q)[0] for q in qtexts])[:, None]
    z = lambda m: np.array([zscore(row) for row in m])  # noqa: E731
    first_stage = {
        "BM25": B,
        "Dense (bge-small)": D,
        "SPLADE++": S,
        "Hybrid RRF (dense + BM25)": rrf([D, B]),
        "Hybrid RRF (dense + BM25 + SPLADE)": rrf([D, B, S]),
        "Adaptive 3-way z-fusion": 2 * alpha * z(D) + (1 - alpha) * (z(B) + z(S)),
    }
    results = {name: evaluate(m, ids, queries, qrels) for name, m in first_stage.items()}

    reranker = None
    for base in ("Hybrid RRF (dense + BM25)", "Adaptive 3-way z-fusion"):
        key = f"ce_{tag}_{base.split()[0].lower()}"
        cand = np.argsort(-first_stage[base], axis=1)[:, :RERANK_DEPTH]

        def ce(cand=cand):
            nonlocal reranker
            reranker = reranker or FastReranker(s.reranker_model)
            return [reranker.score(q, [texts[j] for j in row]) for q, row in zip(qtexts, cand, strict=True)]

        scores = cached(key, ce)
        out = np.full_like(first_stage[base], -1e9)
        for i, row in enumerate(cand):
            out[i, row] = scores[i]
        results[f"{base} + rerank@{RERANK_DEPTH}"] = evaluate(out, ids, queries, qrels)

    # The Satyori pipeline: rerank the top 20 (as the app does) and blend 50/50 with the fusion score.
    base = first_stage["Adaptive 3-way z-fusion"]
    ce = cached(f"ce_{tag}_adaptive", lambda: None)
    cand = np.argsort(-base, axis=1)[:, :RERANK_DEPTH]
    out = base - 1e6  # passages below the top 20 keep their fusion order, beneath the reranked ones
    for i, row in enumerate(cand):
        c, f = np.asarray(ce[i][:20], dtype=float), base[i, row[:20]]
        out[i, row[:20]] = 0.5 * (c - c.mean()) / (c.std() or 1) + 0.5 * (f - f.mean()) / (f.std() or 1)
    results["Satyori: adaptive 3-way + 50/50 blended rerank@20"] = evaluate(out, ids, queries, qrels)

    (RESULTS / f"scifact_{tag}.json").write_text(json.dumps(results, indent=2))
    width = max(map(len, results))
    print(f"\n{'technique':{width}}  nDCG@10  MRR@10  Recall@100")
    for name, r in results.items():
        print(f"{name:{width}}  {r['nDCG@10']:.3f}    {r['MRR@10']:.3f}   {r['Recall@100']:.3f}")


if __name__ == "__main__":
    main()
