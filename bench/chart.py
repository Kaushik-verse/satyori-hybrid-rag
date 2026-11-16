"""Draw docs/images/benchmark.svg from bench/results/*.json, so the README chart always matches the measurements.

uv run python bench/chart.py
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "docs" / "images" / "benchmark.svg"
INK, MUTED, LINE, PAPER, CARD, VERMILION = "#1C1C1E", "#64646B", "#E0D9CB", "#F6F3EC", "#FFFDF8", "#C8412B"
FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
SERIF = "'Hiragino Mincho ProN',Georgia,serif"

# Published SciFact nDCG@10 reference points (BEIR, Thakur et al. 2021, Table 2).
PUBLISHED = {"BM25 (published)": 0.665, "BM25 + rerank (published)": 0.688}


def bars(
    x: int,
    y: int,
    w: int,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, bool]],
    lo: float,
    hi: float,
    fmt: str = "{:.3f}",
) -> str:
    row_h, label_w = 34, 250
    track = w - label_w - 70
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{88 + row_h * len(rows)}" rx="16" fill="{CARD}" stroke="{LINE}"/>',
        f'<text x="{x + 24}" y="{y + 38}" font-family="{SERIF}" font-size="20" font-weight="700" fill="{INK}">'
        f"{escape(title)}</text>",
        f'<text x="{x + 24}" y="{y + 60}" font-size="12.5" fill="{MUTED}">{escape(subtitle)}</text>',
    ]
    for i, (label, value, highlight) in enumerate(rows):
        ry = y + 84 + i * row_h
        length = max(4, (value - lo) / (hi - lo) * track)
        color = VERMILION if highlight else ("#B9B3A6" if label.endswith("(published)") else "#3A3A40")
        weight = "700" if highlight else "500"
        parts += [
            f'<text x="{x + 24}" y="{ry + 15}" font-size="13" font-weight="{weight}" fill="{INK}">{escape(label)}</text>',
            f'<rect x="{x + label_w}" y="{ry + 3}" width="{track}" height="16" rx="8" fill="#F1ECE2"/>',
            f'<rect x="{x + label_w}" y="{ry + 3}" width="{length:.1f}" height="16" rx="8" fill="{color}"/>',
            f'<text x="{x + label_w + track + 10}" y="{ry + 16}" font-size="13" font-weight="{weight}" '
            f'fill="{VERMILION if highlight else INK}">{fmt.format(value)}</text>',
        ]
    return "\n".join(parts)


def main() -> None:
    scifact = json.loads((ROOT / "results" / "scifact_full.json").read_text())
    prop = json.loads((ROOT / "results" / "property.json").read_text())

    short = {
        "Dense (bge-small)": "Dense (bge-small)",
        "Hybrid RRF (dense + BM25)": "Hybrid RRF",
        "Hybrid RRF (dense + BM25 + SPLADE)": "3-way RRF",
        "Adaptive 3-way z-fusion": "Adaptive 3-way",
        "Hybrid RRF (dense + BM25) + rerank@100": "Hybrid RRF + hard rerank",
        "Adaptive 3-way z-fusion + rerank@100": "Adaptive 3-way + hard rerank",
        "Satyori: adaptive 3-way + 50/50 blended rerank@20": "Satyori pipeline",
    }
    sci_rows = [(short.get(name, name), r["nDCG@10"], False) for name, r in scifact.items()]
    sci_rows += [(name, v, False) for name, v in PUBLISHED.items()]
    sci_rows.sort(key=lambda r: r[1])
    sci_rows = [(n, v, n == "Satyori pipeline") for n, v, _ in sci_rows]

    names = {
        "sparse": "BM25",
        "dense": "Dense",
        "splade": "SPLADE++",
        "rrf": "Hybrid RRF",
        "weighted": "Weighted",
        "adaptive": "Adaptive hybrid",
        "rerank": "Hybrid + rerank",
        "rrf3": "3-way RRF",
        "adaptive3": "Adaptive 3-way",
        "satyori": "Satyori pipeline",
        "hyde": "HyDE",
        "multi_query": "RAG-Fusion",
    }
    gold = sorted(
        ((names.get(r["technique"], r["technique"]), r["mrr"], r["technique"] == "satyori") for r in prop["golden"]),
        key=lambda r: r[1],
    )

    h1 = 88 + 34 * len(sci_rows)
    h2 = 88 + 34 * len(gold)
    height = 130 + max(h1, h2) + 40
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 {height}" width="1280" height="{height}" role="img" aria-labelledby="t">
<title id="t">Benchmark results: BEIR SciFact nDCG@10 and property golden-set MRR by technique</title>
<rect width="1280" height="{height}" rx="24" fill="{PAPER}"/>
<g font-family="{FONT}">
<text x="56" y="62" font-size="13" font-weight="700" fill="{VERMILION}" letter-spacing="2.4">RESULTS</text>
<text x="56" y="98" font-family="{SERIF}" font-size="32" font-weight="700" fill="{INK}">Measured, not claimed</text>
{bars(56, 130, 600, "BEIR SciFact · nDCG@10", "Public benchmark: 5,183 abstracts, 300 queries. Grey = published.", sci_rows, 0.5, 0.8)}
{bars(680, 130, 544, "Property golden set · MRR", "30 due-diligence questions: IDs, paraphrases, both.", gold, 0.5, 1.0, "{:.2f}")}
</g>
</svg>
"""
    OUT.write_text(svg)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
