"""
benchmark.py — measures the compression-vs-retention tradeoff.

Sweeps how aggressively memories are compressed (keep_ratio from 1.0 down
to 0.2) and, at each level, records:
    - compression ratio (tokens kept / original)
    - fact retention    (how many QA-probe facts survived compression)

Produces benchmark_results.csv. Pure standard library — runs anywhere.
Run:  python benchmark.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# import the engine from the pure-python single file
from engine import (MemoryStore, compress, extractive_compress,
                    keyword_recall, CONVERSATION, PROBES, split_sentences)


def retention_at(keep_ratio: float) -> dict:
    """Compress every turn at a fixed keep_ratio, then check fact survival."""
    compressed_texts = []
    orig_tokens = comp_tokens = 0
    for turn in CONVERSATION:
        c = extractive_compress(turn, keep_ratio=keep_ratio)
        compressed_texts.append(c)
        orig_tokens += max(1, int(len(turn.split()) / 0.75))
        comp_tokens += max(1, int(len(c.split()) / 0.75))

    blob = " ".join(compressed_texts)
    kept = sum(1 for p in PROBES
               if keyword_recall(blob, p["answer_keywords"]) >= 0.5)

    return {
        "keep_ratio": keep_ratio,
        "compression_ratio": round(comp_tokens / orig_tokens, 3),
        "fact_retention": round(kept / len(PROBES), 3),
        "facts_kept": kept,
        "facts_total": len(PROBES),
    }


def main():
    ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    rows = [retention_at(r) for r in ratios]

    out = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
    with open(out, "w") as f:
        f.write("keep_ratio,compression_ratio,fact_retention,facts_kept,facts_total\n")
        for r in rows:
            f.write(f"{r['keep_ratio']},{r['compression_ratio']},"
                    f"{r['fact_retention']},{r['facts_kept']},{r['facts_total']}\n")

    print(f"{'keep':>5} {'comp_ratio':>11} {'retention':>10} {'facts':>8}")
    print("-" * 38)
    for r in rows:
        print(f"{r['keep_ratio']:>5} {r['compression_ratio']:>11} "
              f"{r['fact_retention']*100:>9.1f}% "
              f"{r['facts_kept']}/{r['facts_total']:>3}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
