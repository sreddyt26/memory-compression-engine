"""
make_chart.py — renders the compression-vs-retention tradeoff chart.

Reads benchmark_results.csv (produced by benchmark.py) and saves
docs/tradeoff.png. This is the one step that needs matplotlib; the
benchmark itself is pure-python. If matplotlib is blocked on your machine,
the CSV + README table already tell the story.

Run:  python make_chart.py
"""
import csv, os

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "benchmark_results.csv")
OUT = os.path.join(HERE, "..", "docs", "tradeoff.png")


def main():
    keep, comp, ret = [], [], []
    with open(CSV) as f:
        for row in csv.DictReader(f):
            keep.append(float(row["keep_ratio"]))
            comp.append(float(row["compression_ratio"]) * 100)
            ret.append(float(row["fact_retention"]) * 100)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(comp, ret, "-o", color="#58a6ff", linewidth=2.2,
            markersize=7, label="fact retention")
    for x, y, k in zip(comp, ret, keep):
        ax.annotate(f"keep={k}", (x, y), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color="#7d8590")

    ax.set_xlabel("Compression ratio  (tokens kept ÷ original, %)  →  smaller = more compressed")
    ax.set_ylabel("Fact retention (%)")
    ax.set_title("Compression vs. Fact Retention Tradeoff", fontweight="bold", pad=14)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(50, 105)
    ax.invert_xaxis()  # more compression to the right
    ax.legend(loc="lower left", frameon=False)

    # shade the "sweet spot" region
    ax.axhspan(70, 105, alpha=0.05, color="green")
    ax.text(comp[3], 101, "usable region (≥70% retention)",
            fontsize=8, color="#2f7d32")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
