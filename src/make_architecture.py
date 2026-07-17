"""Generates docs/architecture.png — a clean pipeline diagram."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture.png")

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")

INK = "#1c2330"; BLUE = "#58a6ff"; MUTED = "#57606a"
COLORS = {"raw": "#3fb950", "light": "#d29922", "heavy": "#8b5cf6"}

def box(x, y, w, h, text, fc="#eef2f7", ec=BLUE, tc=INK, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                 linewidth=1.6, edgecolor=ec, facecolor=fc))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=16, linewidth=1.5, color=MUTED))

# input
box(0.2, 3.6, 1.9, 1.0, "Raw text\n(a turn / note)", fc="#f6f8fa", ec=MUTED, fs=9)
arrow(2.1, 4.1, 2.7, 4.1)

# compressor
box(2.7, 3.4, 2.4, 1.4, "COMPRESSOR\n\nextractive (TF-IDF)\n+ importance score", ec=BLUE, bold=False, fs=9)
arrow(5.1, 4.1, 5.7, 4.1)

# store
box(5.7, 2.6, 4.0, 2.2, "", ec=BLUE, fc="#f0f6ff")
ax.text(7.7, 4.5, "MEMORY STORE", ha="center", fontsize=10, fontweight="bold", color=INK)
box(5.95, 3.7, 1.05, 0.55, "raw", fc="#e9f9ee", ec=COLORS["raw"], fs=8)
box(7.15, 3.7, 1.05, 0.55, "light", fc="#fbf3e2", ec=COLORS["light"], fs=8)
box(8.35, 3.7, 1.05, 0.55, "heavy", fc="#f2ecfd", ec=COLORS["heavy"], fs=8)
ax.text(7.7, 3.25, "hierarchical decay:\nnewest = vivid · oldest = gist", ha="center",
        va="center", fontsize=8, color=MUTED)
ax.text(7.7, 2.78, "TF-IDF vectors · cosine index", ha="center",
        va="center", fontsize=8, color=MUTED, style="italic")

# query path
box(2.7, 0.7, 2.4, 1.0, "Query\n\"what's the budget?\"", fc="#f6f8fa", ec=MUTED, fs=9)
arrow(5.1, 1.2, 6.2, 1.2); arrow(6.2, 1.2, 6.2, 2.5)
ax.text(5.65, 1.4, "retrieve top-k", ha="center", fontsize=8, color=MUTED)

# output
box(6.9, 0.7, 2.8, 1.0, "Relevant memories\nranked by similarity", fc="#eef9f0",
    ec=COLORS["raw"], fs=9)
arrow(7.7, 2.5, 8.0, 1.72)

ax.set_title("AI Digital Memory Compression Engine — Architecture",
             fontsize=13, fontweight="bold", color=INK, pad=10)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor="white")
print("Saved", OUT)
