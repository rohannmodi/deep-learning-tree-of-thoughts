"""
Plot stratified results across IO / CoT / ToT methods on Game of 24.

The key analytical move: bin puzzles by *intrinsic difficulty* using the
behaviour of the simpler methods as the difficulty signal:

  Tier 1 — IO solves it:                 "trivial" (single-shot suffices)
  Tier 2 — IO fails, CoT solves:         "needs reasoning"
  Tier 3 — IO fails AND CoT fails:       "needs search" (paper's hard subset)

This is the same difficulty stratification the paper uses (puzzles 901-1000
were chosen specifically because IO+CoT have very low accuracy on them).
"""
import os
import numpy as np
import matplotlib.pyplot as plt


GAME24_PUZZLES = [
    "1 2 3 4", "4 5 6 7", "5 6 7 8", "2 2 4 6", "4 9 10 13",
    "1 1 8 8", "3 3 7 7", "4 4 10 10", "3 5 6 8", "2 8 8 12",
    "2 3 7 11", "5 5 5 5", "1 3 4 6", "3 3 8 8", "1 5 5 5",
]

# Per-puzzle outcomes — Gemini 2.5 Flash, n=15 (from /tmp/game24_15.log)
GAME24 = {
    "IO":      [1, 0, 0, 0, 0,  0, 0, 1, 0, 0,  0, 0, 1, 0, 1],  # 4/15
    "CoT":     [1, 0, 1, 1, 1,  0, 0, 0, 1, 1,  1, 1, 0, 0, 0],  # 8/15
    "ToT-BFS": [1, 1, 1, 1, 0,  1, 0, 0, 0, 1,  1, 0, 0, 0, 0],  # 7/15
}
GAME24_AVG_TIME  = {"IO": 0.7,  "CoT": 4.0,  "ToT-BFS": 51.5}
GAME24_AVG_CALLS = {"IO": 1.0,  "CoT": 1.0,  "ToT-BFS": 68.5}


# ─────────────────────────────────────────────────────────────────────────────
# Stratification
# ─────────────────────────────────────────────────────────────────────────────
def stratify(outcomes: dict[str, list[int]], n_total: int) -> dict[str, list[int]]:
    """Return tier index (0=trivial, 1=needs-reasoning, 2=needs-search) per puzzle."""
    tiers = []
    for i in range(n_total):
        if outcomes["IO"][i] == 1:
            tiers.append(0)              # trivial
        elif outcomes["CoT"][i] == 1:
            tiers.append(1)              # needs reasoning
        else:
            tiers.append(2)              # needs search
    return tiers


def accuracy_by_tier(outcomes: dict[str, list[int]], tiers: list[int]) -> dict:
    methods = list(outcomes.keys())
    by_tier = {t: {m: [] for m in methods} for t in (0, 1, 2)}
    for i, t in enumerate(tiers):
        for m in methods:
            by_tier[t][m].append(outcomes[m][i])
    accs = {t: {m: 100 * sum(v) / len(v) if v else 0 for m, v in by_tier[t].items()}
            for t in (0, 1, 2)}
    counts = {t: {m: (sum(by_tier[t][m]), len(by_tier[t][m])) for m in methods}
              for t in (0, 1, 2)}
    return accs, counts


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
COLORS  = ["#4C72B0", "#55A868", "#C44E52"]
TIER_LABELS = [
    "Tier 1 — Trivial\n(IO solves)",
    "Tier 2 — Needs reasoning\n(IO fails, CoT solves)",
    "Tier 3 — Needs search\n(IO + CoT both fail)",
]


def overall_bar(ax, outcomes, n_total):
    methods = list(outcomes.keys())
    accs = [100 * sum(outcomes[m]) / n_total for m in methods]
    bars = ax.bar(methods, accs, color=COLORS, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall accuracy on all 15 puzzles", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, m in zip(bars, methods):
        n = sum(outcomes[m])


def tiered_bar(ax, accs, counts):
    methods = list(accs[0].keys())
    n_methods = len(methods)
    n_tiers   = 3
    x = np.arange(n_tiers)
    width = 0.26
    for i, m in enumerate(methods):
        ys = [accs[t][m] for t in range(n_tiers)]
        bars = ax.bar(x + (i - 1) * width, ys, width,
                      color=COLORS[i], label=m, edgecolor="black", linewidth=0.5)
        for j, bar in enumerate(bars):
            n_ok, n_tot = counts[j][m]
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABELS, fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy stratified by puzzle difficulty",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Highlight Tier 3 (where ToT's value lies)
    ax.axvspan(1.5, 2.5, color="gold", alpha=0.08)
    ax.text(2, 105, "← ToT's territory",
            ha="center", fontsize=9, fontweight="bold", color="#8B6F00")


def heatmap(ax, outcomes, tiers, puzzles):
    methods = list(outcomes.keys())
    # Reorder columns by tier so the heatmap groups visually
    order = sorted(range(len(puzzles)), key=lambda i: (tiers[i], i))
    matrix = np.array([[outcomes[m][i] for i in order] for m in methods])
    cmap = plt.matplotlib.colors.ListedColormap(["#E8736C", "#7CB97C"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks(range(len(puzzles)))
    ax.set_xticklabels([puzzles[i] for i in order], rotation=60, fontsize=7.5, ha="right")
    ax.set_title("Per-puzzle outcomes (sorted by difficulty tier)",
                 fontweight="bold")
    for i in range(len(methods)):
        for j in range(len(puzzles)):
            ax.text(j, i, "✓" if matrix[i, j] else "✗",
                    ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
    # Vertical separators between tiers
    sorted_tiers = [tiers[i] for i in order]
    for k in range(1, len(sorted_tiers)):
        if sorted_tiers[k] != sorted_tiers[k - 1]:
            ax.axvline(k - 0.5, color="white", linewidth=2.5)
    # Tier band labels at top
    edges = [0]
    for k in range(1, len(sorted_tiers)):
        if sorted_tiers[k] != sorted_tiers[k - 1]:
            edges.append(k)
    edges.append(len(sorted_tiers))
    band_names = ["Trivial", "Needs reasoning", "Needs search"]
    for i, name in enumerate(band_names):
        if i + 1 < len(edges):
            mid = (edges[i] + edges[i + 1] - 1) / 2
            ax.text(mid, -0.85, name, ha="center", fontsize=8.5,
                    fontweight="bold", color="#444")





def make_plot():
    tiers = stratify(GAME24, len(GAME24_PUZZLES))
    accs, counts = accuracy_by_tier(GAME24, tiers)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1], hspace=0.55)

    ax_overall = fig.add_subplot(gs[0, 0])
    ax_tiered  = fig.add_subplot(gs[1, 0])
    ax_heat    = fig.add_subplot(gs[2, 0])

    overall_bar(ax_overall, GAME24, len(GAME24_PUZZLES))
    tiered_bar(ax_tiered, accs, counts)
    heatmap(ax_heat, GAME24, tiers, GAME24_PUZZLES)

    fig.suptitle("Game of 24: IO vs CoT vs ToT (Gemini 2.5 Flash, n=15)",
                 fontsize=15, fontweight="bold", y=0.995)

    out = os.path.join(os.path.dirname(__file__), "comparison.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")

    # Print the tier breakdown to console
    print("\nTier breakdown:")
    for t, label in enumerate(["Trivial", "Needs reasoning", "Needs search"]):
        n = sum(1 for x in tiers if x == t)
        print(f"  Tier {t+1} ({label:18}): {n} puzzles")
        for m in GAME24:
            n_ok, n_tot = counts[t][m]
            pct = 100 * n_ok / n_tot if n_tot else 0
            print(f"    {m:8} {n_ok}/{n_tot}  ({pct:.0f}%)")

    return out


if __name__ == "__main__":
    make_plot()
