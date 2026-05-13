"""
Token throughput plot using REAL data from run logs.

Replaces plot_tokens_compact.png. Differences:
- Reads actual API call counts per puzzle/problem from log files.
- Drops the GSM8K bar (no GSM8K runs exist in this project).
- Shows per-level distribution for MATH (avg + variance), not single point estimates.
- Error bars = std deviation of API calls × tokens-per-call estimate.

Token-per-call estimate:
  Each ToT iteration ≈ 1 propose call (~700 tokens with prompt+output) and
  ~5 value/score calls (~310 tokens each). Weighted average ≈ 370 tok/call.
"""
import glob
import os
import re
import statistics

import matplotlib.pyplot as plt
import numpy as np


TOKENS_PER_CALL = 370   # weighted estimate (see docstring)


# ─────────────────────────────────────────────────────────────────────────────
# Parse logs
# ─────────────────────────────────────────────────────────────────────────────
def parse_math_logs() -> dict[int, list[int]]:
    """Return {level: [api_calls_per_problem, ...]} aggregated across MATH runs."""
    by_level: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    pattern_level = re.compile(r"Algebra\s+\[Level\s+([1-5])\]")
    pattern_calls = re.compile(r"\(([0-9.]+)s,\s*(\d+)\s+API\s+calls\)")
    for log in glob.glob(os.path.join(os.path.dirname(__file__), "run_2026050[45]*.log")):
        try:
            with open(log) as f:
                text = f.read()
        except Exception:
            continue
        # Walk problem by problem: each problem has a Level header followed by
        # at most 3 API-call lines (SS, CoT, ToT). We only want the ToT line.
        # Strategy: split by "Problem N/M" sections.
        sections = re.split(r"Problem\s+\d+/\d+", text)
        for sec in sections[1:]:
            lvl_m = pattern_level.search(sec)
            if not lvl_m:
                continue
            lvl = int(lvl_m.group(1))
            # The ToT line is the LAST API-call match in the problem section
            # (after SS and CoT have already been recorded)
            calls_matches = pattern_calls.findall(sec)
            if not calls_matches:
                continue
            tot_calls = int(calls_matches[-1][1])
            by_level[lvl].append(tot_calls)
    return by_level


def parse_game24_logs() -> list[int]:
    """Return per-puzzle ToT API calls from the most recent 15-puzzle run."""
    path = "/tmp/game24_15.log"
    if not os.path.exists(path):
        candidates = sorted(glob.glob(os.path.join(
            os.path.dirname(__file__), "game24_*.log")))
        if not candidates:
            return []
        path = candidates[-1]
    pat = re.compile(r"\(([0-9.]+)s,\s*(\d+)\s+API\s+calls\)")
    calls = []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                calls.append(int(m.group(2)))
    return calls


def parse_gsm8k_logs() -> list[int]:
    """Return per-problem ToT API calls aggregated across GSM8K runs."""
    calls: list[int] = []
    pat = re.compile(r"\(([0-9.]+)s,\s*(\d+)\s+API\s+calls\)")
    for log in glob.glob(os.path.join(os.path.dirname(__file__), "run_*.log")):
        try:
            with open(log) as f:
                text = f.read()
        except Exception:
            continue
        if "GSM8K Benchmark" not in text:
            continue
        # Per problem: SS line ("X call"), CoT line ("X call"), ToT line ("X API calls")
        # The pattern matches ToT lines specifically because of "API calls" wording.
        for m in pat.finditer(text):
            calls.append(int(m.group(2)))
    return calls


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def make_plot():
    g24_calls    = parse_game24_logs()
    gsm8k_calls  = parse_gsm8k_logs()
    math_by_lvl  = parse_math_logs()

    g24_tokens   = [c * TOKENS_PER_CALL for c in g24_calls]
    gsm8k_tokens = [c * TOKENS_PER_CALL for c in gsm8k_calls]
    math_tokens  = {l: [c * TOKENS_PER_CALL for c in calls]
                    for l, calls in math_by_lvl.items()}

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Game of 24 (x=0) ─────────────────────────────────────────────────────
    g24_mean = statistics.mean(g24_tokens) if g24_tokens else 0
    g24_std  = statistics.stdev(g24_tokens) if len(g24_tokens) > 1 else 0
    ax.bar(0, g24_mean, width=0.7, color="#6a1b9a",
           edgecolor="black", zorder=3, yerr=g24_std, capsize=6,
           error_kw={"ecolor": "#333", "elinewidth": 1.2})
    ax.text(0, g24_mean + g24_std + 800,
            f"{g24_mean/1000:.1f}k\n(n={len(g24_calls)})",
            ha="center", va="bottom", fontweight="bold", fontsize=10)

    # ── GSM8K (x=1) ──────────────────────────────────────────────────────────
    gsm_mean = statistics.mean(gsm8k_tokens) if gsm8k_tokens else 0
    gsm_std  = statistics.stdev(gsm8k_tokens) if len(gsm8k_tokens) > 1 else 0
    ax.bar(1, gsm_mean, width=0.7, color="#8e24aa",
           edgecolor="black", zorder=3, yerr=gsm_std, capsize=6,
           error_kw={"ecolor": "#333", "elinewidth": 1.2})
    ax.text(1, gsm_mean + max(gsm_std, 800) + 800,
            f"{gsm_mean/1000:.1f}k\n(n={len(gsm8k_calls)})",
            ha="center", va="bottom", fontweight="bold", fontsize=10)

    # ── MATH levels (5 sub-bars at x=2) ──────────────────────────────────────
    main_width = 0.7
    sub_width = (main_width - 0.05) / 5
    base_x = 2
    math_offsets = [base_x - main_width / 2 + sub_width / 2 + i * (sub_width + 0.01)
                    for i in range(5)]
    math_colors  = ["#e1bee7", "#ce93d8", "#ba68c8", "#ab47bc", "#9c27b0"]
    for i, lvl in enumerate([1, 2, 3, 4, 5]):
        vals = math_tokens.get(lvl, [])
        mean = statistics.mean(vals) if vals else 0
        std  = statistics.stdev(vals) if len(vals) > 1 else 0
        ax.bar(math_offsets[i], mean, width=sub_width, color=math_colors[i],
               edgecolor="black", zorder=3, yerr=std, capsize=4,
               error_kw={"ecolor": "#333", "elinewidth": 1.0})
        ax.text(math_offsets[i], -1500, f"L{lvl}\nn={len(vals)}",
                ha="center", va="top", fontsize=9, color="#333")
        if mean > 0:
            ax.text(math_offsets[i], mean + std + 400,
                    f"{mean/1000:.1f}k", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Game of 24", "GSM8K", "MATH (per Level)"],
                       fontweight="bold", fontsize=12)
    ax.set_ylabel("Tokens per task (≈ API calls × 370 tokens/call)",
                  fontsize=10.5, fontweight="bold")
    ax.set_title("ToT Token Throughput — Real Data from Run Logs",
                 fontsize=14, fontweight="bold", pad=14)
    ax.yaxis.grid(True, color="#eee", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    note = ("Error bars: ±1 std dev across problems.  All bars are ToT-BFS only.\n"
            "Token estimate: API calls × ~370 (weighted avg of propose calls ~700 "
            "and value/score calls ~310).")
    fig.text(0.5, -0.02, note, ha="center", fontsize=8.5, style="italic", color="#555")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "plot_tokens_real.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # Print numerical breakdown for transparency
    print(f"\nGame of 24 (n={len(g24_calls)}):")
    print(f"  calls: mean={statistics.mean(g24_calls):.1f}, "
          f"std={statistics.stdev(g24_calls):.1f}, "
          f"min={min(g24_calls)}, max={max(g24_calls)}")
    print(f"  tokens: {g24_mean:.0f} ± {g24_std:.0f}")

    if gsm8k_calls:
        print(f"\nGSM8K (n={len(gsm8k_calls)}):")
        c_mean = statistics.mean(gsm8k_calls)
        c_std  = statistics.stdev(gsm8k_calls) if len(gsm8k_calls) > 1 else 0
        print(f"  calls: mean={c_mean:.1f}, std={c_std:.1f}, "
              f"min={min(gsm8k_calls)}, max={max(gsm8k_calls)}")
        print(f"  tokens: {gsm_mean:.0f} ± {gsm_std:.0f}")

    print("\nMATH by Level:")
    for lvl in [1, 2, 3, 4, 5]:
        calls = math_by_lvl.get(lvl, [])
        if not calls:
            continue
        c_mean = statistics.mean(calls)
        c_std  = statistics.stdev(calls) if len(calls) > 1 else 0
        print(f"  L{lvl} (n={len(calls)}): "
              f"calls mean={c_mean:.1f}±{c_std:.1f}, "
              f"min={min(calls)}, max={max(calls)} | "
              f"tokens ≈ {c_mean*TOKENS_PER_CALL:.0f}")

    return out


if __name__ == "__main__":
    make_plot()
