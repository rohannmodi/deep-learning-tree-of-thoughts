#!/usr/bin/env python3
"""
visualize_crossword.py
======================
Render the Tree-of-Thoughts DFS reasoning trace for a single crossword puzzle
as a sequence of crossword-grid images.

Usage
-----
  python visualize_crossword.py                          # list available puzzles
  python visualize_crossword.py --puzzle 0               # render ALL steps
  python visualize_crossword.py --puzzle 0 --max 20      # cap at 20 images
  python visualize_crossword.py --puzzle 0 --outdir frames/
  python visualize_crossword.py --results other_file.json --puzzle 1

Requirements
------------
  pip install matplotlib
"""

import argparse
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import to_rgba
except ImportError:
    sys.exit("ERROR: run  pip install matplotlib  first.")


# ── Palette ───────────────────────────────────────────────────────────────────
BG             = "#F0F2F5"     # figure background
GRID_BG        = "#FFFFFF"     # cell default (empty)
CELL_EMPTY     = "#FFFFFF"
CELL_ACTIVE    = "#1A237E"     # deep indigo  – word placed this step
CELL_FILLED    = "#E3F2FD"     # ice blue     – previously placed word
CELL_CONFLICT  = "#B71C1C"     # deep red     – conflict cell

LETTER_ACTIVE    = "#FFFFFF"
LETTER_FILLED    = "#0D47A1"
LETTER_CONFLICT  = "#FFFFFF"
LETTER_EMPTY     = "#CFD8DC"   # placeholder dash

BORDER_NORMAL  = "#B0BEC5"
BORDER_ACTIVE  = "#1A237E"
BORDER_CONFLICT= "#B71C1C"

LABEL_COLOR    = "#546E7A"
HEADER_COLOR   = "#1A237E"
SUBHEADER_COLOR= "#455A64"
META_COLOR     = "#607D8B"

# status bar chip colours
CHIP_SURE      = "#2E7D32"
CHIP_MAYBE     = "#F57F17"
CHIP_IMPOSSIBLE= "#C62828"
CHIP_WORD_OK   = "#1B5E20"
CHIP_WORD_WRONG= "#B71C1C"


# ── Board helpers (no env import needed) ─────────────────────────────────────

def _get_ans(board):
    ans = [""] * 10
    for i in range(5):
        ans[i] = "".join(board[i*5:(i+1)*5])
    for i in range(5):
        ans[i+5] = "".join(board[i::5])
    return ans


def _board_from_actions(actions):
    board = ["_"] * 25
    status = [0] * 10
    ans = _get_ans(board)

    for raw in actions:
        parts = raw.split(". ", 1)
        if len(parts) != 2:
            continue
        pos, word = parts[0], parts[1].strip().upper()
        if len(word) != 5:
            continue

        old_ans = ans[:]
        if pos.startswith("h"):
            idx = int(pos[1:]) - 1
            board[idx*5:(idx+1)*5] = list(word)
        elif pos.startswith("v"):
            idx = int(pos[1:]) - 1
            board[idx::5] = list(word)
            idx += 5
        else:
            continue

        ans = _get_ans(board)
        status = [
            2 if any(a != b and a != "_" for a, b in zip(old_ans[i], ans[i])) else s
            for i, s in enumerate(status)
        ]
        status[idx] = 1

    return board, status


def _active_cells(action: str) -> set:
    """Return set of board-cell indices that the last action touches."""
    if not action:
        return set()
    parts = action.split(". ", 1)
    if len(parts) != 2:
        return set()
    pos = parts[0].strip().lower()
    if pos.startswith("h"):
        r = int(pos[1:]) - 1
        return set(range(r*5, r*5+5))
    elif pos.startswith("v"):
        c = int(pos[1:]) - 1
        return set(range(c, 25, 5))
    return set()


# ── Drawing ───────────────────────────────────────────────────────────────────

def _rounded_rect(ax, x, y, w, h, color, edge_color, lw=1.0, radius=0.03,
                  transform=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge_color,
        facecolor=color,
        transform=transform or ax.transAxes,
        clip_on=False,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def _chip(ax, x, y, label, bg, fg="#FFFFFF", fs=6.5, transform=None):
    """Draw a small pill-shaped label."""
    t = ax.transAxes if transform is None else transform
    txt = ax.text(x, y, label, ha="center", va="center",
                  fontsize=fs, fontweight="bold", color=fg,
                  transform=t, zorder=4,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor=bg,
                            edgecolor="none", alpha=0.92))
    return txt


def draw_step(
    fig, ax,
    board, status, board_gt,
    step_num, total_steps,
    action, info, count,
    actions_so_far,
    puzzle_idx,
):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    active = _active_cells(action)

    # ── layout constants ──────────────────────────────────────────────────────
    # Grid sits in the centre of the axes square
    LABEL_MARGIN = 0.07   # space for h-/v- labels
    PAD_TOP      = 0.10   # space for header above grid
    PAD_BOT      = 0.13   # space for footer below grid

    # Grid occupies the middle strip
    grid_x0 = LABEL_MARGIN
    grid_y0 = PAD_BOT
    grid_x1 = 1.0 - 0.02
    grid_y1 = 1.0 - PAD_TOP
    grid_w  = grid_x1 - grid_x0
    grid_h  = grid_y1 - grid_y0
    cell_w  = grid_w / 5
    cell_h  = grid_h / 5

    # ── header ────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.975, f"Puzzle {puzzle_idx}  ·  Step {step_num} / {total_steps}",
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            color=HEADER_COLOR, transform=ax.transAxes)

    action_str = action if action else "—"
    ax.text(0.5, 0.955, f"action: {action_str}",
            ha="center", va="top", fontsize=7, color=SUBHEADER_COLOR,
            transform=ax.transAxes)

    # thin separator line under header
    ax.plot([0.02, 0.98], [1 - PAD_TOP + 0.005, 1 - PAD_TOP + 0.005],
            color=BORDER_NORMAL, linewidth=0.6, transform=ax.transAxes, clip_on=False)

    # ── column labels (v1-v5) – above grid ────────────────────────────────────
    for c in range(5):
        cx = grid_x0 + c * cell_w + cell_w / 2
        cy = grid_y1 + 0.015
        v_idx = c + 5
        st = status[v_idx]
        color = (CHIP_SURE if st == 1 else
                 (CHIP_IMPOSSIBLE if st == 2 else LABEL_COLOR))
        ax.text(cx, cy, f"v{c+1}", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color=color,
                transform=ax.transAxes)

    # ── row labels (h1-h5) – left of grid ────────────────────────────────────
    for r in range(5):
        cy = grid_y1 - r * cell_h - cell_h / 2
        h_idx = r
        st = status[h_idx]
        color = (CHIP_SURE if st == 1 else
                 (CHIP_IMPOSSIBLE if st == 2 else LABEL_COLOR))
        ax.text(grid_x0 - 0.015, cy, f"h{r+1}", ha="right", va="center",
                fontsize=6.5, fontweight="bold", color=color,
                transform=ax.transAxes)

    # ── 5×5 cells ─────────────────────────────────────────────────────────────
    for r in range(5):
        for c in range(5):
            cell_idx = r * 5 + c
            letter   = board[cell_idx]
            gt_ltr   = board_gt[cell_idx]
            h_st     = status[r]
            v_st     = status[c + 5]

            x0 = grid_x0 + c * cell_w
            y0 = grid_y1 - (r + 1) * cell_h

            # pick colours
            is_conflict = h_st == 2 or v_st == 2
            is_active   = cell_idx in active and letter != "_"
            is_filled   = letter != "_"

            if is_conflict:
                fill   = CELL_CONFLICT
                edge   = BORDER_CONFLICT
                lw     = 1.5
                txt_c  = LETTER_CONFLICT
            elif is_active:
                fill   = CELL_ACTIVE
                edge   = BORDER_ACTIVE
                lw     = 1.5
                txt_c  = LETTER_ACTIVE
            elif is_filled:
                fill   = CELL_FILLED
                edge   = BORDER_NORMAL
                lw     = 0.8
                txt_c  = LETTER_FILLED
            else:
                fill   = CELL_EMPTY
                edge   = BORDER_NORMAL
                lw     = 0.8
                txt_c  = LETTER_EMPTY

            INSET = 0.003
            _rounded_rect(ax,
                          x0 + INSET, y0 + INSET,
                          cell_w - 2*INSET, cell_h - 2*INSET,
                          fill, edge, lw=lw, radius=0.015)

            # letter
            display = letter if letter != "_" else "·"
            ax.text(x0 + cell_w/2, y0 + cell_h/2, display,
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color=txt_c, transform=ax.transAxes, zorder=3)

            # ground-truth hint (tiny, bottom-right corner, only if wrong)
            if letter not in ("_", gt_ltr):
                ax.text(x0 + cell_w*0.88, y0 + cell_h*0.12,
                        gt_ltr.lower(),
                        ha="center", va="center",
                        fontsize=4.5, color="#EF5350",
                        transform=ax.transAxes, zorder=3)

    # ── thin separator line above footer ─────────────────────────────────────
    ax.plot([0.02, 0.98], [PAD_BOT - 0.01, PAD_BOT - 0.01],
            color=BORDER_NORMAL, linewidth=0.6, transform=ax.transAxes, clip_on=False)

    # ── footer: score chips ───────────────────────────────────────────────────
    r_word   = info.get("r_word",   0.0)
    r_letter = info.get("r_letter", 0.0)
    r_game   = info.get("r_game",   0)
    sure       = count.get("sure",       0)
    maybe      = count.get("maybe",      0)
    impossible = count.get("impossible", 0)

    # word/letter score bar
    score_y  = PAD_BOT - 0.045
    chips = [
        (f"words {int(r_word*10)}/10", CHIP_WORD_OK if r_word >= 0.5 else META_COLOR),
        (f"letters {int(r_letter*25)}/25", CHIP_WORD_OK if r_letter >= 0.5 else META_COLOR),
        ("SOLVED ✓" if r_game else "unsolved", CHIP_WORD_OK if r_game else CHIP_WORD_WRONG),
    ]
    chip_xs = [0.18, 0.45, 0.72]
    for cx, (lbl, clr) in zip(chip_xs, chips):
        _chip(ax, cx, score_y, lbl, clr, fs=6.2)

    # value chips (sure / maybe / impossible)
    val_y = PAD_BOT - 0.092
    _chip(ax, 0.18, val_y, f"sure {sure}",       CHIP_SURE,       fs=5.8)
    _chip(ax, 0.45, val_y, f"maybe {maybe}",      CHIP_MAYBE, fg="#212121", fs=5.8)
    _chip(ax, 0.72, val_y, f"impossible {impossible}", CHIP_IMPOSSIBLE, fs=5.8)

    # path breadcrumb
    path_y = PAD_BOT - 0.135
    path_str = " → ".join(actions_so_far) if actions_so_far else "—"
    if len(path_str) > 72:
        path_str = "…" + path_str[-(70):]
    ax.text(0.5, path_y, path_str,
            ha="center", va="top", fontsize=5.5, color=META_COLOR,
            transform=ax.transAxes, style="italic")


# ── Main render function ──────────────────────────────────────────────────────

def render_puzzle(puzzle_data: dict, outdir: str, max_steps=None):
    puzzle_idx = puzzle_data["puzzle"]
    board_gt   = puzzle_data["board_gt"]
    trace      = puzzle_data.get("reasoning_trace", [])

    os.makedirs(outdir, exist_ok=True)

    if not trace:
        print(f"  Puzzle {puzzle_idx}: no reasoning trace recorded.")
        return

    total = len(trace)
    print(f"  Puzzle {puzzle_idx}: {total} DFS steps recorded.")

    if max_steps is None or max_steps <= 0 or max_steps >= total:
        indices = list(range(total))
    else:
        # evenly sampled, always include first and last
        step = (total - 1) / (max_steps - 1)
        indices = sorted(set(round(i * step) for i in range(max_steps)))

    print(f"  Rendering {len(indices)} frames …")

    for frame_num, step_idx in enumerate(indices):
        entry = trace[step_idx]
        actions_so_far = entry.get("actions", [])

        if "board" in entry and "status" in entry:
            board  = entry["board"]
            status = entry["status"]
        else:
            board, status = _board_from_actions(actions_so_far)

        action = actions_so_far[-1] if actions_so_far else ""
        info   = entry.get("info",  {})
        count  = entry.get("count", {})

        fig, ax = plt.subplots(figsize=(4.8, 5.2))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        draw_step(
            fig=fig, ax=ax,
            board=board, status=status, board_gt=board_gt,
            step_num=step_idx + 1, total_steps=total,
            action=action, info=info, count=count,
            actions_so_far=actions_so_far,
            puzzle_idx=puzzle_idx,
        )

        fname = os.path.join(outdir, f"step_{frame_num+1:04d}_dfs{step_idx+1:04d}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"    {fname}")

    print(f"\n  Done — {len(indices)} images saved to '{outdir}'.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize ToT crossword DFS reasoning steps as images."
    )
    parser.add_argument("--results", default="crossword_results.json",
                        help="Crossword results JSON (default: crossword_results.json)")
    parser.add_argument("--puzzle", type=int, default=None,
                        help="Puzzle index to render")
    parser.add_argument("--list", action="store_true",
                        help="List available puzzles and exit")
    parser.add_argument("--max", type=int, default=None, dest="max_steps",
                        help="Cap number of frames (default: render all steps)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: crossword_frames/puzzle_N/)")
    args = parser.parse_args()

    results_path = args.results
    if not os.path.isabs(results_path):
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_path)

    if not os.path.exists(results_path):
        sys.exit(
            f"ERROR: results file not found: {results_path}\n"
            "Run crossword_tot.py first to generate it."
        )

    with open(results_path) as f:
        all_results = json.load(f)

    if args.list or args.puzzle is None:
        print(f"\nPuzzles in {os.path.basename(results_path)}:\n")
        for r in all_results:
            n = len(r.get("reasoning_trace", []))
            solved = "SOLVED" if r.get("r_game") else "unsolved"
            print(f"  puzzle {r['puzzle']:2d}  |  {n:4d} steps  |  "
                  f"r_word={r['r_word']:.2f}  |  {solved}")
        if args.puzzle is None:
            print("\nUse --puzzle N to visualize a puzzle.")
        return

    puzzle_data = next((r for r in all_results if r["puzzle"] == args.puzzle), None)
    if puzzle_data is None:
        sys.exit(
            f"ERROR: puzzle {args.puzzle} not found.\n"
            f"Available: {[r['puzzle'] for r in all_results]}"
        )

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "crossword_frames",
        f"puzzle_{args.puzzle}",
    )

    cap = f"(capped at {args.max_steps})" if args.max_steps else "(all steps)"
    print(f"\nVisualizing puzzle {args.puzzle}  {cap}")
    print(f"Output: {outdir}\n")

    render_puzzle(puzzle_data, outdir=outdir, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
