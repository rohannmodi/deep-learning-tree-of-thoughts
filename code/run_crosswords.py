#!/usr/bin/env python3
"""
run_crosswords.py
=================
Runs the Tree-of-Thoughts DFS on 5×5 mini crosswords, exactly as done in the
paper's notebook (scripts/crosswords/search_crosswords-dfs.ipynb), but using
Gemini instead of GPT.

Setup
-----
  export GEMINI_API_KEY="your-key"
  # optional: override model (default: gemini-2.5-flash-lite)
  export GEMINI_MODEL="gemini-2.5-flash-lite"

Usage
-----
  # from the project root  ("DL final project/")
  python run_crosswords.py                         # all puzzles, DFS + pruning
  python run_crosswords.py --puzzles 0 1 2         # specific puzzles
  python run_crosswords.py --no_prune              # DFS without pruning
  python run_crosswords.py --n_generate 5          # 5 LLM calls per state (paper used 8)
  python run_crosswords.py --max_per_state 3       # branches per DFS node (paper default)
  python run_crosswords.py --time_limit 100        # max DFS steps per puzzle (paper default)
  python run_crosswords.py --show_reasoning        # print LLM outputs as they happen

What the paper ran
------------------
  dfs(env, actions, infos, time_limit=100, prune=True, max_per_state=3)
  for every 5th puzzle (i in range(0, 100, 5)) from mini0505.json (100 puzzles).
  They used GPT-4 with n=8 proposal samples per state.

  Here we run the same algorithm on crossword_data.json (3 puzzles) with
  n_generate=3 by default to keep costs reasonable on Flash Lite.
  Increase --n_generate toward 8 for closer replication of the paper's setup.
"""

import sys
import os
import re
import json
import copy
import argparse

# ── make the cloned repo importable ──────────────────────────────────────────
REPO = os.path.join(os.path.dirname(__file__), "tree-of-thought-llm", "src")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tot.prompts.crosswords import propose_prompt, value_prompt
from tot.models import gpt
from tot.tasks.crosswords import MiniCrosswordsEnv

# ── helpers (from the paper's notebook verbatim) ─────────────────────────────

def parse_line(input_str):
    pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'
    match = re.match(pattern, input_str.strip())
    if match:
        return [match.group(1), match.group(2), match.group(3)]
    return None


confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}


def parse_response(response):
    lines = response.split('\n')
    parsed = [parse_line(l) for l in lines]
    parsed = [
        (line[0].lower() + '. ' + line[1].lower(), confidence_to_value.get(line[2], 0))
        for line in parsed if line is not None
    ]
    return parsed if len(parsed) >= 1 else None


def get_candidates_to_scores(env, n_generate, show_reasoning=False):
    obs = env.render()
    if obs in env.cache:
        if show_reasoning:
            print('  [cache hit]')
        return env.cache[obs]

    if show_reasoning:
        print('  [calling LLM for proposals]')

    responses = gpt(propose_prompt.format(input=obs), n=n_generate)

    if show_reasoning:
        for i, r in enumerate(responses):
            print(f'  [proposal {i+1}]\n{r.strip()}\n')

    candidates_to_scores = {}
    for response in responses:
        parsed = parse_response(response)
        if parsed:
            for candidate, score in parsed:
                candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score

    if show_reasoning:
        ranked = sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True)
        print(f'  [candidates] {ranked[:10]}')

    env.cache[obs] = candidates_to_scores
    return candidates_to_scores


# ── DFS (paper's notebook, lightly adapted for configurable n_generate) ──────

def dfs(env, actions, infos, time_limit, prune, max_per_state, n_generate, show_reasoning):
    candidates_to_scores = get_candidates_to_scores(env, n_generate, show_reasoning)
    if len(candidates_to_scores) == 0:
        return 0, [], []

    # back up current state
    board, status, steps = env.board.copy(), env.status.copy(), env.steps

    cnt_per_state = 0
    for action in sorted(candidates_to_scores, key=candidates_to_scores.get, reverse=True):
        obs, r, done, info = env.step(action)

        if len(infos) < time_limit and env.steps < 10 and not any(s == 2 for s in env.status):
            cnt_per_state += 1
            if cnt_per_state > max_per_state:
                env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)
                continue

            count = env.prompt_status()
            actions.append(action)

            if show_reasoning:
                print(f'  Step {len(infos)+1}: {action} | r_word={info["r_word"]:.2f} | {count}')
                print(env.render_board())

            info_entry = {
                'total_step': len(infos),
                'env_step': env.steps,
                'actions': actions.copy(),
                'info': info,
                'count': count,
            }
            infos.append(info_entry)

            if not prune or count['impossible'] < 1:
                dfs(env, actions, infos, time_limit, prune, max_per_state, n_generate, show_reasoning)

            actions.pop()

        env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)


# ── main ─────────────────────────────────────────────────────────────────────

def best_info(infos):
    if not infos:
        return {'info': {'r_word': 0.0, 'r_letter': 0.0, 'r_game': 0}}
    return max(infos, key=lambda x: x['info'].get('r_word', 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='crossword_data.json',
                        help='JSON file (relative to this script or absolute)')
    parser.add_argument('--puzzles', nargs='*', type=int, default=None,
                        help='Puzzle indices to run (default: all)')
    parser.add_argument('--n_generate', type=int, default=3,
                        help='LLM calls per state for proposals (paper used 8 with GPT-4)')
    parser.add_argument('--max_per_state', type=int, default=3,
                        help='Max DFS branches per node (paper default: 3)')
    parser.add_argument('--time_limit', type=int, default=100,
                        help='Max total DFS steps per puzzle (paper default: 100)')
    parser.add_argument('--no_prune', action='store_true',
                        help='Disable impossibility pruning')
    parser.add_argument('--show_reasoning', action='store_true',
                        help='Print LLM responses during search')
    args = parser.parse_args()

    # resolve data path
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)

    # copy data into repo's expected location if needed
    repo_data_dir = os.path.join(REPO, 'tot', 'data', 'crosswords')
    repo_data_path = os.path.join(repo_data_dir, os.path.basename(data_path))
    if not os.path.exists(repo_data_path):
        import shutil
        os.makedirs(repo_data_dir, exist_ok=True)
        shutil.copy(data_path, repo_data_path)

    env = MiniCrosswordsEnv(file=os.path.basename(data_path))
    prune = not args.no_prune
    puzzle_indices = args.puzzles if args.puzzles is not None else list(range(len(env)))

    from tot.models import GEMINI_MODEL
    print(f'\n{"="*60}')
    print(f'Tree of Thoughts – Mini Crosswords (paper DFS)')
    print(f'Model:        {GEMINI_MODEL}')
    print(f'Puzzles:      {puzzle_indices}')
    print(f'Pruning:      {prune}')
    print(f'n_generate:   {args.n_generate}  (paper used 8 with GPT-4)')
    print(f'max_per_state:{args.max_per_state}  |  time_limit: {args.time_limit}')
    print(f'{"="*60}\n')

    all_results = []

    for puzzle_idx in puzzle_indices:
        print(f'── Puzzle {puzzle_idx} {"─"*42}')
        env.reset(puzzle_idx)
        print('Clues:')
        print(env.render_clues())

        infos = []
        actions = []
        dfs(env, actions, infos,
            time_limit=args.time_limit,
            prune=prune,
            max_per_state=args.max_per_state,
            n_generate=args.n_generate,
            show_reasoning=args.show_reasoning)

        best = best_info(infos)
        r_word   = best['info'].get('r_word',   0.0)
        r_letter = best['info'].get('r_letter', 0.0)
        r_game   = best['info'].get('r_game',   0)

        # replay best action sequence to reconstruct the board
        env.reset(puzzle_idx)
        for act in best.get('actions', []):
            env.step(act)

        print(f'\nBest result after {len(infos)} DFS steps:')
        print(env.render_board())
        print(env.render_gt_board())
        print(f'  r_word   = {r_word:.2f}  ({int(r_word*10)}/10 words correct)')
        print(f'  r_letter = {r_letter:.2f}  ({int(r_letter*25)}/25 letters correct)')
        print(f'  Solved:    {"YES ✓" if r_game else "NO"}')
        print(f'  Best path: {best.get("actions", [])}')
        print()

        all_results.append({
            'puzzle': puzzle_idx,
            'r_word': r_word,
            'r_letter': r_letter,
            'r_game': int(r_game),
            'dfs_steps': len(infos),
        })

    # ── summary ──────────────────────────────────────────────────────────────
    print(f'{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    print(f'{"Puzzle":<8} {"r_word":>8} {"r_letter":>10} {"Solved":>8}')
    print(f'{"-"*38}')
    for r in all_results:
        solved = 'YES' if r['r_game'] else 'NO'
        print(f'{r["puzzle"]:<8} {r["r_word"]:>8.2f} {r["r_letter"]:>10.2f} {solved:>8}')
    print(f'{"-"*38}')
    avg_word   = sum(r['r_word']   for r in all_results) / len(all_results)
    avg_letter = sum(r['r_letter'] for r in all_results) / len(all_results)
    n_solved   = sum(r['r_game']   for r in all_results)
    print(f'{"MEAN":<8} {avg_word:>8.2f} {avg_letter:>10.2f} {n_solved}/{len(all_results)} solved')
    print()


if __name__ == '__main__':
    main()
