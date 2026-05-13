#!/usr/bin/env python3
"""
Tree of Thoughts – 5×5 Mini Crosswords
Implements the DFS-based ToT algorithm from:
  "Tree of Thoughts: Deliberate Problem Solving with LLMs" (NeurIPS 2023)
  https://github.com/princeton-nlp/tree-of-thought-llm

Runs against crossword_data.json using the Gemini API.

Usage
-----
  export GEMINI_API_KEY="your-key-here"
  python crossword_tot.py                         # default: all puzzles, DFS+prune
  python crossword_tot.py --puzzles 0 1           # only puzzles 0 and 1
  python crossword_tot.py --show_reasoning        # print LLM reasoning during eval
  python crossword_tot.py --no_prune              # DFS without pruning
  python crossword_tot.py --n_generate 5          # 5 proposal samples per state
  python crossword_tot.py --max_per_state 3       # max branches per state
  python crossword_tot.py --time_limit 50         # max total DFS steps per puzzle
  python crossword_tot.py --model gemini-2.5-flash-lite
"""

import re
import json
import copy
import argparse
import os
import time
import sys

# ── Gemini client ─────────────────────────────────────────────────────────────

_client = None  # lazy-loaded google.genai.Client

# module-level config (filled in main)
_MODEL = "gemini-2.5-flash-lite"
_TEMPERATURE = 0.7
_SHOW_REASONING = False
_LLM_CALL_COUNT = 0


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from google import genai
    except ImportError:
        sys.exit("ERROR: run  pip install google-genai  first.")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: set the GEMINI_API_KEY environment variable.")
    _client = genai.Client(api_key=api_key)
    return _client


def llm(prompt: str, n: int = 1) -> list[str]:
    """Call the Gemini model n times, return list of text responses."""
    global _LLM_CALL_COUNT
    from google.genai import types as genai_types

    client = _get_client()
    config = genai_types.GenerateContentConfig(
        temperature=_TEMPERATURE,
        max_output_tokens=1024,
    )

    results = []
    for _ in range(n):
        _LLM_CALL_COUNT += 1
        try:
            response = client.models.generate_content(
                model=_MODEL,
                contents=prompt,
                config=config,
            )
            results.append(response.text or "")
        except Exception as e:
            print(f"  [LLM error] {e}")
            results.append("")
        # small delay to avoid rate limits on repeated calls
        if n > 1:
            time.sleep(0.4)
    return results


# ── Prompts (paper methodology – code is original) ───────────────────────────

PROPOSE_PROMPT = """\
Let's play a 5 x 5 mini crossword, where each word should have exactly 5 letters.

{input}

Given the current status, list all possible answers for unfilled or changed words, \
and your confidence levels (certain/high/medium/low), using the format \
"h1. apple (medium)". Use "certain" cautiously and only when you are 100%% sure \
this is the correct word. You can list more than one possible answer for each word.
"""

VALUE_PROMPT = """\
Evaluate if there exists a five letter word of some meaning that fit some letter \
constraints (sure/maybe/impossible).

Incorrect; to injure: w _ o _ g
The letter constraint is: 5 letters, letter 1 is w, letter 3 is o, letter 5 is g.
Some possible words that mean "Incorrect; to injure":
wrong (w r o n g): 5 letters, letter 1 is w, letter 3 is o, letter 5 is g. fit!
sure

A person with an all-consuming enthusiasm, such as for computers or anime: _ _ _ _ u
The letter constraint is: 5 letters, letter 5 is u.
Some possible words that mean "A person with an all-consuming enthusiasm, such as for computers or anime":
geek (g e e k): 4 letters, not 5
otaku (o t a k u): 5 letters, letter 5 is u
sure

Dewy; roscid: r _ _ _ l
The letter constraint is: 5 letters, letter 1 is r, letter 5 is l.
Some possible words that mean "Dewy; roscid":
moist (m o i s t): 5 letters, letter 1 is m, not r
humid (h u m i d): 5 letters, letter 1 is h, not r
I cannot think of any words now. Only 2 letters are constrained, it is still likely
maybe

A woodland: _ l _ d e
The letter constraint is: 5 letters, letter 2 is l, letter 4 is d, letter 5 is e.
Some possible words that mean "A woodland":
forest (f o r e s t): 6 letters, not 5
woods (w o o d s): 5 letters, letter 2 is o, not l
grove (g r o v e): 5 letters, letter 2 is r, not l
I cannot think of any words now. 3 letters are constrained, and _ l _ d e seems a common pattern
maybe

An inn: _ d _ w f
The letter constraint is: 5 letters, letter 2 is d, letter 4 is w, letter 5 is f.
Some possible words that mean "An inn":
hotel (h o t e l): 5 letters, letter 2 is o, not d
lodge (l o d g e): 5 letters, letter 2 is o, not d
I cannot think of any words now. 3 letters are constrained, and it is extremely unlikely to have a word with pattern _ d _ w f to mean "An inn"
impossible

Chance; a parasitic worm; a fish: w r a k _
The letter constraint is: 5 letters, letter 1 is w, letter 2 is r, letter 3 is a, letter 4 is k.
Some possible words that mean "Chance; a parasitic worm; a fish":
fluke (f l u k e): 5 letters, letter 1 is f, not w
I cannot think of any words now. 4 letters are constrained, and it is extremely unlikely to have a word with pattern w r a k _ to mean "Chance; a parasitic worm; a fish"
impossible

{input}
"""


# ── Crossword environment ─────────────────────────────────────────────────────

class MiniCrosswordsEnv:
    """
    Stateful 5×5 crossword environment.
    Clue layout: indices 0-4 → horizontal rows h1-h5
                 indices 5-9 → vertical columns v1-v5
    Board: flat list of 25 letters (row-major), '_' = empty.
    """

    def __init__(self, data: list):
        """data: list of [clues, board_gt] pairs loaded from JSON."""
        self.file = data
        self.n = len(data)
        self.cache: dict = {}           # obs → candidates_to_scores
        self.prompt_status_cache: dict = {}
        self.idx = None

    def __len__(self):
        return self.n

    def reset(self, idx: int, board=None, status=None, steps=None):
        self.idx = idx
        self.data, self.board_gt = self.file[idx]
        self.board = ["_"] * 25
        self.ans = ["_____"] * 10
        self.ans_gt = self._get_ans(self.board_gt)
        self.steps = 0
        self.status = [0] * 10  # 0: unfilled; 1: filled; 2: filled-then-changed
        if board is not None:
            self.board = board
            self.ans = self._get_ans(self.board)
        if status is not None:
            self.status = status
        if steps is not None:
            self.steps = steps
        return self.render()

    # ── board helpers ──────────────────────────────────────────────

    def _get_ans(self, board: list) -> list:
        ans = [""] * 10
        for i in range(5):
            ans[i] = "".join(board[i * 5 : (i + 1) * 5])
        for i in range(5):
            ans[i + 5] = "".join(board[i::5])
        return ans

    def render_board(self) -> str:
        s = "Current Board:\n"
        for i in range(5):
            s += "".join(self.board[i * 5 : (i + 1) * 5]) + "\n"
        return s

    def render_clues(self, status=None) -> str:
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += f"h{i+1}. {self.data[i]}\n"
        for i in range(5):
            if status is None or self.status[i + 5] == status:
                s += f"v{i+1}. {self.data[i+5]}\n"
        return s

    def render_ans(self, status=None) -> str:
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += f"h{i+1}. {self.data[i]}: {self.ans[i]}\n"
        for i in range(5):
            if status is None or self.status[i + 5] == status:
                s += f"v{i+1}. {self.data[i+5]}: {self.ans[i+5]}\n"
        return s

    def render_gt_board(self) -> str:
        s = "GT Board:\n"
        for i in range(5):
            s += " ".join(self.board_gt[i * 5 : (i + 1) * 5]) + "\n"
        return s

    def render(self, status=True) -> str:
        if status:
            return (
                self.render_board()
                + "\nUnfilled:\n"
                + self.render_ans(status=0)
                + "\nFilled:\n"
                + self.render_ans(status=1)
                + "\nChanged:\n"
                + self.render_ans(status=2)
            )
        return self.render_board() + "\n" + self.render_ans()

    # ── step ──────────────────────────────────────────────────────

    def step(self, action: str):
        """
        action format: "h1. apple"  or  "v3. lemon"
        Returns (obs, r_all, done, info)
        """
        self.steps += 1
        action = action.split("\n")[-1].split(". ")
        if len(action) != 2:
            return "Invalid! Format: h1. apple", 0, False, {}
        pos, word = action
        word = word.strip()
        if len(word) != 5:
            return "Invalid! Word must be 5 letters.", 0, False, {}

        if pos.startswith("h"):
            idx = int(pos[1:]) - 1
            self.board[idx * 5 : (idx + 1) * 5] = list(word.upper())
        elif pos.startswith("v"):
            idx = int(pos[1:]) - 1
            self.board[idx::5] = list(word.upper())
            idx += 5
        else:
            return "Invalid! Position: h1-h5 or v1-v5", 0, False, {}

        new_ans = self._get_ans(self.board)
        self.status = [
            2
            if any(
                ltr != new_ltr and ltr != "_"
                for ltr, new_ltr in zip(ans, new_ans[i])
            )
            else st
            for i, (st, ans) in enumerate(zip(self.status, self.ans))
        ]
        self.status[idx] = 1
        self.ans = new_ans

        r_all = self.board == self.board_gt
        r_letter = sum(a == b for a, b in zip(self.board, self.board_gt)) / 25
        r_word = sum(a == b for a, b in zip(self.ans, self.ans_gt)) / 10
        return (
            self.render(),
            r_all,
            r_all or self.steps >= 20,
            {"r_letter": r_letter, "r_word": r_word, "r_game": int(r_all)},
        )

    # ── value evaluation (for pruning) ────────────────────────────

    def prompt_status(self) -> dict:
        """
        For each partially-filled word evaluate sure/maybe/impossible via LLM.
        Returns count dict.
        """
        count = {"sure": 0, "maybe": 0, "impossible": 0}
        for ans, data, status in zip(self.ans, self.data, self.status):
            if ans.count("_") >= 4:
                continue
            ans_spaced = " ".join(ans.lower())
            line = f"{data}: {ans_spaced}"
            prompt = VALUE_PROMPT.format(input=line)
            if prompt in self.prompt_status_cache:
                res = self.prompt_status_cache[prompt]
            else:
                raw = llm(prompt, n=1)[0]
                if _SHOW_REASONING:
                    print(f"    [value] {line}")
                    print(f"    → {raw.strip()}")
                res = raw.split("\n")[-1].strip()
                self.prompt_status_cache[prompt] = res
            if res in count:
                count[res] += 1
        return count


# ── Proposal parsing ──────────────────────────────────────────────────────────

_PATTERN = re.compile(r"^([hv][1-5])\. ([a-zA-Z]{5}) \((certain|high|medium|low)\).*$")
_CONF_TO_VAL = {"certain": 1.0, "high": 0.5, "medium": 0.2, "low": 0.1}


def _parse_line(line: str):
    m = _PATTERN.match(line.strip())
    if m:
        return m.group(1).lower(), m.group(2).lower(), m.group(3)
    return None


def _parse_response(response: str):
    parsed = []
    for line in response.split("\n"):
        result = _parse_line(line)
        if result:
            pos, word, conf = result
            proposal = f"{pos}. {word}"
            score = _CONF_TO_VAL.get(conf, 0)
            parsed.append((proposal, score))
    return parsed if parsed else None


def get_candidates_to_scores(env: MiniCrosswordsEnv, n_generate: int) -> dict:
    obs = env.render()
    if obs in env.cache:
        if _SHOW_REASONING:
            print("  [cache hit]")
        return env.cache[obs]

    prompt = PROPOSE_PROMPT.format(input=obs)
    responses = llm(prompt, n=n_generate)
    if _SHOW_REASONING:
        for i, r in enumerate(responses):
            print(f"  [proposal {i+1}]\n{r.strip()}\n")

    candidates_to_scores: dict = {}
    for response in responses:
        parsed = _parse_response(response)
        if parsed:
            for proposal, score in parsed:
                candidates_to_scores[proposal] = (
                    candidates_to_scores.get(proposal, 0) + score
                )

    env.cache[obs] = candidates_to_scores
    return candidates_to_scores


# ── DFS search ───────────────────────────────────────────────────────────────

def dfs(
    env: MiniCrosswordsEnv,
    actions: list,
    infos: list,
    time_limit: int,
    prune: bool,
    max_per_state: int,
    n_generate: int,
):
    candidates_to_scores = get_candidates_to_scores(env, n_generate)
    if not candidates_to_scores:
        return

    if _SHOW_REASONING:
        ranked = sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"  [candidates] {ranked[:10]}")

    # save state
    board_bak = env.board.copy()
    status_bak = env.status.copy()
    steps_bak = env.steps

    cnt_per_state = 0
    for action in sorted(candidates_to_scores, key=candidates_to_scores.get, reverse=True):
        if len(infos) >= time_limit:
            break

        obs, r, done, info = env.step(action)
        r_word = info.get("r_word", 0)

        # only branch if no contradictions and still words to fill
        if (
            env.steps < 10
            and not any(s == 2 for s in env.status)
        ):
            cnt_per_state += 1
            if cnt_per_state > max_per_state:
                # restore and try next action
                env.reset(env.idx, board=board_bak.copy(), status=status_bak.copy(), steps=steps_bak)
                continue

            count = env.prompt_status()
            actions.append(action)

            info_entry = {
                "total_step": len(infos),
                "env_step": env.steps,
                "actions": actions.copy(),
                "board": env.board.copy(),
                "status": env.status.copy(),
                "info": info,
                "count": count,
            }
            infos.append(info_entry)

            if _SHOW_REASONING:
                print(f"  Step {len(infos)}: {action} | r_word={r_word:.2f} | {count}")
                print(env.render_board())

            if not prune or count["impossible"] < 1:
                dfs(env, actions, infos, time_limit, prune, max_per_state, n_generate)

            actions.pop()

        # restore state for next sibling
        env.reset(env.idx, board=board_bak.copy(), status=status_bak.copy(), steps=steps_bak)


# ── Scoring helpers ───────────────────────────────────────────────────────────

def best_info(infos: list) -> dict:
    """Return the info entry with highest r_word."""
    if not infos:
        return {"info": {"r_word": 0.0, "r_letter": 0.0, "r_game": 0}}
    return max(infos, key=lambda x: x["info"].get("r_word", 0))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tree of Thoughts – Mini Crosswords (Gemini)")
    parser.add_argument("--data", default="crossword_data.json", help="Path to crossword JSON file")
    parser.add_argument("--puzzles", nargs="*", type=int, default=None,
                        help="Puzzle indices to run (default: all)")
    parser.add_argument("--model", default="gemini-2.5-flash-lite",
                        help="Gemini model name")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n_generate", type=int, default=3,
                        help="Number of proposal LLM calls per state")
    parser.add_argument("--max_per_state", type=int, default=3,
                        help="Max branches to expand per DFS node")
    parser.add_argument("--time_limit", type=int, default=100,
                        help="Max total DFS steps per puzzle")
    parser.add_argument("--no_prune", action="store_true",
                        help="Disable impossibility pruning")
    parser.add_argument("--show_reasoning", action="store_true",
                        help="Print LLM responses and reasoning during search")
    parser.add_argument("--output", default="crossword_results.json",
                        help="Path to save detailed results JSON (default: crossword_results.json)")
    args = parser.parse_args()

    # set globals
    global _MODEL, _TEMPERATURE, _SHOW_REASONING
    _MODEL = args.model
    _TEMPERATURE = args.temperature
    _SHOW_REASONING = args.show_reasoning

    # load data
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)
    with open(data_path) as f:
        raw = json.load(f)

    env = MiniCrosswordsEnv(raw)
    puzzle_indices = args.puzzles if args.puzzles is not None else list(range(len(env)))
    prune = not args.no_prune

    print(f"\n{'='*60}")
    print(f"Tree of Thoughts – Mini Crosswords")
    print(f"Model:       {_MODEL}")
    print(f"Puzzles:     {puzzle_indices}")
    print(f"Pruning:     {prune}")
    print(f"n_generate:  {args.n_generate}  |  max_per_state: {args.max_per_state}  |  time_limit: {args.time_limit}")
    print(f"{'='*60}\n")

    all_results = []

    for puzzle_idx in puzzle_indices:
        print(f"── Puzzle {puzzle_idx} {'─'*40}")
        env.reset(puzzle_idx)
        print("Clues:")
        print(env.render_clues())

        infos: list = []
        actions: list = []

        dfs(
            env=env,
            actions=actions,
            infos=infos,
            time_limit=args.time_limit,
            prune=prune,
            max_per_state=args.max_per_state,
            n_generate=args.n_generate,
        )

        best = best_info(infos)
        r_word = best["info"].get("r_word", 0.0)
        r_letter = best["info"].get("r_letter", 0.0)
        r_game = best["info"].get("r_game", 0)

        # replay best action sequence to show the board
        env.reset(puzzle_idx)
        for act in best.get("actions", []):
            env.step(act)

        print(f"\nBest result after {len(infos)} DFS steps:")
        print(env.render_board())
        print(env.render_gt_board())
        print(f"  r_word   = {r_word:.2f}  ({int(r_word*10)}/10 words correct)")
        print(f"  r_letter = {r_letter:.2f}  ({int(r_letter*25)}/25 letters correct)")
        print(f"  r_game   = {r_game}  (solved: {'YES' if r_game else 'NO'})")
        print(f"  Best action sequence: {best.get('actions', [])}")
        print()

        clues, board_gt = env.file[puzzle_idx]
        all_results.append({
            "puzzle": puzzle_idx,
            "clues": clues,
            "board_gt": board_gt,
            "r_word": r_word,
            "r_letter": r_letter,
            "r_game": r_game,
            "dfs_steps": len(infos),
            "best_actions": best.get("actions", []),
            "reasoning_trace": infos,
        })

    # ── Save results ─────────────────────────────────────────────────────────
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(__file__), output_path)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Puzzle':<8} {'r_word':>8} {'r_letter':>10} {'Solved':>8}")
    print(f"{'-'*36}")
    for r in all_results:
        solved = "YES" if r["r_game"] else "NO"
        print(f"{r['puzzle']:<8} {r['r_word']:>8.2f} {r['r_letter']:>10.2f} {solved:>8}")
    print(f"{'-'*36}")
    avg_word   = sum(r["r_word"]   for r in all_results) / len(all_results)
    avg_letter = sum(r["r_letter"] for r in all_results) / len(all_results)
    n_solved   = sum(r["r_game"]   for r in all_results)
    print(f"{'MEAN':<8} {avg_word:>8.2f} {avg_letter:>10.2f} {n_solved}/{len(all_results):>4} solved")
    print(f"\nTotal LLM calls made: {_LLM_CALL_COUNT}")
    print()


if __name__ == "__main__":
    main()
