#!/usr/bin/env python3
"""
Game of 24 — Chain-of-Thought (CoT) baseline
Replicates the CoT prompt from Yao et al. 2023 "Tree of Thoughts".

Uses puzzles 901-910 from 24.csv (the hard test set used in the paper).
Runs 1 CoT sample per puzzle (set N_SAMPLES > 1 for CoT-SC).

Usage:
  export OPENAI_API_KEY="..."
  python3 cot_game24.py
"""

import os, sys, re, csv, ast
from pathlib import Path

# ── CoT prompt (verbatim from the ToT repo) ───────────────────────────────────
COT_PROMPT = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
Input: {input}
'''

# ── LLM call ──────────────────────────────────────────────────────────────────
def call_openai(prompt: str, model: str = "gpt-4o", temperature: float = 0.7) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


# ── answer validation ─────────────────────────────────────────────────────────
def extract_answer(response: str) -> str | None:
    """Return the expression after 'Answer:' on any line."""
    for line in response.splitlines():
        m = re.search(r"[Aa]nswer\s*:\s*(.+)", line)
        if m:
            expr = m.group(1).strip()
            # strip trailing '= 24' if present
            expr = re.sub(r"\s*=\s*24\s*$", "", expr).strip()
            return expr
    return None


def _collect_numbers(node) -> list[float]:
    """Recursively collect all numeric literals from an AST."""
    nums = []
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        nums.append(float(node.value))
    for child in ast.iter_child_nodes(node):
        nums.extend(_collect_numbers(child))
    return nums


def is_valid(expression: str, puzzle_numbers: list[int]) -> bool:
    """
    Returns True iff:
      1. The expression evaluates to 24 (within floating-point tolerance).
      2. The numbers used in the expression match the puzzle numbers exactly
         (each used exactly once, same multiset).
    """
    # Safety: only allow arithmetic characters
    if re.search(r"[a-zA-Z_]", expression):
        return False
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False

    # Collect numbers from AST
    used = sorted(_collect_numbers(tree.body))
    expected = sorted(float(n) for n in puzzle_numbers)
    if used != expected:
        return False

    try:
        result = eval(compile(tree, "<string>", "eval"))  # noqa: S307
        return abs(result - 24) < 1e-6
    except Exception:
        return False


# ── load puzzles ──────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "tree-of-thought-llm/src/tot/data/24/24.csv"

def load_puzzles(start: int = 901, n: int = 10) -> list[str]:
    """Load n puzzles beginning at 1-based rank `start`."""
    puzzles = []
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row["Rank"])
            if start <= rank < start + n:
                puzzles.append(row["Puzzles"].strip())
    return sorted(puzzles, key=lambda p: [int(x) for x in p.split()])


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY environment variable.")

    MODEL = "gpt-4o"
    N_SAMPLES = 1          # set to >1 for CoT-SC majority vote
    START_RANK = 901
    N_PUZZLES = 10

    puzzles = load_puzzles(START_RANK, N_PUZZLES)
    if len(puzzles) < N_PUZZLES:
        sys.exit(f"Only found {len(puzzles)} puzzles; expected {N_PUZZLES}.")

    print(f"Game of 24 — CoT Baseline")
    print(f"Model : {MODEL}  |  Samples/puzzle: {N_SAMPLES}")
    print(f"Puzzles: ranks {START_RANK}–{START_RANK + N_PUZZLES - 1} (hard test set)")
    print("=" * 70)

    correct = 0
    results = []

    for idx, puzzle in enumerate(puzzles, 1):
        nums = list(map(int, puzzle.split()))
        prompt = COT_PROMPT.format(input=puzzle)

        # Collect N_SAMPLES responses
        answers = []
        raw_responses = []
        for _ in range(N_SAMPLES):
            raw = call_openai(prompt, model=MODEL)
            raw_responses.append(raw)
            expr = extract_answer(raw)
            if expr:
                answers.append(expr)

        # Pick answer: majority vote if N_SAMPLES > 1, else first answer
        if N_SAMPLES == 1:
            final_expr = answers[0] if answers else None
        else:
            from collections import Counter
            valid_answers = [a for a in answers if is_valid(a, nums)]
            if valid_answers:
                final_expr = Counter(valid_answers).most_common(1)[0][0]
            elif answers:
                final_expr = Counter(answers).most_common(1)[0][0]
            else:
                final_expr = None

        valid = is_valid(final_expr, nums) if final_expr else False
        if valid:
            correct += 1

        status = "✓" if valid else "✗"
        print(f"\n[{idx:02d}] Input: {puzzle}")
        print(f"     CoT output: {raw_responses[0].strip()[:300]}")
        print(f"     Parsed answer: {final_expr}")
        print(f"     Result: {status}")

        results.append({
            "rank": START_RANK + idx - 1,
            "puzzle": puzzle,
            "answer": final_expr,
            "correct": valid,
        })

    accuracy = correct / N_PUZZLES * 100
    print("\n" + "=" * 70)
    print(f"ACCURACY: {correct}/{N_PUZZLES} = {accuracy:.1f}%")
    print(f"(Paper reports CoT accuracy ~4% on 100 hard puzzles with GPT-4)")
    print("=" * 70)

    # Save results
    import json
    out = Path(__file__).parent / "cot_game24_results.json"
    with open(out, "w") as f:
        json.dump({"model": MODEL, "n_samples": N_SAMPLES,
                   "accuracy": accuracy, "results": results}, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
