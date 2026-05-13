"""
Game of 24 Benchmark: CoT Baseline
--------------------------------------
Paper-faithful adaptation of Yao et al. 2023 "Tree of Thoughts".

This script runs the Chain-of-Thought (CoT) baseline for Game of 24.

Usage:
    python3 game24_eval.py --model gpt-4o --n 10
"""
import argparse
import datetime
import re
import sys
import time
from collections import Counter
from textwrap import dedent


from math_eval_3way import llm_call, C, hdr, section, Tee


PUZZLES = [

    ("1 2 3 4",     "1*2*3*4"),
    ("4 5 6 7",     "(5+7)*(6-4)"),
    ("5 6 7 8",     "(8-6)*(5+7)"),
    ("2 2 4 6",     "(2+4)*(6-2)"),
    ("4 9 10 13",   "(13-9)*(10-4)"),

    ("1 1 8 8",     "(1+1)*8+8"),
    ("3 3 7 7",     "(3+3/7)*7"),
    ("4 4 10 10",   "(10*10-4)/4"),
    ("3 5 6 8",     "6*8/(5-3)"),
    ("2 8 8 12",    "(12-8)*(8-2)"),
    ("2 3 7 11",    "11*3-7-2"),
    ("5 5 5 5",     "5*5-5/5"),

    ("1 3 4 6",     "6/(1-3/4)"),
    ("3 3 8 8",     "8/(3-8/3)"),
    ("1 5 5 5",     "(5-1/5)*5"),
]

def parse_numbers(s: str) -> list[int]:
    return [int(n) for n in s.split()]

def is_valid_solution(expr: str, original: list[int]) -> bool:
    """Check that expr evaluates to 24 and uses each original number exactly once."""
    if not expr:
        return False
    expr = expr.strip()

    m = re.search(r"[\d\(\)\+\-\*\/\.\s]+", expr)
    if not m:
        return False
    candidate = m.group(0).strip()

    if "=" in candidate:
        candidate = candidate.split("=")[0].strip()

    if not re.fullmatch(r"[\d\(\)\+\-\*\/\.\s]+", candidate):
        return False

    nums_used = [int(n) for n in re.findall(r"\d+", candidate)]
    if Counter(nums_used) != Counter(original):
        return False
    try:
        result = eval(candidate, {"__builtins__": {}}, {})
        return abs(result - 24) < 1e-6
    except Exception:
        return False

def extract_expression(response: str) -> str:
    """Pull the cleanest math expression out of an LLM response."""
    response = response.strip()
    for tag in ("Answer:", "ANSWER:", "Final answer:", "Expression:"):
        if tag in response:
            tail = response.split(tag, 1)[1]

            for line in tail.splitlines():
                if line.strip():
                    return line.strip()
            return tail.strip()

    for line in reversed(response.splitlines()):
        if re.search(r"\d", line) and re.search(r"[\+\-\*\/]", line):
            return line.strip()
    return response

COT_PROMPT = dedent("""\
    Use the given numbers (each EXACTLY once) with +, -, *, /, and parentheses to make 24.
    Show your steps, then write "Answer:" followed by the final expression.

    Example 1:
    Numbers: 4 9 10 13
    Step 1: 13 - 9 = 4 (left: 4 4 10)
    Step 2: 10 - 4 = 6 (left: 4 6)
    Step 3: 4 * 6 = 24
    Answer: (13-9)*(10-4)

    Example 2:
    Numbers: 1 4 8 8
    Step 1: 8 / 4 = 2 (left: 1 2 8)
    Step 2: 1 + 2 = 3 (left: 3 8)
    Step 3: 3 * 8 = 24
    Answer: (1+8/4)*8

    Now solve:
    Numbers: {numbers}""")

def run_cot(numbers: str, model: str) -> tuple[str, int, float]:
    prompt = COT_PROMPT.format(numbers=numbers)
    t0 = time.perf_counter()
    resp = llm_call(prompt, model, temperature=0.0, max_tokens=2048, thinking_budget=0)
    return resp, 1, time.perf_counter() - t0

def run_eval(model: str, n: int):
    print(hdr("Game of 24  ·  CoT Baseline"))
    print(f"  {C.BOLD}Model:{C.RESET} {C.YELLOW}{model}{C.RESET}   "
          f"{C.BOLD}N:{C.RESET} {n}\n")

    puzzles = PUZZLES[:n]
    records = []

    for idx, (numbers_str, _hint) in enumerate(puzzles, 1):
        original = parse_numbers(numbers_str)
        print(hdr(f"Puzzle {idx}/{n}  —  {numbers_str}"))

        print(section("Chain-of-Thought", C.CYAN))
        cot_resp, cot_calls, cot_t = run_cot(numbers_str, model)
        cot_expr = extract_expression(cot_resp)
        cot_ok = is_valid_solution(cot_expr, original)
        tag = f"{C.GREEN}✓{C.RESET}" if cot_ok else f"{C.RED}✗{C.RESET}"
        print(f"  {cot_expr}  →  {tag}  ({cot_t:.1f}s)")

        records.append({
            "puzzle": numbers_str,
            "cot_ok": cot_ok, "cot_t": cot_t,
        })

    print(hdr("Summary"))
    n_ok = sum(1 for r in records if r["cot_ok"])
    avg_t = sum(r["cot_t"] for r in records) / len(records)
    bar = C.GREEN if n_ok == len(records) else (C.YELLOW if n_ok > 0 else C.RED)
    
    col_w = 14
    print(f"\n  {'Strategy':<{col_w}} {'Correct':>10} {'Accuracy':>10} {'Avg Time':>10}")
    print(f"  {'─'*50}")
    print(f"  {'CoT':<{col_w}} {bar}{n_ok}/{len(records)}{C.RESET}    "
          f"{100*n_ok/len(records):>6.0f}%   {avg_t:>8.1f}s")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game of 24: CoT Baseline")
    parser.add_argument("--model",  default="gpt-4o",
                        help="Model to use (default: gpt-4o)")
    parser.add_argument("--n",      type=int, default=10, help="Number of puzzles (default: 10)")
    args = parser.parse_args()

    log_file = f"game24_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    Tee(log_file)
    print(f"Logging to {log_file}\n")

    run_eval(args.model, args.n)
