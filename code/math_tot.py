#!/usr/bin/env python3
"""
Tree of Thoughts – Competition Math (MATH Dataset)
Compares Standard Prompting, Chain-of-Thought, and Tree of Thoughts (BFS)
on the qwedsacf/competition_math HuggingFace dataset.

Implements the ToT framework from:
  "Tree of Thoughts: Deliberate Problem Solving with LLMs" (NeurIPS 2023)
  https://arxiv.org/abs/2305.10601

Usage:
  export GEMINI_API_KEY="your-key"
  python math_tot.py                              # default: 15 problems, all methods
  python math_tot.py --n_problems 10              # fewer problems
  python math_tot.py --methods io cot             # skip ToT
  python math_tot.py --b 3 5                      # test branching factor 3 and 5
  python math_tot.py --level "Level 3"            # only Level 3 problems
  python math_tot.py --verbose                    # print ToT reasoning traces
  python math_tot.py --output results.json        # save full results
"""

import re
import os
import sys
import time
import json
import random
import argparse
from typing import Optional

# ── Gemini client ──────────────────────────────────────────────────────────────

_client = None
_MODEL = "gemini-2.5-flash-lite"
_TEMPERATURE = 0.7
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


def llm(prompt: str, n: int = 1, max_tokens: int = 1024) -> list[str]:
    """Call the Gemini model n times, return list of text responses."""
    global _LLM_CALL_COUNT
    from google.genai import types as genai_types

    client = _get_client()
    config = genai_types.GenerateContentConfig(
        temperature=_TEMPERATURE,
        max_output_tokens=max_tokens,
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
        if n > 1:
            time.sleep(0.3)
    return results


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_math_problems(
    n: int = 15,
    seed: int = 42,
    level_filter: Optional[str] = None,
    exclude_problems: Optional[set] = None,
) -> list[dict]:
    """Load n problems from the MATH dataset via HuggingFace datasets.

    exclude_problems: set of problem text prefixes (first 80 chars) to skip.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: run  pip install datasets  first.")

    print("Loading MATH dataset from HuggingFace (streaming)...")
    ds = load_dataset("qwedsacf/competition_math", split="train", streaming=True)

    pool = []
    for item in ds:
        if level_filter and item.get("level") != level_filter:
            continue
        # Skip problems with Asymptote diagrams — they get truncated mid-stream
        # and are also not parseable as plain text by the LLM
        if "[asy]" in item["problem"] or "\\begin{asy}" in item["problem"]:
            continue
        ans = _extract_boxed(item["solution"])
        if ans is None:
            continue
        # Skip already-seen problems
        if exclude_problems and item["problem"][:80] in exclude_problems:
            continue
        pool.append({
            "problem": item["problem"],
            "solution": item["solution"],
            "level": item["level"],
            "type": item["type"],
            "answer": ans,
        })
        # Collect a large pool so we have plenty after exclusions
        if len(pool) >= max(n * 15, 300):
            break

    random.seed(seed)
    random.shuffle(pool)
    return pool[:n]


def _extract_boxed(text: str) -> Optional[str]:
    """Extract the innermost/last \\boxed{...} from a string (handles nesting)."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        # Also try \boxed without raw string issues
        idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def _normalize(ans: str) -> str:
    """
    Normalize an answer string for loose comparison.
    Handles common LaTeX variants so that e.g.
      -\\frac12  ==  -\\frac{1}{2}  ==  -1/2
      \\sqrt2    ==  \\sqrt{2}
      29x^5(2-7x^6)  ==  -29x^5(7x^6-2)   [not handled — needs CAS]
    """
    if not ans:
        return ""
    s = ans.strip()
    # Collapse all whitespace
    s = re.sub(r"\s+", "", s)
    # LaTeX command aliases
    for old, new in [
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("\\left",  ""),
        ("\\right", ""),
        ("\\,",     ""),
        ("\\!",     ""),
        ("\\cdot",  "*"),
        ("\\times", "*"),
        ("{,}",     ","),
        ("\\displaystyle", ""),
    ]:
        s = s.replace(old, new)
    # Expand shorthand fractions: \frac12 → \frac{1}{2}, \frac{1}2 → \frac{1}{2}
    # Pattern: \frac followed by two single-char or single-brace-group arguments
    def _expand_frac(m):
        num = m.group(1).strip("{}")
        den = m.group(2).strip("{}")
        return f"\\frac{{{num}}}{{{den}}}"
    s = re.sub(r"\\frac(\{[^{}]+\}|[0-9a-zA-Z])(\{[^{}]+\}|[0-9a-zA-Z])", _expand_frac, s)
    # Expand shorthand square roots: \sqrt2 → \sqrt{2}
    s = re.sub(r"\\sqrt([0-9a-zA-Z])(?!\{)", lambda m: f"\\sqrt{{{m.group(1)}}}", s)
    # Plain fractions: convert  a/b  to  \frac{a}{b}  when both sides are integers
    s = re.sub(r"(?<![\\a-zA-Z])(-?\d+)/(\d+)", r"\\frac{\1}{\2}", s)
    # Canonicalise sign position in fracs: \frac{-a}{b} → -\frac{a}{b}
    s = re.sub(r"\\frac\{-([^{}]+)\}", r"-\\frac{\1}", s)
    return s.lower()


def answers_match(pred: Optional[str], gt: str) -> bool:
    """Return True if predicted answer matches ground truth after normalisation."""
    if pred is None:
        return False
    return _normalize(pred) == _normalize(gt)


# ── Prompts ────────────────────────────────────────────────────────────────────

IO_PROMPT = """\
Solve the following competition math problem.
Provide ONLY your final answer enclosed in \\boxed{{}}.

Problem: {problem}

Final answer:"""

COT_PROMPT = """\
Solve the following competition math problem.
Show your reasoning step by step, then enclose your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

GENERATE_THOUGHTS_PROMPT = """\
We are solving a competition math problem by reasoning one step at a time.

Problem:
{problem}

Reasoning steps completed so far:
{steps}

Propose {k} distinct possible NEXT reasoning steps.

CRITICAL REQUIREMENTS for each step:
- The step MUST include the actual mathematical work with explicit numbers and expressions.
- Write the resulting equation or value after performing the operation, not just a description.
- BAD example: "Complete the square for the x terms."
- GOOD example: "Complete the square for x: $(x+2)^2 - 4 + y^2 - 6y = 3$, giving $(x+2)^2 + y^2 - 6y = 7$."
- BAD example: "Apply the quadratic formula."
- GOOD example: "Apply quadratic formula to $x^2 - x - 182 = 0$: $x = \\frac{{1 \\pm \\sqrt{{1+728}}}}{{2}} = \\frac{{1 \\pm 27}}{{2}}$, giving $x=14$ or $x=-13$."
If a prior step is a description without actual computation, compute it concretely.
If a prior step contains an error, propose a corrected computation.

Respond EXACTLY in this format (no extra text before or after):
Step A: <one reasoning step WITH actual computed values>
Step B: <one reasoning step WITH actual computed values>
Step C: <one reasoning step WITH actual computed values>
(and so on up to Step {k_letter})"""

EVALUATE_STATE_PROMPT = """\
We are checking whether a partial solution to a competition math problem is on the right track.

Problem:
{problem}

Partial solution:
{steps}

Classify this reasoning path as exactly one of:
  sure       – clearly correct, all steps valid, progressing well toward the answer
  likely     – plausible and probably correct, but not yet verified
  impossible – contains a definite mathematical error or logical contradiction

Respond with ONLY one word (sure / likely / impossible):"""

FINAL_ANSWER_PROMPT = """\
We have been solving a competition math problem step by step.

Problem:
{problem}

Completed reasoning:
{steps}

YOU MUST enclose your final answer in \\boxed{{}}. \
If the reasoning above is incomplete, finish it in one line, then write the boxed answer.
Do not write anything after the \\boxed{{}} line.

Final answer: \\boxed{{"""


# ── Standard (IO) Prompting ────────────────────────────────────────────────────

def solve_io(problem: str) -> Optional[str]:
    """Direct input → output prompting."""
    response = llm(IO_PROMPT.format(problem=problem))[0]
    return _extract_boxed(response)


# ── Chain-of-Thought ───────────────────────────────────────────────────────────

def solve_cot(problem: str) -> Optional[str]:
    """Chain-of-thought: step-by-step reasoning then final answer."""
    response = llm(COT_PROMPT.format(problem=problem), max_tokens=2048)[0]
    return _extract_boxed(response)


# ── Tree of Thoughts – BFS ─────────────────────────────────────────────────────

_EVAL_SCORE: dict[str, float] = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}
_K_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _generate_thoughts(problem: str, steps: list[str], k: int) -> list[str]:
    """Ask the LLM to propose k candidate next reasoning steps."""
    steps_text = (
        "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        if steps
        else "  (none – start from the beginning)"
    )
    k_letter = _K_LETTERS[k - 1] if k <= len(_K_LETTERS) else str(k)
    prompt = GENERATE_THOUGHTS_PROMPT.format(
        problem=problem,
        steps=steps_text,
        k=k,
        k_letter=k_letter,
    )
    response = llm(prompt, max_tokens=1024)[0]

    # Parse "Step A: ...", "Step B: ...", etc.
    thoughts: list[str] = []
    for line in response.splitlines():
        m = re.match(r"^\s*Step\s+[A-Za-z0-9]+\s*:\s*(.+)$", line)
        if m:
            t = m.group(1).strip()
            if t:
                thoughts.append(t)

    # Fallback: numbered list
    if not thoughts:
        for line in response.splitlines():
            m = re.match(r"^\s*[\d]+[.)]\s*(.+)$", line)
            if m:
                t = m.group(1).strip()
                if t:
                    thoughts.append(t)

    # Last-resort fallback: any non-trivial line
    if not thoughts:
        thoughts = [ln.strip() for ln in response.splitlines()
                    if ln.strip() and len(ln.strip()) > 15]

    return thoughts[:k]


def _evaluate_state(problem: str, steps: list[str]) -> float:
    """Score a partial solution as sure (1.0), likely (0.5), or impossible (0.0)."""
    if not steps:
        return 0.5
    steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
    prompt = EVALUATE_STATE_PROMPT.format(problem=problem, steps=steps_text)
    raw = llm(prompt, max_tokens=16)[0].strip().lower()
    for label in ("impossible", "sure", "likely"):
        if label in raw:
            return _EVAL_SCORE[label]
    return 0.5


def _extract_any_answer(text: str) -> Optional[str]:
    """
    Fallback extractor when the model doesn't use \\boxed{}.
    Tries common patterns like 'the answer is X', '= X', or a bare number/expression
    at the end of the response.
    """
    if not text:
        return None
    # "the answer is X" / "answer: X" / "= X" patterns (last occurrence)
    for pat in [
        r"(?:the\s+(?:final\s+)?answer\s+is|answer\s*[:=])\s*([^\n.]+)",
        r"=\s*(-?[\d\/\.\w\\{},\s\+\-\*\^()]+?)\s*(?:\.|$)",
    ]:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip().rstrip(".")
    # Last non-empty line that looks like a math expression
    for line in reversed(text.strip().splitlines()):
        line = line.strip().rstrip(".")
        if line and re.search(r"[\d\w]", line) and len(line) < 60:
            return line
    return None


def _extract_final_answer(problem: str, steps: list[str]) -> Optional[str]:
    """Extract or derive the final answer from completed reasoning steps."""
    # 1. Check if any step already contains a \\boxed{} answer
    for step in reversed(steps):
        ans = _extract_boxed(step)
        if ans is not None:
            return ans

    # 2. Ask the model for the final answer — prompt ends with \boxed{ to force format
    steps_text = (
        "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        if steps
        else "  (none)"
    )
    prompt = FINAL_ANSWER_PROMPT.format(problem=problem, steps=steps_text)
    response = llm(prompt, max_tokens=1024)[0]

    # The prompt ends with \boxed{ so prepend it for extraction
    full = "\\boxed{" + response
    ans = _extract_boxed(full)
    if ans is not None:
        return ans
    # Also try without the prepend (model may have written its own \boxed{})
    ans = _extract_boxed(response)
    if ans is not None:
        return ans

    # 3. Fallback: plain-text answer extraction (handles "The answer is 5" etc.)
    return _extract_any_answer(response)


def solve_tot_bfs(
    problem: str,
    k: int = 5,
    b: int = 3,
    max_depth: int = 4,
    verbose: bool = False,
) -> Optional[str]:
    """
    Solve using Tree of Thoughts with Breadth-First Search.

    At each depth level:
      1. For every state in the frontier, generate k candidate next thoughts.
      2. Evaluate each candidate as sure / likely / impossible.
      3. Prune impossible candidates, keep top-b by score.
    After max_depth levels, extract the final answer from the best state.
    """
    # State = list of reasoning steps taken so far
    frontier: list[list[str]] = [[]]

    for depth in range(max_depth):
        if verbose:
            print(f"    [ToT BFS] depth={depth+1}  frontier_size={len(frontier)}")

        candidates: list[tuple[float, list[str]]] = []

        for state in frontier:
            thoughts = _generate_thoughts(problem, state, k)
            if verbose:
                print(f"      → {len(thoughts)} thoughts generated")

            for thought in thoughts:
                new_state = state + [thought]
                score = _evaluate_state(problem, new_state)
                if verbose:
                    verdict = {1.0: "sure", 0.5: "likely", 0.0: "impossible"}.get(score, "?")
                    print(f"      [{verdict}] {thought[:70]}")
                if score > 0.0:  # prune impossible
                    candidates.append((score, new_state))

        if not candidates:
            if verbose:
                print("      [ToT BFS] all paths pruned, stopping early")
            break

        # Keep top-b states (ties broken by depth, i.e. longer = more specific)
        candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        frontier = [state for _, state in candidates[:b]]

        if verbose:
            print(f"      → kept {len(frontier)} states")

    if not frontier:
        return None

    # Try to extract an answer from frontier states in order of score
    for state in frontier:
        ans = _extract_final_answer(problem, state)
        if ans is not None:
            return ans
    return None


# ── Evaluation harness ─────────────────────────────────────────────────────────

def evaluate_method(
    problems: list[dict],
    method: str,
    **kwargs,
) -> dict:
    """Run a method on all problems and return aggregated results."""
    results = []

    for i, prob in enumerate(problems):
        print(f"  [{i+1:02d}/{len(problems)}] {prob['level']} {prob['type']}: "
              f"{prob['problem'][:55]}...")

        t0 = time.time()
        if method == "io":
            pred = solve_io(prob["problem"])
        elif method == "cot":
            pred = solve_cot(prob["problem"])
        elif method == "tot":
            pred = solve_tot_bfs(
                prob["problem"],
                k=kwargs.get("k", 5),
                b=kwargs.get("b", 3),
                max_depth=kwargs.get("max_depth", 4),
                verbose=kwargs.get("verbose", False),
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        elapsed = time.time() - t0

        correct = answers_match(pred, prob["answer"])
        marker = "✓" if correct else "✗"
        print(f"       {marker}  GT={prob['answer']!r}  Pred={pred!r}  ({elapsed:.1f}s)")

        results.append({
            "problem": prob["problem"][:120],
            "answer": prob["answer"],
            "predicted": pred,
            "correct": correct,
            "level": prob["level"],
            "type": prob["type"],
            "elapsed_s": round(elapsed, 2),
        })

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0.0
    return {
        "method": method,
        "accuracy": accuracy,
        "n_correct": sum(r["correct"] for r in results),
        "n_total": len(results),
        "results": results,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tree of Thoughts – Competition Math (MATH Dataset)"
    )
    parser.add_argument("--n_problems", type=int, default=15,
                        help="Number of problems to evaluate (default: 15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for problem selection")
    parser.add_argument("--methods", nargs="+", default=["io", "cot", "tot"],
                        choices=["io", "cot", "tot"],
                        help="Which methods to run")
    parser.add_argument("--k", type=int, default=5,
                        help="ToT: candidate thoughts generated per state (default: 5)")
    parser.add_argument("--b", nargs="+", type=int, default=[3, 5],
                        help="ToT: beam widths to compare (default: 3 5)")
    parser.add_argument("--max_depth", type=int, default=4,
                        help="ToT: maximum BFS depth / reasoning steps (default: 4)")
    parser.add_argument("--model", default="gemini-2.5-flash-lite",
                        help="Gemini model name")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--level", default=None,
                        help="Filter to one difficulty level, e.g. 'Level 3'")
    parser.add_argument("--verbose", action="store_true",
                        help="Print ToT reasoning traces")
    parser.add_argument("--output", default=None,
                        help="Save full results to a JSON file")
    parser.add_argument("--append_to", default=None,
                        help="Load existing results JSON, run on new problems only, "
                             "and merge everything back into this file")
    args = parser.parse_args()

    global _MODEL, _TEMPERATURE
    _MODEL = args.model
    _TEMPERATURE = args.temperature

    print(f"\n{'='*65}")
    print("Tree of Thoughts – Competition Math  (qwedsacf/competition_math)")
    print(f"Model      : {_MODEL}")
    print(f"Problems   : {args.n_problems}  (seed={args.seed})")
    print(f"Methods    : {args.methods}")
    print(f"ToT config : k={args.k}, b={args.b}, max_depth={args.max_depth}")
    if args.level:
        print(f"Level filter: {args.level}")
    print(f"{'='*65}\n")

    # ── Load existing results for append mode ─────────────────────────────────
    existing_data: Optional[dict] = None
    exclude_problems: set = set()
    if args.append_to and os.path.exists(args.append_to):
        with open(args.append_to) as f:
            existing_data = json.load(f)
        # Build exclusion set from every problem already in the file
        for p in existing_data.get("problems", []):
            exclude_problems.add(p["problem"][:80])
        print(f"  Append mode: found {len(exclude_problems)} existing problems to skip.")

    # ── Load problems ─────────────────────────────────────────────────────────
    problems = load_math_problems(
        n=args.n_problems,
        seed=args.seed,
        level_filter=args.level,
        exclude_problems=exclude_problems if exclude_problems else None,
    )
    print(f"\nSelected {len(problems)} new problems:\n")
    for i, p in enumerate(problems):
        print(f"  {i+1:2d}. [{p['level']} {p['type']:20s}] "
              f"{p['problem'][:55]}...  answer={p['answer']!r}")
    print()

    all_results: dict[str, dict] = {}

    # ── Run Standard (IO) ────────────────────────────────────────────────────
    if "io" in args.methods:
        print(f"\n{'─'*65}")
        print("Method: Standard Prompting (IO)")
        print(f"{'─'*65}")
        all_results["io"] = evaluate_method(problems, "io")

    # ── Run Chain-of-Thought ──────────────────────────────────────────────────
    if "cot" in args.methods:
        print(f"\n{'─'*65}")
        print("Method: Chain-of-Thought (CoT)")
        print(f"{'─'*65}")
        all_results["cot"] = evaluate_method(problems, "cot")

    # ── Run ToT for each beam width ───────────────────────────────────────────
    if "tot" in args.methods:
        for b_val in args.b:
            key = f"tot_b{b_val}"
            print(f"\n{'─'*65}")
            print(f"Method: Tree of Thoughts – BFS  (k={args.k}, b={b_val}, depth={args.max_depth})")
            print(f"{'─'*65}")
            all_results[key] = evaluate_method(
                problems,
                "tot",
                k=args.k,
                b=b_val,
                max_depth=args.max_depth,
                verbose=args.verbose,
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"{'Method':<22} {'Correct':>10}   {'Accuracy':>10}")
    print(f"{'─'*46}")
    for key, res in all_results.items():
        bar_len = int(res["accuracy"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {key:<20} {res['n_correct']:>3}/{res['n_total']:<3}   "
              f"{res['accuracy']*100:>6.1f}%   {bar}")

    print(f"\n  Total LLM API calls: {_LLM_CALL_COUNT}")

    # Per-problem breakdown
    print(f"\n{'─'*65}")
    print("Per-problem breakdown:")
    print(f"{'─'*65}")
    method_keys = list(all_results.keys())
    header = f"  {'#':>3}  {'Level':<8} {'Type':<18} " + "  ".join(f"{k:<10}" for k in method_keys)
    print(header)
    print(f"  {'─'*60}")
    for i, prob in enumerate(problems):
        row = f"  {i+1:>3}  {prob['level']:<8} {prob['type']:<18} "
        for key in method_keys:
            r = all_results[key]["results"][i]
            cell = f"{'✓' if r['correct'] else '✗'} {r['predicted'] or '–':>8}"
            row += f"{cell:<12}"
        print(row)

    # ── Build output data for this run ────────────────────────────────────────
    save_path = args.append_to or args.output
    if save_path:
        new_problems_meta = [
            {
                "problem": p["problem"][:300],
                "answer": p["answer"],
                "level": p["level"],
                "type": p["type"],
            }
            for p in problems
        ]
        new_summary = {
            k: {
                "accuracy": v["accuracy"],
                "n_correct": v["n_correct"],
                "n_total": v["n_total"],
            }
            for k, v in all_results.items()
        }
        new_per_problem = {k: v["results"] for k, v in all_results.items()}

        if existing_data is not None:
            # ── Merge with existing results ────────────────────────────────
            merged_problems = existing_data.get("problems", []) + new_problems_meta
            merged_per_problem: dict = {}
            all_method_keys = set(existing_data.get("per_problem", {}).keys()) | set(new_per_problem.keys())
            for key in all_method_keys:
                old_rows = existing_data.get("per_problem", {}).get(key, [])
                new_rows = new_per_problem.get(key, [])
                merged_per_problem[key] = old_rows + new_rows

            # Recompute merged summaries
            merged_summary = {}
            for key, rows in merged_per_problem.items():
                n_correct = sum(1 for r in rows if r.get("correct"))
                merged_summary[key] = {
                    "accuracy": round(n_correct / len(rows), 4) if rows else 0.0,
                    "n_correct": n_correct,
                    "n_total": len(rows),
                }

            existing_calls = existing_data.get("total_llm_calls", 0)
            output_data = {
                "config": existing_data.get("config", {}),
                "config_runs": existing_data.get("config_runs", []) + [{
                    "model": _MODEL,
                    "n_problems": args.n_problems,
                    "seed": args.seed,
                    "tot_k": args.k,
                    "tot_b": args.b,
                    "tot_max_depth": args.max_depth,
                    "level_filter": args.level,
                }],
                "problems": merged_problems,
                "summary": merged_summary,
                "per_problem": merged_per_problem,
                "total_llm_calls": existing_calls + _LLM_CALL_COUNT,
            }
            print(f"\n  Merged: {len(merged_problems)} total problems across all runs.")
        else:
            output_data = {
                "config": {
                    "model": _MODEL,
                    "n_problems": args.n_problems,
                    "seed": args.seed,
                    "tot_k": args.k,
                    "tot_b": args.b,
                    "tot_max_depth": args.max_depth,
                    "level_filter": args.level,
                },
                "problems": new_problems_meta,
                "summary": new_summary,
                "per_problem": new_per_problem,
                "total_llm_calls": _LLM_CALL_COUNT,
            }

        with open(save_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"  Results saved → {save_path}")

    print()


if __name__ == "__main__":
    main()
