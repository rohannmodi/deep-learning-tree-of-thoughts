#!/usr/bin/env python3
"""
Debug script: run a single problem with full ToT tracing to diagnose failures.

Usage:
  export GEMINI_API_KEY="..."
  python3 debug_tot.py                  # runs P8 (circle) by default
  python3 debug_tot.py --problem p10    # runs P10 (integer square)
  python3 debug_tot.py --problem both   # runs both P8 and P10
"""

import os, sys, json, time, re, argparse
from typing import Optional

# ── copy the minimal LLM client from math_tot.py ─────────────────────────────
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
        sys.exit("pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY")
    _client = genai.Client(api_key=api_key)
    return _client

def llm(prompt: str, max_tokens: int = 1024) -> str:
    global _LLM_CALL_COUNT
    from google.genai import types as genai_types
    client = _get_client()
    config = genai_types.GenerateContentConfig(
        temperature=_TEMPERATURE, max_output_tokens=max_tokens)
    _LLM_CALL_COUNT += 1
    try:
        r = client.models.generate_content(model=_MODEL, contents=prompt, config=config)
        return r.text or ""
    except Exception as e:
        return f"[ERROR: {e}]"

# ── answer helpers ────────────────────────────────────────────────────────────
def _extract_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":   depth += 1
        elif text[i] == "}":
            if depth == 0:   return text[start:i].strip()
            depth -= 1
    return None

# ── problems to debug ─────────────────────────────────────────────────────────
PROBLEMS = {
    "p8": {
        "problem": "Let $C$ be the circle with equation $x^2-6y-3=-y^2-4x$. "
                   "If $(a,b)$ is the center of $C$ and $r$ is its radius, what is the value of $a+b+r$?",
        "answer": "5",
        "label": "P8 – Circle equation (Level 4 Algebra)",
    },
    "p10": {
        "problem": "The square of an integer is 182 greater than the integer itself. "
                   "What is the sum of all integers for which this is true?",
        "answer": "1",
        "label": "P10 – Integer square sum (Level 3 Algebra)",
    },
}

# ── prompts (same as math_tot.py) ─────────────────────────────────────────────
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

_K_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_EVAL_SCORE = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}

# ── instrumented ToT BFS ──────────────────────────────────────────────────────
def run_tot_debug(problem: str, k: int = 5, b: int = 3, max_depth: int = 4) -> dict:
    """
    Runs ToT-BFS with full tracing.
    Returns a log dict with every prompt, response, and decision made.
    """
    log = {
        "problem": problem,
        "config": {"k": k, "b": b, "max_depth": max_depth},
        "depths": [],
        "final_answer_attempts": [],
        "predicted": None,
    }

    frontier: list[list[str]] = [[]]

    for depth in range(max_depth):
        depth_log = {
            "depth": depth + 1,
            "frontier_size": len(frontier),
            "states": [],
            "kept_states": [],
        }

        candidates: list[tuple[float, list[str]]] = []

        for state_idx, state in enumerate(frontier):
            steps_text = (
                "\n".join(f"  {i+1}. {s}" for i, s in enumerate(state))
                if state else "  (none – start from the beginning)"
            )
            k_letter = _K_LETTERS[k - 1]
            gen_prompt = GENERATE_THOUGHTS_PROMPT.format(
                problem=problem, steps=steps_text, k=k, k_letter=k_letter)

            gen_response = llm(gen_prompt)
            time.sleep(0.3)

            # Parse thoughts
            thoughts: list[str] = []
            for line in gen_response.splitlines():
                m = re.match(r"^\s*Step\s+[A-Za-z0-9]+\s*:\s*(.+)$", line)
                if m:
                    t = m.group(1).strip()
                    if t:
                        thoughts.append(t)
            if not thoughts:
                for line in gen_response.splitlines():
                    m = re.match(r"^\s*[\d]+[.)]\s*(.+)$", line)
                    if m:
                        t = m.group(1).strip()
                        if t:
                            thoughts.append(t)
            if not thoughts:
                thoughts = [ln.strip() for ln in gen_response.splitlines()
                            if ln.strip() and len(ln.strip()) > 15]
            thoughts = thoughts[:k]

            state_log = {
                "state_idx": state_idx,
                "current_steps": state,
                "generation_prompt": gen_prompt,
                "generation_response": gen_response,
                "parsed_thoughts": thoughts,
                "evaluations": [],
            }

            for thought in thoughts:
                new_state = state + [thought]
                eval_steps_text = "\n".join(
                    f"  {i+1}. {s}" for i, s in enumerate(new_state))
                eval_prompt = EVALUATE_STATE_PROMPT.format(
                    problem=problem, steps=eval_steps_text)
                eval_response = llm(eval_prompt, max_tokens=16)
                time.sleep(0.2)

                raw = eval_response.strip().lower()
                score = 0.5
                verdict = "likely (default)"
                for label in ("impossible", "sure", "likely"):
                    if label in raw:
                        score = _EVAL_SCORE[label]
                        verdict = label
                        break

                state_log["evaluations"].append({
                    "thought": thought,
                    "new_state": new_state,
                    "eval_prompt": eval_prompt,
                    "eval_response": eval_response,
                    "verdict": verdict,
                    "score": score,
                })

                if score > 0.0:
                    candidates.append((score, new_state))

            depth_log["states"].append(state_log)

        if not candidates:
            depth_log["pruned_all"] = True
            log["depths"].append(depth_log)
            break

        candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        frontier = [state for _, state in candidates[:b]]
        depth_log["kept_states"] = [
            {"steps": state, "score": score}
            for score, state in candidates[:b]
        ]
        log["depths"].append(depth_log)

    # ── extract final answer ──────────────────────────────────────────────────
    for state in frontier:
        # 1) Check if any step already has \boxed{}
        for step in reversed(state):
            ans = _extract_boxed(step)
            if ans:
                log["final_answer_attempts"].append({
                    "method": "found_in_step",
                    "state": state,
                    "answer": ans,
                })
                log["predicted"] = ans
                return log

        # 2) Ask model for final answer
        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(state)) if state else "  (none)"
        final_prompt = FINAL_ANSWER_PROMPT.format(problem=problem, steps=steps_text)
        final_response = llm(final_prompt, max_tokens=1024)
        time.sleep(0.3)

        # Prompt ends with \boxed{ — prepend it for extraction
        full = "\\boxed{" + final_response
        ans = _extract_boxed(full) or _extract_boxed(final_response)

        # Fallback: look for "the answer is X" or bare expression on last line
        if ans is None:
            for pat in [
                r"(?:the\s+(?:final\s+)?answer\s+is|answer\s*[:=])\s*([^\n.]+)",
                r"=\s*(-?[\d\/\.\w\\{},\s\+\-\*\^()]+?)\s*(?:\.|$)",
            ]:
                import re as _re
                matches = list(_re.finditer(pat, final_response, _re.IGNORECASE))
                if matches:
                    ans = matches[-1].group(1).strip().rstrip(".")
                    break

        attempt = {
            "method": "final_answer_prompt",
            "state": state,
            "prompt": final_prompt,
            "response": final_response,
            "full_for_extraction": full[:200],
            "extracted": ans,
        }
        log["final_answer_attempts"].append(attempt)

        if ans:
            log["predicted"] = ans
            return log

    return log


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="p8",
                        choices=["p8", "p10", "both"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--b", type=int, default=3)
    parser.add_argument("--max_depth", type=int, default=4)
    args = parser.parse_args()

    keys = ["p8", "p10"] if args.problem == "both" else [args.problem]

    for key in keys:
        info = PROBLEMS[key]
        print(f"\n{'='*70}")
        print(f"Debugging: {info['label']}")
        print(f"Problem: {info['problem']}")
        print(f"Expected answer: {info['answer']}")
        print(f"Config: k={args.k}, b={args.b}, max_depth={args.max_depth}")
        print(f"{'='*70}\n")

        t0 = time.time()
        log = run_tot_debug(info["problem"], k=args.k, b=args.b, max_depth=args.max_depth)
        elapsed = time.time() - t0

        # ── print readable trace ──────────────────────────────────────────────
        for d in log["depths"]:
            print(f"\n{'─'*70}")
            print(f"DEPTH {d['depth']}  (frontier_size={d['frontier_size']})")
            print(f"{'─'*70}")
            for s in d["states"]:
                print(f"\n  State {s['state_idx']}: steps so far = {s['current_steps']}")
                print(f"  Generated {len(s['parsed_thoughts'])} thoughts:")
                for ev in s["evaluations"]:
                    score_str = {1.0: "SURE    ", 0.5: "LIKELY  ", 0.0: "IMPOSBL "}.get(ev["score"], "?       ")
                    print(f"    [{score_str}] {ev['thought'][:80]}")
            if d.get("kept_states"):
                print(f"\n  Kept {len(d['kept_states'])} states:")
                for ks in d["kept_states"]:
                    print(f"    score={ks['score']}  last_step={ks['steps'][-1][:60] if ks['steps'] else '(root)'!r}")

        print(f"\n{'─'*70}")
        print("FINAL ANSWER EXTRACTION ATTEMPTS:")
        for att in log["final_answer_attempts"]:
            print(f"\n  Method: {att['method']}")
            if "response" in att:
                print(f"  Prompt (last 200 chars): ...{att['prompt'][-200:]}")
                print(f"  Response: {att['response'][:300]}")
            print(f"  Extracted: {att.get('extracted') or att.get('answer')!r}")

        print(f"\n{'='*70}")
        print(f"RESULT: predicted={log['predicted']!r}  expected={info['answer']!r}")
        correct = log["predicted"] and log["predicted"].strip() == info["answer"].strip()
        print(f"Correct: {correct}")
        print(f"Elapsed: {elapsed:.1f}s   LLM calls so far: {_LLM_CALL_COUNT}")
        print(f"{'='*70}\n")

        # ── save full log ──────────────────────────────────────────────────────
        outfile = f"debug_{key}_trace.json"
        with open(outfile, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"Full trace saved → {outfile}")


if __name__ == "__main__":
    main()
