"""
MATH Dataset Evaluation with Local Ollama LLM
----------------------------------------------
Loads a few examples from the Hendrycks MATH dataset (HuggingFace),
sends them to a local Ollama model, and prints a structured report.

Usage:
    python3 math_eval.py
    python3 math_eval.py --model llama3.1:8b --n 3 --subject algebra

Available subjects (dataset config names):
    algebra, counting_and_probability, geometry,
    intermediate_algebra, number_theory, prealgebra, precalculus
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.error
from textwrap import dedent

from datasets import load_dataset




DEFAULT_MODEL   = "llama3.1:8b"
DEFAULT_N       = 2
DEFAULT_SUBJECT = "algebra"
OLLAMA_URL      = "http://localhost:11434/api/generate"


ALL_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]




class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    GREY   = "\033[90m"
    MAGENTA = "\033[95m"

def hdr(title: str, width: int = 72) -> str:
    pad = (width - len(title) - 2) // 2
    return f"\n{C.BOLD}{C.CYAN}{'─'*pad} {title} {'─'*pad}{C.RESET}\n"




def ollama_generate(prompt: str, model: str, timeout: int = 120) -> str:
    """Call Ollama's /api/generate endpoint and return the full response text."""
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 512,
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return body.get("response", "").strip()
    except urllib.error.URLError as e:
        return f"[ERROR contacting Ollama: {e}]"




SYSTEM_PROMPT = dedent("""\
    You are an expert mathematician. Solve the following problem step-by-step.
    Show your reasoning clearly, then on the very last line write:
    FINAL ANSWER: <your answer here>
    The final answer should be a concise mathematical expression or number.
""")

def build_prompt(problem: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nProblem:\n{problem}\n\nSolution:"




def extract_final_answer(text: str) -> str | None:
    """Pull out the value after 'FINAL ANSWER:' if the model followed instructions."""
    match = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_boxed(text: str) -> str | None:
    """Extract the value inside the last \\boxed{...} in a string."""

    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip()

def normalize(ans: str) -> str:
    """Strip LaTeX formatting and whitespace for rough string comparison."""

    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("$", "").replace("\\,", "").replace("\\!", "")
    ans = ans.replace("\\left", "").replace("\\right", "")

    ans = ans.lower()
    return ans

def is_correct(predicted: str | None, ground_truth: str) -> bool:
    """Compare model's predicted answer against the ground truth.

    Ground truth is the full solution; we try to extract its \\boxed{} answer.
    We also try to extract \\boxed{} from the model prediction if present.
    """
    if predicted is None:
        return False


    gt_boxed   = extract_boxed(ground_truth) or ground_truth
    pred_boxed = extract_boxed(predicted)    or predicted

    return normalize(pred_boxed) == normalize(gt_boxed)




def run_eval(model: str, n: int, subject: str | None):
    print(hdr("MATH Dataset  ×  Ollama Local LLM Eval"))
    print(f"  {C.BOLD}Model   :{C.RESET} {C.YELLOW}{model}{C.RESET}")
    print(f"  {C.BOLD}Examples:{C.RESET} {n}")
    print(f"  {C.BOLD}Subject :{C.RESET} {subject or 'any'}")
    print()


    print(f"{C.GREY}Loading MATH dataset from HuggingFace…{C.RESET}")


    cfg = subject if subject else "algebra"
    if cfg not in ALL_SUBJECTS:
        print(f"{C.RED}Unknown subject '{cfg}'. Choose from: {', '.join(ALL_SUBJECTS)}{C.RESET}")
        raise SystemExit(1)

    ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")


    examples = list(ds.select(range(min(n, len(ds)))))
    print(f"{C.GREY}Loaded {len(examples)} example(s).{C.RESET}\n")

    results = []

    for i, ex in enumerate(examples, 1):
        print(hdr(f"Example {i} / {len(examples)}"))

        problem      = ex["problem"]
        ground_truth = ex["solution"]
        level        = ex.get("level", "?")
        ex_type      = ex.get("type", "?")

        print(f"{C.BOLD}Subject:{C.RESET} {ex_type}   {C.BOLD}Level:{C.RESET} {level}\n")
        print(f"{C.BOLD}{C.MAGENTA}Problem:{C.RESET}\n{problem}\n")
        print(f"{C.BOLD}Ground Truth Solution:{C.RESET}")
        print(f"{C.GREY}{ground_truth}{C.RESET}\n")

        prompt = build_prompt(problem)

        print(f"{C.BOLD}Querying {model}…{C.RESET}")
        t0 = time.perf_counter()
        response = ollama_generate(prompt, model)
        elapsed  = time.perf_counter() - t0

        predicted = extract_final_answer(response)
        gt_answer  = extract_boxed(ground_truth) or ground_truth
        correct    = is_correct(predicted, ground_truth)


        print(f"\n{C.BOLD}{C.CYAN}Model Response:{C.RESET}")
        print(response)

        print(f"\n{C.BOLD}Predicted answer :{C.RESET} {C.YELLOW}{predicted or '[not found]'}{C.RESET}")
        print(f"{C.BOLD}Expected answer  :{C.RESET} {C.GREY}{gt_answer}{C.RESET}")
        status_str = f"{C.GREEN}✓ CORRECT{C.RESET}" if correct else f"{C.RED}✗ WRONG{C.RESET}"
        print(f"{C.BOLD}Result           :{C.RESET} {status_str}")
        print(f"{C.BOLD}Time             :{C.RESET} {elapsed:.1f}s")

        results.append({
            "index":      i,
            "type":       ex_type,
            "level":      level,
            "problem":    problem[:120] + "…" if len(problem) > 120 else problem,
            "predicted":  predicted,
            "correct":    correct,
            "time_s":     round(elapsed, 2),
        })


    print(hdr("Summary"))
    n_correct = sum(r["correct"] for r in results)
    print(f"  {C.BOLD}Accuracy:{C.RESET} {n_correct}/{len(results)}")
    print()
    print(f"  {'#':<4} {'Type':<20} {'Level':<8} {'Time':>7}  {'Result'}")
    print(f"  {'─'*55}")
    for r in results:
        tag = f"{C.GREEN}✓{C.RESET}" if r["correct"] else f"{C.RED}✗{C.RESET}"
        print(f"  {r['index']:<4} {r['type']:<20} {r['level']:<8} {r['time_s']:>6.1f}s  {tag}")
    print()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MATH dataset on a local Ollama model")
    parser.add_argument("--model",   default=DEFAULT_MODEL,   help="Ollama model name  (default: llama3.1:8b)")
    parser.add_argument("--n",       default=DEFAULT_N, type=int, help="Number of examples  (default: 2)")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT,  help="Filter by subject, e.g. algebra | geometry | number_theory")
    args = parser.parse_args()

    run_eval(model=args.model, n=args.n, subject=args.subject)

