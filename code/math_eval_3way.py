"""
3-Way MATH Benchmark: Single-Shot vs Chain-of-Thought vs Tree-of-Thoughts
--------------------------------------------------------------------------
Runs all three prompting strategies on the same problems from the
Hendrycks MATH dataset and prints a side-by-side comparison table.

Uses local Ollama — no API key needed.

Usage:
    python3 math_eval_3way.py
    python3 math_eval_3way.py --model llama3.1:8b --n 2 --subject algebra
    python3 math_eval_3way.py --model llama3.1:8b --n 3 --tot-k 3 --tot-b 2

Subjects: algebra | counting_and_probability | geometry |
          intermediate_algebra | number_theory | prealgebra | precalculus
"""

import argparse
import heapq
import difflib
import json
import re
import time
import urllib.request
import urllib.error
import sys
import datetime
from textwrap import dedent

try:
    from sympy import simplify
    from sympy.parsing.latex import parse_latex
    parse_latex("1")
    _SYMPY_OK = True
except Exception:
    _SYMPY_OK = False
    print("[WARNING] SymPy LaTeX parsing unavailable. Install antlr4-python3-runtime for symbolic answer matching. Falling back to string normalization.")

class Tee:
    def __init__(self, name, mode="w"):

        self.file = open(name, mode, encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def __del__(self):
        try:
            import sys
            if sys is not None:
                sys.stdout = self.stdout
            if hasattr(self, 'file') and not self.file.closed:
                self.file.close()
        except Exception:
            pass

    def write(self, data):
        self.stdout.write(data)
        self.file.write(self.ansi_escape.sub('', data))
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

from datasets import load_dataset




DEFAULT_MODEL   = "gemini-2.5-flash"
DEFAULT_N       = 2
DEFAULT_SUBJECT = "algebra"
DEFAULT_TOT_K   = 5
DEFAULT_TOT_B   = 5
DEFAULT_TOT_D   = 3

ALL_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]




class C:
    RESET   = "\033[0m";  BOLD    = "\033[1m"
    CYAN    = "\033[96m"; GREEN   = "\033[92m"
    YELLOW  = "\033[93m"; RED     = "\033[91m"
    GREY    = "\033[90m"; MAGENTA = "\033[95m"
    BLUE    = "\033[94m"; WHITE   = "\033[97m"

def hdr(title: str, width: int = 74) -> str:
    pad = max(0, (width - len(title) - 2) // 2)
    return f"\n{C.BOLD}{C.CYAN}{'─'*pad} {title} {'─'*pad}{C.RESET}\n"

def section(title: str, color: str = C.MAGENTA) -> str:
    return f"\n{C.BOLD}{color}▶ {title}{C.RESET}"




def _load_dotenv(path: str = ".env") -> None:
    import os
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_dotenv()




def _gemini_call(prompt: str, model: str, temperature: float, max_tokens: int,
                 timeout: int, thinking_budget: int = 0) -> str:
    import os
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[GEMINI ERROR: GOOGLE_API_KEY not set]"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,


            "thinkingConfig": {"thinkingBudget": thinking_budget},
        },
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            candidates = body.get("candidates", [])
            if not candidates:
                return f"[GEMINI ERROR: no candidates — {body}]"
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()
    except urllib.error.HTTPError as e:
        return f"[GEMINI HTTP {e.code}: {e.read().decode()[:200]}]"
    except urllib.error.URLError as e:
        return f"[GEMINI ERROR: {e}]"

def _ollama_call(prompt: str, model: str, temperature: float, max_tokens: int, timeout: int) -> str:
    payload = json.dumps({
        "model":  model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return body.get("message", {}).get("content", "").strip()
    except urllib.error.URLError as e:
        return f"[OLLAMA ERROR: {e}]"

def _openai_call(prompt: str, model: str, temperature: float, max_tokens: int, timeout: int) -> str:
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "[OPENAI ERROR: OPENAI_API_KEY not set]"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"].strip()
    except urllib.error.URLError as e:
        return f"[OPENAI ERROR: {e}]"

def llm_call(prompt: str, model: str, temperature: float = 0.0,
             max_tokens: int = 512, timeout: int = 120,
             thinking_budget: int = 0) -> str:
    if model.startswith("gemini"):
        return _gemini_call(prompt, model, temperature, max_tokens, timeout, thinking_budget)
    elif model.startswith("gpt"):
        return _openai_call(prompt, model, temperature, max_tokens, timeout)
    return _ollama_call(prompt, model, temperature, max_tokens, timeout)




def extract_boxed(text: str) -> str | None:
    """Extract value from the last \\boxed{...}, handling nested braces."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":  depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[start:i - 1].strip()

def normalize(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    s = s.replace("$", "").replace("\\,", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"^x=", "", s)
    return s.lower()

def _sympy_equal(a: str, b: str) -> bool | None:
    """Return True/False if SymPy can decide equality, None if it can't parse."""
    if not _SYMPY_OK:
        return None
    try:
        expr_a = parse_latex(a)
        expr_b = parse_latex(b)
        return simplify(expr_a - expr_b) == 0
    except Exception:
        return None

def answers_match(predicted: str | None, ground_truth: str) -> bool:
    if not predicted:
        return False
    gt   = extract_boxed(ground_truth) or ground_truth
    pred = extract_boxed(predicted)    or predicted


    if normalize(pred) == normalize(gt):
        return True



    sym_result = _sympy_equal(pred, gt)
    if sym_result is not None:
        return sym_result

    return False




def run_single_shot(problem: str, model: str) -> tuple[str, int, float]:
    """One prompt → one answer. No reasoning guidance."""
    prompt = dedent(f"""\
        Solve this math problem. Provide only the final answer enclosed in a
        boxed tag like \\boxed{ answer} .
        Problem: {problem}
    """)
    t0 = time.perf_counter()
    response = llm_call(prompt, model, temperature=0.0)
    return response, 1, time.perf_counter() - t0




COT_FEW_SHOT = dedent("""\
    Example 1:
    Problem: What is 15% of 80?
    Solution: 15% of 80 = (15/100) × 80 = 0.15 × 80 = 12
    ANSWER: \\boxed{12}

    Example 2:
    Problem: How many integers from 1 to 50 are divisible by 3?
    Solution: Divide 50 by 3: 50/3 ≈ 16.67, so there are 16 multiples of 3 (3,6,...,48).
    ANSWER: \\boxed{16}
""")

def run_chain_of_thought(problem: str, model: str) -> tuple[str, int, float]:
    """Few-shot CoT: show worked examples, then ask model to reason step-by-step."""
    prompt = dedent(f"""\
        You are an expert mathematician. Use the examples below as a guide.
        Reason step-by-step, then end with ANSWER: \\boxed{ final_answer} .

        {COT_FEW_SHOT}
        Now solve:
        Problem: {problem}
        Solution:
    """)
    t0 = time.perf_counter()
    response = llm_call(prompt, model, temperature=0.0, max_tokens=768)
    return response, 1, time.perf_counter() - t0




class TreeOfThoughts:
    def __init__(self, model: str, k: int = 5, b: int = 5,
                 max_depth: int = 3, sim_threshold: float = 0.85,
                 eval_samples: int = 3):
        self.model          = model
        self.k              = k
        self.b              = b
        self.max_depth      = max_depth
        self.sim_threshold  = sim_threshold
        if eval_samples < 1:
            raise ValueError(f"eval_samples must be >= 1, got {eval_samples}")
        self.eval_samples   = eval_samples
        self.api_calls      = 0

    def _is_duplicate(self, candidate: str, existing: list[str]) -> bool:
        """Return True if candidate is character-level similar to any thought in existing.

        Uses difflib.SequenceMatcher (not semantic/embedding similarity).
        Scope: within a single generation batch only, not across beam nodes.
        """
        c_norm = " ".join(candidate.lower().split())
        for ex in existing:
            ex_norm = " ".join(ex.lower().split())
            ratio = difflib.SequenceMatcher(None, c_norm, ex_norm, autojunk=False).ratio()
            if ratio >= self.sim_threshold:
                return True
        return False

    def _parse_proposals(self, raw: str) -> list[str]:
        """Parse [1]...[k] numbered proposals from a single LLM response.
        Falls back to double-newline splitting if numbered format not found."""


        first_marker = re.search(r'\[1\]', raw)
        if first_marker:
            raw = raw[first_marker.start():]
        parts = re.split(r'\[\d+\]', raw)
        candidates = [p.strip() for p in parts if p.strip() and len(p.strip()) > 40]
        if len(candidates) >= 2:
            return candidates

        return [p.strip() for p in raw.split('\n\n') if p.strip() and len(p.strip()) > 40]

    def _generate_thoughts(self, problem: str, steps_so_far: str, depth: int) -> list[str]:
        """Sample k solutions: one greedy (temp=0) anchor + (k-1) diverse (temp=0.7)."""
        prompt = dedent(f"""\
            You are an expert mathematician. Use the examples below as a guide.
            Reason step-by-step, then end with ANSWER: \\boxed{ final_answer} .

            {COT_FEW_SHOT}
            Now solve:
            Problem: {problem}
            {f"Progress so far:{chr(10)}{steps_so_far}{chr(10)}Continue from here." if steps_so_far else ""}
            Solution:
        """)
        thoughts: list[str] = []

        self.api_calls += 1
        anchor = llm_call(prompt, self.model, temperature=0.0, max_tokens=768)
        if anchor:
            thoughts.append(anchor)

        for _ in range(max(0, self.k - 1)):
            self.api_calls += 1
            t = llm_call(prompt, self.model, temperature=0.7, max_tokens=768)
            if t and not self._is_duplicate(t, thoughts):
                thoughts.append(t)
        return thoughts

    def _verify_solution(self, problem: str, steps: str) -> bool:
        """Second-pass check: independently solve with same prompt as thought-gen, compare answers."""
        candidate_answer = extract_boxed(steps)
        if not candidate_answer:
            return False


        prompt = dedent(f"""\
            You are an expert mathematician. Use the examples below as a guide.
            Reason step-by-step, then end with ANSWER: \\boxed{ final_answer} .

            {COT_FEW_SHOT}
            Now solve:
            Problem: {problem}
            Solution:
        """)
        self.api_calls += 1
        independent = llm_call(prompt, self.model, temperature=0.0, max_tokens=768)
        return answers_match(extract_boxed(independent) or independent, candidate_answer)

    def _consensus_thought(self, thoughts: list[str]) -> tuple[str | None, int, int]:
        """Self-consistency: group thoughts by their boxed answer; return
        (representative_thought, agree_count, total_with_boxed) for the most
        common answer. Returns (None, 0, 0) if no thoughts have boxed answers.
        """
        groups: dict[str, list[str]] = {}
        for t in thoughts:
            ans = extract_boxed(t)
            if ans:
                key = normalize(ans)
                groups.setdefault(key, []).append(t)
        total = sum(len(v) for v in groups.values())
        if not groups:
            return None, 0, 0
        best_key = max(groups, key=lambda k: len(groups[k]))
        best_group = groups[best_key]
        return best_group[0], len(best_group), total

    def _score_state(self, problem: str, steps: str) -> int:
        """0 = dead end, 1 = promising, 2 = complete and verified.

        Calls the evaluator self.eval_samples times, maps verdicts to weights
        (sure=2, likely=1, impossible=0), averages, then thresholds:
          avg >= 1.5 -> 2 (sure)   avg >= 0.5 -> 1 (likely)   else -> 0
        Score-2 paths are then independently verified by _verify_solution.
        """
        prompt = dedent(f"""\
            You are evaluating a partial solution to a math problem.

            Problem: {problem}
            Partial Solution:
            {steps}

            Analyze the partial solution for accuracy and direction.
            - Are the calculations correct?
            - Is the logic sound?
            - Is it moving towards a valid answer?

            Based on your analysis, provide a judgement:
            - "impossible" if there is a clear calculation error, logical fallacy, or it is going in a completely wrong direction.
            - "sure" if the problem is fully and correctly solved, ending with a \\boxed{ answer} .
            - "likely" if the steps so far are correct and making good progress towards the solution.

            Reply ONLY with one word: sure, likely, or impossible.
        """)



        latest_thought = steps.split("\n\n")[-1] if steps else ""
        if r"\boxed{" in latest_thought:
            if self._verify_solution(problem, steps):
                return 2


        total = 0
        for _ in range(self.eval_samples):
            self.api_calls += 1
            verdict = llm_call(prompt, self.model, temperature=0.1, max_tokens=10).lower()
            if "sure" in verdict:
                total += 2
            elif "impossible" not in verdict:
                total += 1

        avg = total / self.eval_samples
        if avg >= 0.5:
            return 1
        return 0

    def _solve_bfs(self, problem: str) -> tuple[str, int]:
        """Paper Algorithm 1: level-by-level BFS beam search.

        Expands all nodes at the current depth before going deeper.
        Keeps top-b states by score at each level.
        """

        beam: list[tuple[str, int]] = [("", 0)]
        greedy_fallback: str | None = None

        while beam:
            next_candidates: list[tuple[int, str, int]] = []

            for steps, depth in beam:
                if depth >= self.max_depth:
                    continue

                print(f"    {C.GREY}[BFS] depth={depth}, expanding node…{C.RESET}")
                thoughts = self._generate_thoughts(problem, steps, depth)


                if depth == 0 and greedy_fallback is None and thoughts:
                    greedy_fallback = thoughts[0]


                rep, agree, total = self._consensus_thought(thoughts)
                if rep and agree > total / 2 and agree >= 2:
                    new_steps = (steps + "\n\n" + rep).strip() if steps else rep
                    if self._verify_solution(problem, new_steps):
                        print(f"    {C.GREEN}[BFS] Consensus + verified: {agree}/{total} agree at depth {depth + 1}.{C.RESET}")
                        return new_steps, self.api_calls
                    print(f"    {C.YELLOW}[BFS] Consensus {agree}/{total} but verifier disagrees — falling through.{C.RESET}")

                for i, thought in enumerate(thoughts):
                    new_steps = (steps + "\n\n" + thought).strip() if steps else thought
                    score     = self._score_state(problem, new_steps)

                    short       = thought.replace('\n', ' ')[:80] + ("..." if len(thought) > 80 else "")
                    verdict_str = {0: "impossible", 1: "likely", 2: "sure"}[score]
                    color       = C.RED if score == 0 else (C.GREEN if score == 2 else C.YELLOW)
                    print(f"      {C.GREY}↳ T{i+1}: {short} -> {color}{verdict_str}{C.RESET}")

                    if score == 2:
                        print(f"    {C.GREEN}[BFS] Solution found at depth {depth + 1}.{C.RESET}")
                        return new_steps, self.api_calls

                    if score > 0:
                        next_candidates.append((score, new_steps, depth + 1))

            if not next_candidates:
                if greedy_fallback:
                    print(f"    {C.YELLOW}[BFS] Search exhausted — returning greedy fallback.{C.RESET}")
                    return greedy_fallback, self.api_calls
                print(f"    {C.RED}[BFS] All paths pruned — search failed.{C.RESET}")
                return "Search failed.", self.api_calls


            next_candidates.sort(key=lambda x: x[0], reverse=True)
            beam = [(steps, depth) for _, steps, depth in next_candidates[:self.b]]

        if greedy_fallback:
            print(f"    {C.YELLOW}[BFS] Depth limit reached — returning greedy fallback.{C.RESET}")
            return greedy_fallback, self.api_calls
        return "Search failed.", self.api_calls

    def _solve_dfs(self, problem: str, v_threshold: float = 0.5,
                   max_steps: int = 50) -> tuple[str, int]:
        """Paper Algorithm 2: DFS with pruning and backtracking.

        Always expands the most promising child first. Prunes subtrees where
        score <= v_threshold (score=0 with default threshold=0.5).
        Backtracks to the parent when all children are pruned.
        Hard step limit prevents infinite loops.
        """


        stack: list[tuple[str, int]] = [("", 0)]
        steps_taken = 0
        greedy_fallback: str | None = None

        while stack and steps_taken < max_steps:
            steps, depth = stack.pop()

            if depth >= self.max_depth:
                continue

            steps_taken += 1
            print(f"    {C.GREY}[DFS] step={steps_taken}/{max_steps}, depth={depth}, expanding node…{C.RESET}")
            thoughts = self._generate_thoughts(problem, steps, depth)
            if depth == 0 and greedy_fallback is None and thoughts:
                greedy_fallback = thoughts[0]


            rep, agree, total = self._consensus_thought(thoughts)
            if rep and agree > total / 2 and agree >= 2:
                new_steps = (steps + "\n\n" + rep).strip() if steps else rep
                if self._verify_solution(problem, new_steps):
                    print(f"    {C.GREEN}[DFS] Consensus + verified: {agree}/{total} agree at depth {depth + 1}.{C.RESET}")
                    return new_steps, self.api_calls
                print(f"    {C.YELLOW}[DFS] Consensus {agree}/{total} but verifier disagrees — falling through.{C.RESET}")

            candidates: list[tuple[int, str, int]] = []

            for i, thought in enumerate(thoughts):
                new_steps = (steps + "\n\n" + thought).strip() if steps else thought
                score     = self._score_state(problem, new_steps)

                short       = thought.replace('\n', ' ')[:80] + ("..." if len(thought) > 80 else "")
                verdict_str = {0: "impossible", 1: "likely", 2: "sure"}[score]
                color       = C.RED if score == 0 else (C.GREEN if score == 2 else C.YELLOW)
                print(f"      {C.GREY}↳ T{i+1}: {short} -> {color}{verdict_str}{C.RESET}")

                if score == 2:
                    print(f"    {C.GREEN}[DFS] Solution found at depth {depth + 1}.{C.RESET}")
                    return new_steps, self.api_calls

                if score > v_threshold:
                    candidates.append((score, new_steps, depth + 1))

            if not candidates:
                print(f"    {C.YELLOW}[DFS] All children pruned, backtracking…{C.RESET}")
                continue


            candidates.sort(key=lambda x: x[0])
            for score, new_steps, new_depth in candidates:
                stack.append((new_steps, new_depth))

        if greedy_fallback:
            if steps_taken >= max_steps:
                print(f"    {C.YELLOW}[DFS] Step limit reached — returning greedy fallback.{C.RESET}")
            else:
                print(f"    {C.YELLOW}[DFS] Stack exhausted — returning greedy fallback.{C.RESET}")
            return greedy_fallback, self.api_calls
        return "Search failed.", self.api_calls

    def _solve_best_first(self, problem: str) -> tuple[str, int]:
        seq_counter = 0
        greedy_fallback: str | None = None


        heap: list[tuple[int, int, str, int]] = []
        seq_counter += 1
        heapq.heappush(heap, (0, seq_counter, "", 0))

        while heap:
            neg_score, _, steps, depth = heapq.heappop(heap)

            if depth >= self.max_depth:
                continue

            print(f"    {C.GREY}[ToT] depth={depth}, expanding node (score={-neg_score})…{C.RESET}")
            thoughts = self._generate_thoughts(problem, steps, depth)
            if depth == 0 and greedy_fallback is None and thoughts:
                greedy_fallback = thoughts[0]


            rep, agree, total = self._consensus_thought(thoughts)
            if rep and agree > total / 2 and agree >= 2:
                new_steps = (steps + "\n\n" + rep).strip() if steps else rep
                if self._verify_solution(problem, new_steps):
                    print(f"    {C.GREEN}[ToT] Consensus + verified: {agree}/{total} agree at depth {depth + 1}.{C.RESET}")
                    return new_steps, self.api_calls
                print(f"    {C.YELLOW}[ToT] Consensus {agree}/{total} but verifier disagrees — falling through.{C.RESET}")

            candidates      = []
            any_thought_gen = False

            for i, thought in enumerate(thoughts):
                new_steps = (steps + "\n\n" + thought).strip() if steps else thought
                score     = self._score_state(problem, new_steps)

                short_thought = thought.replace('\n', ' ')[:80] + ("..." if len(thought) > 80 else "")
                verdict_str   = {0: "impossible", 1: "likely", 2: "sure"}[score]
                color         = C.RED if score == 0 else (C.GREEN if score == 2 else C.YELLOW)
                print(f"      {C.GREY}↳ T{i+1}: {short_thought} -> {color}{verdict_str}{C.RESET}")

                if score == 2:
                    print(f"    {C.GREEN}[ToT] Solution found at depth {depth + 1}.{C.RESET}")
                    return new_steps, self.api_calls

                if score > 0:
                    candidates.append((score, new_steps, depth + 1))
                any_thought_gen = True

            if not candidates:
                if any_thought_gen:
                    print(f"    {C.YELLOW}[ToT] All paths impossible. Regenerating with higher temperature.{C.RESET}")
                    for _ in range(self.k):
                        self.api_calls += 1
                        t = llm_call(
                            dedent(f"""\
                                You are an expert mathematician. The previous attempts at this step were all incorrect.
                                Try a completely different approach.

                                Problem: {problem}

                                Progress so far:
                                {steps if steps else "(none — start fresh)"}

                                Next logical step (try a different method):
                            """),
                            self.model, temperature=1.0, max_tokens=256
                        )
                        if t:
                            seq_counter += 1
                            heapq.heappush(heap, (-1, seq_counter, (steps + "\n\n" + t).strip() if steps else t, depth + 1))
                else:
                    if greedy_fallback:
                        print(f"    {C.YELLOW}[ToT] All paths pruned — returning greedy fallback.{C.RESET}")
                        return greedy_fallback, self.api_calls
                    print(f"    {C.RED}[ToT] All paths pruned — search failed.{C.RESET}")
                    return "Search failed.", self.api_calls
                continue


            candidates.sort(key=lambda x: x[0], reverse=True)
            for score, new_steps, new_depth in candidates[:self.b]:
                seq_counter += 1
                heapq.heappush(heap, (-score, seq_counter, new_steps, new_depth))

        if greedy_fallback:
            print(f"    {C.YELLOW}[ToT] Heap exhausted — returning greedy fallback.{C.RESET}")
            return greedy_fallback, self.api_calls
        return "Search failed.", self.api_calls

    def solve(self, problem: str, search_mode: str = "bfs",
              dfs_threshold: float = 0.5, dfs_max_steps: int = 50) -> tuple[str, int]:
        """Dispatch to the selected search algorithm.

        Args:
            search_mode: 'bfs' | 'dfs' | 'best_first'
            dfs_threshold: score threshold below which DFS prunes a branch (default 0.5)
            dfs_max_steps: hard step limit for DFS (default 50)
        """
        self.api_calls = 0
        if search_mode == "bfs":
            return self._solve_bfs(problem)
        if search_mode == "dfs":
            return self._solve_dfs(problem, v_threshold=dfs_threshold,
                                   max_steps=dfs_max_steps)
        if search_mode == "best_first":
            return self._solve_best_first(problem)
        raise ValueError(f"Unknown search_mode {search_mode!r}. Choose: bfs, dfs, best_first")


def run_tot(problem: str, model: str, k: int, b: int, depth: int,
            sim_threshold: float = 0.85, eval_samples: int = 3,
            search_mode: str = "bfs", dfs_threshold: float = 0.5,
            dfs_max_steps: int = 50) -> tuple[str, int, float]:
    solver = TreeOfThoughts(model=model, k=k, b=b, max_depth=depth,
                            sim_threshold=sim_threshold, eval_samples=eval_samples)
    t0 = time.perf_counter()
    result, api_calls = solver.solve(problem, search_mode=search_mode,
                                     dfs_threshold=dfs_threshold,
                                     dfs_max_steps=dfs_max_steps)
    return result, api_calls, time.perf_counter() - t0




def run_eval(model: str, n: int, subject: str, tot_k: int, tot_b: int,
             tot_d: int, level: str = None, sim_threshold: float = 0.85,
             eval_samples: int = 3, search_mode: str = "bfs",
             dfs_threshold: float = 0.5, dfs_max_steps: int = 50,
             dataset: str = "math"):
    print(hdr(f"{dataset.upper()} Benchmark  ·  Single-Shot | CoT | ToT"))
    print(f"  {C.BOLD}Model  :{C.RESET} {C.YELLOW}{model}{C.RESET}")
    print(f"  {C.BOLD}Subject:{C.RESET} {subject}   {C.BOLD}N:{C.RESET} {n}")
    if level:
        print(f"  {C.BOLD}Level  :{C.RESET} {level}")
    search_label = search_mode.upper() if search_mode != "all" else "BFS + DFS + BestFirst"
    print(f"  {C.BOLD}ToT    :{C.RESET} k={tot_k} thoughts, b={tot_b} beam, depth≤{tot_d}, "
          f"search={search_label}, eval_samples={eval_samples}")

    if subject not in ALL_SUBJECTS:
        print(f"{C.RED}Unknown subject. Choose from: {', '.join(ALL_SUBJECTS)}{C.RESET}")
        raise SystemExit(1)

    print(f"\n{C.GREY}Loading dataset {dataset}…{C.RESET}")
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
    else:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        if level:
            ds = ds.filter(lambda x: x["level"] == f"Level {level}")

    examples = list(ds.select(range(min(n, len(ds)))))
    print(f"{C.GREY}Loaded {len(examples)} example(s).{C.RESET}\n")

    records = []

    for idx, ex in enumerate(examples, 1):
        if dataset == "gsm8k":
            problem = ex["question"]
            raw_gt  = ex["answer"]
            gt      = raw_gt.split("####")[-1].strip() if "####" in raw_gt else raw_gt
            prob_level = "gsm8k"
            etype   = "gsm8k"
        else:
            problem = ex["problem"]
            gt      = ex["solution"]
            prob_level = ex.get("level", "?")
            etype   = ex.get("type", "?")

        print(hdr(f"Problem {idx}/{len(examples)}  —  {etype}  [{prob_level}]"))
        print(f"{C.BOLD}{C.MAGENTA}Problem:{C.RESET}\n{problem}\n")
        print(f"{C.BOLD}Ground truth answer:{C.RESET} {C.GREY}{extract_boxed(gt) or gt[:80]}{C.RESET}\n")


        print(section("Single-Shot", C.BLUE))
        ss_resp, ss_calls, ss_time = run_single_shot(problem, model)
        ss_pred    = extract_boxed(ss_resp) or ss_resp.strip()[-60:]
        ss_correct = answers_match(ss_pred, gt)
        print(f"\n{ss_resp}")
        print(f"\n  Predicted: {C.YELLOW}{ss_pred}{C.RESET}  "
              f"→ {C.GREEN+'✓ CORRECT'+C.RESET if ss_correct else C.RED+'✗ WRONG'+C.RESET}"
              f"  ({ss_time:.1f}s, {ss_calls} call)")


        print(section("Chain-of-Thought", C.CYAN))
        cot_resp, cot_calls, cot_time = run_chain_of_thought(problem, model)
        cot_pred    = extract_boxed(cot_resp) or cot_resp.strip()[-60:]
        cot_correct = answers_match(cot_pred, gt)
        print(f"\n{cot_resp}")
        print(f"\n  Predicted: {C.YELLOW}{cot_pred}{C.RESET}  "
              f"→ {C.GREEN+'✓ CORRECT'+C.RESET if cot_correct else C.RED+'✗ WRONG'+C.RESET}"
              f"  ({cot_time:.1f}s, {cot_calls} call)")


        searches = ["bfs", "dfs", "best_first"] if search_mode == "all" else [search_mode]
        tot_results: dict[str, tuple[str, int, float]] = {}

        for sm in searches:
            label = {"bfs": "ToT-BFS", "dfs": "ToT-DFS", "best_first": "ToT-BestFirst"}[sm]
            print(section(label, C.MAGENTA))
            print(f"  {C.GREY}k={tot_k}, b={tot_b}, depth≤{tot_d}, eval_samples={eval_samples}{C.RESET}")
            resp, calls, elapsed = run_tot(
                problem, model, tot_k, tot_b, tot_d,
                sim_threshold=sim_threshold, eval_samples=eval_samples,
                search_mode=sm, dfs_threshold=dfs_threshold, dfs_max_steps=dfs_max_steps,
            )
            pred    = extract_boxed(resp) or "Search failed"
            correct = answers_match(pred, gt)
            preview = resp[-600:] if len(resp) > 600 else resp
            print(f"\n{C.GREY}…(trajectory preview)…{C.RESET}\n{preview}")
            print(f"\n  Predicted: {C.YELLOW}{pred}{C.RESET}  "
                  f"→ {C.GREEN+'✓ CORRECT'+C.RESET if correct else C.RED+'✗ WRONG'+C.RESET}"
                  f"  ({elapsed:.1f}s, {calls} API calls)")
            tot_results[sm] = (pred, correct, elapsed, calls)


        def _tot(sm):
            if sm in tot_results:
                pred, correct, elapsed, calls = tot_results[sm]
                return pred, correct, round(elapsed, 1), calls
            return None, None, None, None

        bfs_pred, bfs_correct, bfs_time, bfs_calls     = _tot("bfs")
        dfs_pred, dfs_correct, dfs_time, dfs_calls     = _tot("dfs")
        bf_pred,  bf_correct,  bf_time,  bf_calls      = _tot("best_first")

        records.append({
            "problem_id":   idx,
            "type":         etype,
            "level":        prob_level,
            "gt_answer":    extract_boxed(gt),
            "ss_pred":      ss_pred,   "ss_correct":  ss_correct,
            "ss_time_s":    round(ss_time, 1), "ss_calls": ss_calls,
            "cot_pred":     cot_pred,  "cot_correct": cot_correct,
            "cot_time_s":   round(cot_time, 1), "cot_calls": cot_calls,
            "bfs_pred":     bfs_pred,  "bfs_correct": bfs_correct,
            "bfs_time_s":   bfs_time,  "bfs_calls":   bfs_calls,
            "dfs_pred":     dfs_pred,  "dfs_correct": dfs_correct,
            "dfs_time_s":   dfs_time,  "dfs_calls":   dfs_calls,
            "bf_pred":      bf_pred,   "bf_correct":  bf_correct,
            "bf_time_s":    bf_time,   "bf_calls":    bf_calls,
        })


    print(hdr("Summary"))

    def _acc(key_correct):
        vals = [r[key_correct] for r in records if r[key_correct] is not None]
        return sum(vals), len(vals)

    col_w = 22
    print(f"\n  {'Strategy':<{col_w}} {'Correct':>8} {'Accuracy':>10} {'Avg Time':>10} {'Avg Calls':>11}")
    print(f"  {'─'*67}")

    strategies = [
        ("Single-Shot",      "ss_correct",  "ss_time_s",  "ss_calls"),
        ("Chain-of-Thought", "cot_correct", "cot_time_s", "cot_calls"),
    ]
    if search_mode in ("bfs",  "all"): strategies.append(("ToT-BFS",       "bfs_correct", "bfs_time_s", "bfs_calls"))
    if search_mode in ("dfs",  "all"): strategies.append(("ToT-DFS",       "dfs_correct", "dfs_time_s", "dfs_calls"))
    if search_mode in ("best_first", "all"): strategies.append(("ToT-BestFirst", "bf_correct",  "bf_time_s",  "bf_calls"))

    for label, key_c, key_t, key_calls in strategies:
        n_ok, n_total = _acc(key_c)
        if n_total == 0:
            continue
        acc  = 100 * n_ok / n_total
        avgt = sum(r[key_t]    for r in records if r[key_t]    is not None) / n_total
        avgc = sum(r[key_calls] for r in records if r[key_calls] is not None) / n_total
        bar  = C.GREEN if n_ok == n_total else (C.YELLOW if n_ok > 0 else C.RED)
        print(f"  {label:<{col_w}} {bar}{n_ok}/{n_total}{C.RESET}  "
              f"  {acc:>6.0f}%   {avgt:>8.1f}s   {avgc:>9.1f}")

    print(f"\n  {C.GREY}Note: accuracy uses \\boxed{{ }}  extraction + LaTeX normalization.{C.RESET}\n")


    tot_col_keys = []
    if search_mode in ("bfs",        "all"): tot_col_keys.append(("BFS",       "bfs_pred",  "bfs_correct"))
    if search_mode in ("dfs",        "all"): tot_col_keys.append(("DFS",       "dfs_pred",  "dfs_correct"))
    if search_mode in ("best_first", "all"): tot_col_keys.append(("BestFirst", "bf_pred",   "bf_correct"))

    header = f"  {'#':<4} {'Level':<8} {'SS':<12} {'CoT':<12}"
    for name, _, _ in tot_col_keys:
        header += f" {name:<12}"
    header += " GT"
    print(f"\n  {C.BOLD}Per-problem breakdown:{C.RESET}")
    print(header)
    print(f"  {'─'*90}")

    for r in records:
        def fmt(pred, ok):
            if pred is None:
                return f"{'—':<12}"
            color = C.GREEN if ok else C.RED
            p = str(pred).replace('\n', ' ')
            p = p[:10] + ".." if len(p) > 10 else p
            return f"{color}{p:<12}{C.RESET}"

        row = (f"  {r['problem_id']:<4} {r['level']:<8} "
               f"{fmt(r['ss_pred'], r['ss_correct'])}"
               f"{fmt(r['cot_pred'], r['cot_correct'])}")
        for _, pred_key, ok_key in tot_col_keys:
            row += fmt(r[pred_key], r[ok_key])
        row += f" {C.GREY}{r['gt_answer']}{C.RESET}"
        print(row)
    print()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-way MATH benchmark: Single-Shot | CoT | ToT via local Ollama"
    )
    parser.add_argument("--model",   default=DEFAULT_MODEL,   help="Model to use (default: gemini-2.5-flash)")
    parser.add_argument("--dataset", default="math", choices=["math", "gsm8k"], help="Dataset to evaluate (default: math)")
    parser.add_argument("--n",       default=DEFAULT_N, type=int, help="Number of examples (default: 2)")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT,  help="MATH subject (default: algebra)")
    parser.add_argument("--tot-k",   default=DEFAULT_TOT_K, type=int, help="ToT thoughts per node (default: 5)")
    parser.add_argument("--tot-b",   default=DEFAULT_TOT_B, type=int, help="ToT beam width (default: 5)")
    parser.add_argument("--tot-d",   default=DEFAULT_TOT_D, type=int, help="ToT max depth (default: 3)")
    parser.add_argument("--level",   default=None, help="Filter by difficulty level (e.g. '5' for Level 5)")
    parser.add_argument("--sim-threshold", default=0.85, type=float,
                        help="Character-level dedup similarity threshold 0–1 (default: 0.85)")
    parser.add_argument("--search", default="bfs",
                        choices=["bfs", "dfs", "best_first", "all"],
                        help="ToT search algorithm (default: bfs). Use 'all' to compare all three.")
    parser.add_argument("--eval-samples", default=3, type=int,
                        help="Evaluator calls to aggregate per state (default: 3)")
    parser.add_argument("--dfs-threshold", default=0.5, type=float,
                        help="DFS pruning threshold — states scored below this are pruned (default: 0.5)")
    parser.add_argument("--dfs-max-steps", default=50, type=int,
                        help="DFS hard step limit (default: 50)")
    args = parser.parse_args()

    log_file = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = Tee(log_file)
    print(f"Logging output to {log_file}...\n")

    run_eval(
        model          = args.model,
        n              = args.n,
        subject        = args.subject,
        tot_k          = args.tot_k,
        tot_b          = args.tot_b,
        tot_d          = args.tot_d,
        level          = args.level,
        sim_threshold  = args.sim_threshold,
        eval_samples   = args.eval_samples,
        search_mode    = args.search,
        dfs_threshold  = args.dfs_threshold,
        dfs_max_steps  = args.dfs_max_steps,
        dataset        = args.dataset,
    )

