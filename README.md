# Tree of Thoughts: Deliberate Problem Solving with LLMs

> **CS 4782 / CS 5782: Deep Learning**
> 
> Ishaan Agarwal (`ia299`) · Rohan Modi (`rm2346`)

---

### Introduction

This repository is a course project re-implementation of the NeurIPS 2023 paper:

> **"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"**
> Yao et al., NeurIPS 2023 · [arXiv](https://arxiv.org/abs/2305.10601) · [Official Repo](https://github.com/princeton-nlp/tree-of-thought-llm)

Standard LLMs generate answers token-by-token in a single left-to-right pass. Chain-of-Thought (CoT) adds intermediate steps, but still follows one linear path. **Tree of Thoughts (ToT)** treats reasoning as a search problem: the model generates multiple candidate "thoughts" at each step, evaluates which branches are promising, and backtracks when a path looks wrong — mimicking human System 2 deliberate reasoning.

The original paper used GPT-4. **This project re-implements ToT using Gemini 2.5 Flash Lite** (and GPT-4o for one baseline) and benchmarks it across three tasks to test whether ToT's advantages generalize beyond the paper's original scope.

---

### Chosen Result

The paper's central claim is that ToT dramatically outperforms IO and CoT prompting on tasks where single-path greedy reasoning fails — reporting **74% accuracy on Game of 24** with ToT vs. **~4% with CoT** (Table 1 / Figure 3 in the original paper).

We reproduce this CoT baseline (**10% on puzzles 901–910 with GPT-4o**), and extend evaluation to:
- **Competition algebra (MATH dataset)**: ToT BFS vs. CoT vs. IO
- **5×5 Mini Crosswords**: ToT DFS with backtracking
- **GSM8K**: Grade-school math word problems (CoT / IO baseline)
- **Game of 24**: ToT BFS vs. CoT vs. IO

Our key finding: **ToT's advantage is task-dependent**. On Game of 24, where CoT reliably fails, the search benefit is real. On structured algebra, CoT already traces the natural solution path, so ToT's overhead yields no benefit.

---

### Repository Structure

```
.
├── code/
│   ├── math_tot.py            # IO / CoT / ToT BFS on MATH dataset (Gemini)
│   ├── debug_tot.py           # Full trace logging for single ToT problems
│   ├── cot_game24.py          # CoT baseline for Game of 24 (GPT-4o)
│   ├── crossword_tot.py       # ToT DFS for 5×5 mini crosswords (Gemini)
│   ├── run_crosswords.py      # Batch crossword runner
│   └── visualize_crossword.py # Renders DFS trace as PNG frames
├── data/
│   └── README.md              # Instructions for obtaining datasets
├── results/
│   ├── results.json           # MATH benchmark run 1 (seed=42, n=30)
│   ├── results_fixed.json     # MATH benchmark run 2 (seed=99, n=10)
│   ├── cot_game24_results.json
│   ├── crossword_data.json    # 3-puzzle crossword traces
│   ├── crossword_data_20.json # 20-puzzle crossword traces
│   ├── debug_p8_trace.json    # Full ToT trace, problem 8
│   └── debug_p10_trace.json   # Full ToT trace, problem 10
├── poster/
│   └── tot_poster.pdf
├── report/
│   └── group_treeofthoughts_2page_report.pdf
├── tree-of-thought-llm/       # Original repo (models.py swapped for Gemini backend)
├── crossword_frames/          # PNG visualization frames per puzzle
├── LICENSE
├── .gitignore
└── README.md
```

---

### Re-implementation Details

#### Tasks

| Task | Dataset | Search | Model |
|------|---------|--------|-------|
| Competition Math | `qwedsacf/competition_math` (HuggingFace) | BFS, depth=4, b∈{3,5} | Gemini 2.5 Flash Lite |
| Game of 24 | Puzzles 901–910 (paper's hard test set) | CoT baseline only | GPT-4o |
| 5×5 Mini Crosswords | GooBix scraped puzzles | DFS with backtracking | Gemini 2.5 Flash Lite |
| GSM8K | `openai/gsm8k` (HuggingFace) | CoT / IO baseline | Gemini 2.5 Flash Lite |

#### Framework Components

- **Thought Generator**: Samples `k=5` candidate next reasoning steps from the current state using a task-specific prompt.
- **State Evaluator**: LLM scores each candidate as `sure / likely / impossible`. Impossible branches are pruned; top `b` states advance.
- **BFS (Math)**: Maintains a frontier of the top `b` states per depth level, expanding each for 4 levels before extracting a boxed answer.
- **DFS (Crosswords)**: Proposes 5-letter word candidates with confidence ratings, aggregates scores across samples, and backtracks on `impossible` evaluations.

#### Key Modification

The original codebase targets the OpenAI API. `tree-of-thought-llm/src/tot/models.py` was replaced with a Gemini backend exposing the same `gpt(prompt, n=1) → list[str]` interface, leaving all prompt logic, search algorithms, and environment code untouched.

---

### Reproduction Steps

#### Prerequisites

```bash
pip install google-generativeai openai datasets matplotlib pillow
```

#### API Keys

```bash
export GEMINI_API_KEY="your_gemini_key_here"
export OPENAI_API_KEY="your_openai_key_here"   # only needed for cot_game24.py
```

#### Run MATH Benchmark (IO / CoT / ToT BFS)

```bash
cd code/
python3 math_tot.py --n_problems 20 --b 3 --output results.json
# Append a second run with a different seed:
python3 math_tot.py --n_problems 20 --seed 99 --b 3 --append_to results.json
```

Options: `--n_problems`, `--b` (beam width), `--seed`, `--output`, `--append_to`

#### Run ToT with Full Trace Logging

```bash
python3 debug_tot.py --problem p8    # single problem
python3 debug_tot.py --problem both  # problems 8 and 10
```

Saves full prompt/response/branch traces to `debug_p8_trace.json` / `debug_p10_trace.json`.

#### Run Game of 24 CoT Baseline

```bash
python3 cot_game24.py
```

Runs on puzzles 901–910. Validates that each answer uses exactly the four input numbers and evaluates to 24.

#### Run Crossword ToT DFS

```bash
python3 crossword_tot.py --puzzles 0 1 2 --n_generate 3 --output crossword_results.json
python3 crossword_tot.py --show_reasoning   # print LLM proposals and evaluations
```

#### Visualize Crossword DFS Trace

```bash
python3 visualize_crossword.py --puzzle 0 --outdir crossword_frames/puzzle_0/
python3 visualize_crossword.py --puzzle 0 --max 20   # cap at 20 frames
```

#### Computational Resources

All experiments are inference-only (no training). A full MATH run of 30 problems takes roughly 30–60 minutes on ToT due to ~60 LLM calls per problem. A GPU is not required. Expected API costs: ~$2–5 for a 30-problem MATH run with Gemini 2.5 Flash Lite; ~$1–2 for Game of 24 with GPT-4o.

---

### Results & Insights

#### Competition Math (30 problems, MATH dataset)

| Method | Accuracy | LLM Calls (total) | Avg. Time/Problem |
|--------|----------|-------------------|-------------------|
| IO | 84% | ~30 | 2–4s |
| CoT | 88% | ~30 | 15–27s |
| ToT BFS (b=3) | 88% | ~1,800 | 36–188s |

CoT outperforms ToT despite ToT using ~60× more compute. A separate run of 10 different problems (seed=99) showed CoT at 100%, reinforcing the pattern.

**Why?** Algebra problems have a natural linear solution path. The LLM evaluator cannot reliably distinguish good from bad intermediate steps in math, so BFS pruning misfires — cutting correct branches and retaining wrong ones.

#### Game of 24 (puzzles 901–910, GPT-4o)

| Method | Accuracy |
|--------|----------|
| CoT (ours) | 10% (1/10) |
| CoT (paper) | ~4% |
| ToT (paper) | ~74% |

Our CoT baseline reproduces the paper's reported failure mode. This confirms Game of 24 as a task where tree search provides genuine value — the arithmetic search space requires exploring multiple candidate sequences that CoT cannot recover from once it takes a wrong step.

#### Key Takeaway

ToT's advantage is real but narrow: **it helps when the task genuinely requires exploration and backtracking, and when the evaluator can reliably score partial states**. On tasks where CoT already follows the correct reasoning path naturally (structured math), ToT adds overhead without benefit. Task selection is the dominant factor.

---

### Conclusion

- ToT enables deliberate reasoning with exploration and backtracking — capabilities unavailable in CoT.
- Gains are large on combinatorial tasks (Game of 24, crosswords); minimal or negative on structured algebra.
- Compute cost is the main practical barrier: ToT requires 50–70× more LLM calls than CoT per problem.
- The evaluator quality is the binding constraint: on math, a weak evaluator means branching is essentially random, wasting compute.

**Future directions**: learned evaluators trained on labeled reasoning traces; MCTS integration for better exploration-exploitation balance; ToT applied to code generation where partial programs can be executed and evaluated exactly.

---

### References

1. Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." NeurIPS 2023.
2. Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.
3. Hendrycks et al. "Measuring Mathematical Problem Solving With the MATH Dataset." NeurIPS 2021.
4. Cobbe et al. "Training Verifiers to Solve Math Word Problems." arXiv 2021. (GSM8K)
5. Princeton NLP. Official ToT repository. https://github.com/princeton-nlp/tree-of-thought-llm

---

### Acknowledgements

This project was completed as the final project for **CS 4782 / CS 5782: Intro to Deep Learning** at Cornell University, Spring 2026. The implementation builds on the official Tree of Thoughts repository by Princeton NLP. All experiments use the Gemini 2.5 Flash Lite API (Google) and GPT-4o API (OpenAI).
