# Data

Datasets used in this project are loaded at runtime and are not committed to this repository.

## Competition Math (MATH dataset)

Loaded automatically via HuggingFace `datasets`:

```bash
pip install datasets
```

The script `code/math_tot.py` streams from `qwedsacf/competition_math` (train split).
No manual download required. A HuggingFace token is not required but will increase rate limits:

```bash
export HF_TOKEN="your_token_here"   # optional
```

## Game of 24

The puzzle file `24.csv` is part of the official Tree of Thoughts repository.
It is included at `tree-of-thought-llm/src/tot/data/24/24.csv` after cloning that submodule.
`code/cot_game24.py` reads from that path directly.

## 5×5 Mini Crosswords

Pre-processed puzzle files are stored in `results/`:

- `results/crossword_data.json` — 3 puzzles used in initial experiments
- `results/crossword_data_20.json` — 20 puzzles used in extended experiments

Each file is a JSON array where each entry is `[clues, board_gt]`:
- `clues`: list of 10 strings (h1–h5 horizontal, v1–v5 vertical)
- `board_gt`: flat list of 25 letters (row-major, 5×5 grid)
