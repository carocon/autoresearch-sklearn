# nanoml

A CPU-friendly autonomous ML research loop, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The idea: point an AI coding agent at this repo, describe your research directions in `program.md`, and walk away. The agent iterates on `train.py`, evaluates each change against a fixed metric, commits improvements, and reverts failures — autonomously, overnight, on a single CPU.

![ratchet loop](https://img.shields.io/badge/experiments-autonomous-blue) ![license](https://img.shields.io/badge/license-MIT-green)

## How it works

Three files do all the work:

| File | Who touches it | What it does |
|---|---|---|
| `prepare.py` | Nobody | Fixed data prep + metric (`val_accuracy`). The fairness guarantee. |
| `train.py` | The agent | sklearn model, hyperparameters, feature engineering. Everything is fair game. |
| `program.md` | You | Plain-English research directions. This is your only job. |

The agent runs on a **feature branch**. Every improvement is committed there. `main` stays clean as the original baseline. When you wake up, you review the branch, cherry-pick what you like, or merge it all.

## Quick start

**Requirements:** Python 3.10+, git, a Claude Code or Codex subscription.

```bash
# 1. Clone and install
git clone https://github.com/yourusername/nanoml
cd nanoml
pip install scikit-learn numpy

# 2. Prepare data (one-time, takes a second)
python prepare.py

# 3. Establish the baseline
python run.py
git add .
git commit -m "baseline"

# 4. Create a research branch
git checkout -b research/run-001
```

## Running the agent

Point Claude Code at the repo and say:

```
Read program.md and run 20 experiments autonomously on this branch.
```

Or with Codex / any other agent that can run shell commands and edit files.

The agent will:
- Read `program.md` for research directions
- Propose a change to `train.py`
- Run `python run.py` to evaluate
- Commit improvements, revert failures
- Log everything to `results.tsv`
- Repeat

## Reviewing results

```bash
git log --oneline          # only the kept improvements
cat results.tsv            # full experiment log including failures
cat .best_score            # current best val_accuracy
git diff main train.py     # what the agent changed overall
```

## Tuning the research directions

Edit `program.md` before starting a run. The quality of your Markdown is the quality of your research org. Karpathy calls this "programming the research org in Markdown" — the durable artifact from a run is the instruction file that produced it, not just the code changes.

## Baseline

- Dataset: Iris (120 train / 30 val, stratified, fixed seed)
- Metric: `val_accuracy` — higher is better
- Default model: `RandomForestClassifier` — scores ~0.90
- Theoretical ceiling: 1.0000 (achievable on this split)
- Time budget: 60 seconds per experiment

## License

MIT
