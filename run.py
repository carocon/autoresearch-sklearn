"""
run.py — Pure evaluator. No git logic.

Runs one experiment and prints the score. That's it.
The agent (Claude Code) reads the output and decides whether to
commit the improvement or revert train.py.

Usage:
    python run.py

Output (always printed to stdout for the agent to read):
    val_accuracy : 0.9667
    best so far  : 0.9333
    elapsed      : 0.4s
    RESULT       : IMPROVED  (+0.0334)   <- or: NO IMPROVEMENT

The agent is responsible for:
    - Reading program.md for research directions
    - Editing train.py with each hypothesis
    - Running `python run.py` to evaluate
    - Committing train.py if IMPROVED, reverting if not
    - Logging results to results.tsv
    - Repeating until done

Manual use (no agent):
    Edit train.py yourself, then run `python run.py` to score it.
"""

import os
import time
import importlib
import sys

BEST_SCORE_FILE = ".best_score"


def read_best():
    if os.path.exists(BEST_SCORE_FILE):
        with open(BEST_SCORE_FILE) as f:
            return float(f.read().strip())
    return 0.0


def write_best(score):
    with open(BEST_SCORE_FILE, "w") as f:
        f.write(str(score))


def run():
    import prepare
    # Always reload train so edits on disk are picked up
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train as train_module

    prepare.prepare()

    start = time.time()
    try:
        model, X_val, y_val = train_module.train()
        score = prepare.evaluate(model, X_val, y_val)
    except Exception as e:
        print(f"ERROR: {e}")
        print("RESULT: ERROR — revert train.py")
        sys.exit(1)

    elapsed = time.time() - start
    best = read_best()
    improved = score > best
    delta = score - best

    print(f"val_accuracy : {score:.4f}")
    print(f"best so far  : {best:.4f}")
    print(f"elapsed      : {elapsed:.1f}s")

    if improved:
        write_best(score)
        print(f"RESULT: IMPROVED (+{delta:.4f}) — commit train.py")
    else:
        print(f"RESULT: NO IMPROVEMENT ({delta:+.4f}) — revert train.py")


if __name__ == "__main__":
    run()
