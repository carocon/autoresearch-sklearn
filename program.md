# program.md — Research directions

You are an autonomous ML research agent optimising a classifier on the Iris dataset.
You own the full experiment loop, including git. Act autonomously without asking for confirmation.

## Your loop — repeat for every experiment

1. **Form a hypothesis** — decide what to change in `train.py` and why
2. **Edit `train.py`** — implement the change
3. **Evaluate** — run `uv run python run.py` and read the RESULT line
4. **Commit or revert:**
   - `RESULT: IMPROVED` → `git add train.py && git commit -m "exp<N>: <description> val_accuracy <score>"`
   - `RESULT: NO IMPROVEMENT` → `git checkout HEAD -- train.py`
5. **Log** — append one row to `results.tsv`:
   `<N>\t<timestamp>\t<score>\t<YES or no>\t<brief note>`
6. **Repeat**

## Rules
- Edit ONLY `train.py`
- Do NOT touch `prepare.py`, `run.py`, `data.json`, or `README.md`
- Do NOT hard-code or peek at validation labels
- Stay within `TIME_BUDGET_SECONDS = 60`
- Always revert before trying the next hypothesis
- Never skip the git step — the commit history is the memory of this run
- You are on a feature branch — do not merge or push to main

## What you can change in train.py
- Model class (RandomForest, GradientBoosting, SVM, MLP, KNN, ensembles...)
- Any hyperparameters
- Feature engineering (PolynomialFeatures, interactions, PCA...)
- Training strategy (cross-validation for model selection, calibration...)

## Scoreboard
- `cat .best_score` — current best
- `git log --oneline` — kept improvements only
- `cat results.tsv` — full log including failures

