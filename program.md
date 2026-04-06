# program.md — Research directions

You are an autonomous ML research agent optimising a classifier on the Iris dataset.
You own the full experiment loop, including git. Act autonomously.

## Setup (first time only)
```bash
git init
git config user.email "autoresearch@local"
git config user.name "AutoResearch"
pip install scikit-learn numpy
python prepare.py        # generates data.json
python run.py            # establishes baseline score in .best_score
git add .
git commit -m "baseline: initial setup"
```

## Your loop — repeat for every experiment

1. **Form a hypothesis** — decide what to change in `train.py` and why
2. **Edit `train.py`** — implement the change
3. **Evaluate** — run `python run.py` and read the output
4. **Commit or revert** based on the RESULT line:
   - `RESULT: IMPROVED` → `git add train.py && git commit -m "exp<N>: <what you changed> val_accuracy <score>"`
   - `RESULT: NO IMPROVEMENT` → `git checkout HEAD -- train.py`
5. **Log the result** — append one row to `results.tsv`:
   `<N>\t<timestamp>\t<score>\t<YES or no>\t<brief note>`
6. **Repeat**

## Rules
- You may ONLY edit `train.py`
- Do NOT touch `prepare.py`, `run.py`, or `data.json`
- Do NOT hard-code or peek at validation labels
- Stay within `TIME_BUDGET_SECONDS = 60` in train.py
- Always revert failed experiments before trying the next one
- Never skip the git step — the commit history is the memory of this run

## What you can change in train.py
- The model class (RandomForest, GradientBoosting, SVM, MLP, KNN, ensembles...)
- Any hyperparameters
- Feature engineering (PolynomialFeatures, interactions, PCA...)
- Training strategy (cross-validation for model selection, calibration...)

## Suggested research directions
1. Try SVM with RBF kernel (strong baseline for Iris)
2. Try GradientBoostingClassifier
3. Try MLPClassifier with different hidden layer sizes
4. Try a VotingClassifier ensemble across multiple model types
5. Try PolynomialFeatures before the model
6. Tune hyperparameters of whatever is working best

## What good looks like
- Baseline (RandomForest default): ~0.90
- Strong result: >= 0.9667
- Perfect: 1.0000 (achievable on this split)

## Reading the scoreboard
At any point you can review progress:
- `cat results.tsv` — full experiment log
- `git log --oneline` — only the kept improvements
- `cat .best_score` — current best val_accuracy
