"""
train.py — AGENT'S SANDBOX.

This is the only file the agent is allowed to modify.
It must define a train() function that:
  - loads data from data.json
  - trains a model within the TIME_BUDGET_SECONDS limit
  - returns (model, X_val, y_val) so prepare.evaluate() can score it

The agent is free to change:
  - The model class (sklearn estimators, small MLPClassifier, ensembles, etc.)
  - Hyperparameters (n_estimators, max_depth, learning_rate, hidden layers, etc.)
  - Feature engineering (polynomial features, interactions, etc.)
  - Training strategy (cross-validation, early stopping, etc.)

The agent must NOT:
  - Import or modify prepare.py internals
  - Change the TIME_BUDGET_SECONDS variable
  - Change the function signature of train()
  - Hard-code the validation labels to cheat the metric
"""

import json
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

TIME_BUDGET_SECONDS = 60


def train():
    with open("data.json") as f:
        data = json.load(f)

    X_train = np.array(data["X_train"])
    y_train = np.array(data["y_train"])
    X_val = np.array(data["X_val"])
    y_val = np.array(data["y_val"])

    start = time.time()

    # --- Agent modifies below this line ---

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis(solver='svd')
    model.fit(X_train, y_train)

    # --- Agent modifies above this line ---

    elapsed = time.time() - start
    if elapsed > TIME_BUDGET_SECONDS:
        print(f"WARNING: training took {elapsed:.1f}s, over budget.")

    return model, X_val, y_val
