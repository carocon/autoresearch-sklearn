"""
prepare.py — FIXED. Do not modify.

Loads Iris, splits into train/val, saves to disk.
Defines the evaluation metric: val_accuracy (higher is better).
This file is the fairness guarantee — every experiment is scored identically.
"""

import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def prepare():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    data = {
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_val": X_val.tolist(),
        "y_val": y_val.tolist(),
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
    }

    with open("data.json", "w") as f:
        json.dump(data, f)

    print(f"Data prepared: {len(X_train)} train, {len(X_val)} val samples.")
    return data


def evaluate(model, X_val, y_val):
    """
    Returns val_accuracy in [0, 1]. Higher is better.
    This is the single scalar metric the agent optimises.
    """
    preds = model.predict(X_val)
    return float(np.mean(preds == y_val))


if __name__ == "__main__":
    prepare()
