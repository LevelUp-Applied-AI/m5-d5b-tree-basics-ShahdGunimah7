"""Autograder tests for Drill 5B — Tree-Based Model Basics."""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drill import train_decision_tree, get_feature_importances, train_balanced_forest

FEATURES = ["tenure", "monthly_charges", "total_charges",
            "num_support_calls", "senior_citizen", "has_partner",
            "has_dependents", "contract_months"]


@pytest.fixture
def data():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "telecom_churn.csv")
    )
    X = df[FEATURES]
    y = df["churned"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def test_decision_tree_trained(data):
    X_train, X_test, y_train, y_test = data
    model = train_decision_tree(X_train, y_train)
    assert model is not None, "train_decision_tree returned None"
    from sklearn.tree import DecisionTreeClassifier
    assert isinstance(model, DecisionTreeClassifier), "Must return DecisionTreeClassifier"
    assert model.max_depth == 5, f"max_depth should be 5, got {model.max_depth}"
    assert hasattr(model, "classes_"), "Model must be fitted"


def test_feature_importances(data):
    X_train, X_test, y_train, y_test = data
    model = train_decision_tree(X_train, y_train)
    assert model is not None
    importances = get_feature_importances(model, FEATURES)
    assert importances is not None, "get_feature_importances returned None"
    assert len(importances) == len(FEATURES), f"Expected {len(FEATURES)} features"
    total = sum(importances.values())
    assert abs(total - 1.0) < 0.01, f"Importances should sum to ~1.0, got {total}"
    values = list(importances.values())
    assert values == sorted(values, reverse=True), "Importances should be sorted descending"


def test_random_forest_balanced(data):
    X_train, X_test, y_train, y_test = data
    metrics = train_balanced_forest(X_train, y_train, X_test, y_test)
    assert metrics is not None, "train_balanced_forest returned None"
    for key in ["precision", "recall", "f1"]:
        assert key in metrics, f"Missing key: {key}"
        assert metrics[key] > 0, f"{key} should be > 0"


def test_balanced_improves_recall(data):
    X_train, X_test, y_train, y_test = data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import recall_score

    # Baseline without balancing
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default.fit(X_train, y_train)
    recall_default = recall_score(y_test, rf_default.predict(X_test))

    # Balanced
    metrics = train_balanced_forest(X_train, y_train, X_test, y_test)
    assert metrics is not None
    assert metrics["recall"] > recall_default, \
        f"Balanced recall ({metrics['recall']:.3f}) should exceed default ({recall_default:.3f})"
