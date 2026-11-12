"""Modele klasyfikacji statystycznej tekstu (Regresja Logistyczna oraz Progi/Reguly).

Zawiera:
- RuleBasedThresholdClassifier: jawny klasyfikator progowy
- StatisticalLogisticClassifier: pipeline StandardScaler + LogisticRegression
- Analiza wag cech (interpretowalnosc)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text


class RuleBasedThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Prosty klasyfikator progowy oparty na drzewie decyzyjnym niskiej glebokosci (progi)."""

    def __init__(self, max_depth: int = 2, random_state: int = 42) -> None:
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree_ = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            criterion="gini",
        )
        self.feature_names_in_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence[int] | np.ndarray) -> RuleBasedThresholdClassifier:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            self.feature_names_in_ = [f"feat_{i}" for i in range(X_arr.shape[1])]

        self.tree_.fit(X_arr, y)
        self.classes_ = self.tree_.classes_
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.tree_.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.tree_.predict_proba(X_arr)

    def export_rules(self) -> str:
        return export_text(self.tree_, feature_names=self.feature_names_in_)


class StatisticalLogisticClassifier:
    """Klasyfikator regresji logistycznej z normalizacja cech (StandardScaler)."""

    def __init__(self, C: float = 1.0, random_state: int = 42, max_iter: int = 1000) -> None:
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, random_state=random_state, max_iter=max_iter, class_weight="balanced")),
        ])
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: Sequence[int] | np.ndarray) -> StatisticalLogisticClassifier:
        self.feature_names = list(X.columns)
        self.pipeline.fit(X.values, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.pipeline.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.pipeline.predict_proba(X_arr)

    def get_feature_importances(self) -> pd.DataFrame:
        """Zwraca wagi (wspolczynniki) cech: dodatnie wskazuja na klase AI (1), ujemne na Human (0)."""
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        weights = clf.coef_[0]
        df = pd.DataFrame({
            "feature": self.feature_names,
            "weight": weights,
            "abs_weight": np.abs(weights),
            "direction": ["AI" if w > 0 else "Human" for w in weights],
        }).sort_values(by="abs_weight", ascending=False).reset_index(drop=True)
        return df

