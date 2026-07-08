"""Model zoo: one end-to-end sklearn Pipeline per candidate.

Preprocessing lives INSIDE every pipeline (impute + scale numerics, one-hot
categoricals), so a saved artifact accepts raw listing fields directly —
there are no separate encoder files that could drift out of sync with the
model between training and serving.

GPU: XGBoost uses device="cuda", CatBoost task_type="GPU", TorchMLP trains on
CUDA when available. LightGBM pip wheels are CPU-only on Windows.
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, NUMERIC, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def build_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


class TorchMLP(BaseEstimator, RegressorMixin):
    """Small fully-connected net with early stopping, sklearn-compatible."""

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64),
        lr: float = 1e-3,
        epochs: int = 300,
        batch_size: int = 256,
        patience: int = 25,
        random_state: int = RANDOM_STATE,
    ):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def _net(self, n_in: int):
        from torch import nn

        layers: list = []
        for h in self.hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch

        torch.manual_seed(self.random_state)
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() + 1e-9)
        y = (y - self.y_mean_) / self.y_std_

        rng = np.random.default_rng(self.random_state)
        val = rng.random(len(X)) < 0.1
        Xt = torch.tensor(X[~val], device=self.device_)
        yt = torch.tensor(y[~val], device=self.device_)
        Xv = torch.tensor(X[val], device=self.device_)
        yv = torch.tensor(y[val], device=self.device_)

        self.model_ = self._net(X.shape[1]).to(self.device_)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        best, best_state, stale = np.inf, None, 0
        for _ in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(len(Xt), device=self.device_)
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(Xt[idx]).squeeze(-1), yt[idx])
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self.model_(Xv).squeeze(-1), yv))
            if vloss < best - 1e-5:
                best, stale = vloss, 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch

        X = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(X).squeeze(-1).cpu().numpy()
        return out * self.y_std_ + self.y_mean_


def zoo(gpu: bool = True, quick: bool = False) -> dict[str, BaseEstimator]:
    """Candidate estimators. quick=True keeps only the two fastest (tests/CI)."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    models: dict[str, BaseEstimator] = {
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "hist_gb": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "xgboost": XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            device="cuda" if gpu else "cpu",
            random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMRegressor(  # ponytail: CPU — pip wheels ship no CUDA on Windows
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=8,
            task_type="GPU" if gpu else "CPU",
            devices="0",
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        ),
    }
    if find_spec("torch"):  # optional heavyweight — absent in the Docker train image
        models["torch_mlp"] = TorchMLP()
    if quick:
        return {k: models[k] for k in ("ridge", "hist_gb")}
    return models


# randomized-search spaces, applied only to the winning (model, sampling) combo
SEARCH_SPACES: dict[str, dict[str, list]] = {
    "ridge": {"model__alpha": np.logspace(-3, 3, 25).tolist()},
    "hist_gb": {
        "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
        "model__max_iter": [200, 400, 800],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
    },
    "random_forest": {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": [0.5, 0.8, 1.0],
    },
    "xgboost": {
        "model__n_estimators": [300, 600, 1000],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__max_depth": [4, 6, 8, 10],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
        "model__reg_lambda": [0.5, 1.0, 5.0],
    },
    "lightgbm": {
        "model__n_estimators": [300, 600, 1000],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__num_leaves": [31, 63, 127],
        "model__min_child_samples": [10, 20, 40],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    },
    "catboost": {
        "model__iterations": [400, 800, 1200],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__depth": [4, 6, 8, 10],
        "model__l2_leaf_reg": [1, 3, 9],
    },
    "torch_mlp": {
        "model__hidden": [(64,), (128, 64), (256, 128, 64)],
        "model__lr": [3e-4, 1e-3, 3e-3],
        "model__batch_size": [128, 256],
    },
}
