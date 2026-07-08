"""Oversampling strategies for imbalanced regression targets.

Price is right-skewed (few Brand New / 512GB listings) and shipping is
zero-inflated (most listings ship free). Classic oversamplers target class
labels, so each strategy bins the continuous target into EQUAL-WIDTH bins
(rare = sparse value ranges, i.e. the expensive tail — equal-frequency
quantile bins would be balanced by construction and make oversampling a
no-op), oversamples rare bins, and keeps the continuous target intact.

All functions take (X, y) where X is the raw feature DataFrame (mixed
categorical/numeric) and return a resampled (X, y). They are applied to the
TRAINING split only — never to test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORICAL, NUMERIC, RANDOM_STATE


def _bins(y: pd.Series, n_bins: int = 5, min_size: int = 8) -> pd.Series:
    """Equal-width bins over the target; tiny bins merge into their nearest
    neighbour so SMOTE always has enough same-bin samples to interpolate."""
    bins = pd.cut(y, bins=n_bins, labels=False, include_lowest=True)
    counts = bins.value_counts()
    while len(counts) > 1 and counts.min() < min_size:
        small = counts.idxmin()
        nearest = min((b for b in counts.index if b != small), key=lambda b: abs(b - small))
        bins = bins.replace(small, nearest)
        counts = bins.value_counts()
    return bins


def none(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    return X, y


def _partial_balance(bins: pd.Series, factor: int = 3) -> dict:
    """Boost rare bins at most `factor`x instead of fully matching the majority —
    full balancing duplicates a sparse price tail ~25x and wrecks calibration."""
    counts = bins.value_counts()
    return {b: int(min(counts.max(), factor * c)) for b, c in counts.items()}


def random_over(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Duplicate rows of rare target bins (capped partial balancing)."""
    from imblearn.over_sampling import RandomOverSampler

    bins = _bins(y)
    ros = RandomOverSampler(sampling_strategy=_partial_balance(bins), random_state=RANDOM_STATE)
    idx = np.arange(len(X)).reshape(-1, 1)
    idx_res, _ = ros.fit_resample(idx, bins)
    take = idx_res.ravel()
    return X.iloc[take].reset_index(drop=True), y.iloc[take].reset_index(drop=True)


def smotenc(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTER-style synthesis: SMOTENC over features + continuous target.

    The target rides along as a numeric column so synthetic samples get an
    interpolated continuous target, not just a bin label.
    """
    from imblearn.over_sampling import SMOTENC

    bins = _bins(y)
    counts = bins.value_counts()
    k = max(1, min(5, counts.min() - 1))
    Xy = X.copy()
    # SMOTE rejects NaN; this runs on the train split only, so train-median fill is leakage-free
    Xy = Xy.fillna(Xy.median(numeric_only=True))
    Xy["__target__"] = y.to_numpy()
    cat_idx = [Xy.columns.get_loc(c) for c in CATEGORICAL]
    sm = SMOTENC(
        categorical_features=cat_idx,
        k_neighbors=k,
        sampling_strategy=_partial_balance(bins),
        random_state=RANDOM_STATE,
    )
    Xy_res, _ = sm.fit_resample(Xy, bins)
    y_res = Xy_res.pop("__target__")
    return Xy_res.reset_index(drop=True), y_res.reset_index(drop=True)


def gaussian_jitter(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Random oversampling of rare bins + Gaussian noise on numeric features
    (and target) of the duplicated rows — a cheap SMOGN-style variant."""
    X_res, y_res = random_over(X, y)
    rng = np.random.default_rng(RANDOM_STATE)
    dup = X_res.index >= len(X)  # appended rows are the duplicates
    X_res = X_res.copy()
    X_res[NUMERIC] = X_res[NUMERIC].astype(float)  # int columns reject float noise
    for col in NUMERIC:
        scale = 0.05 * X[col].std()
        if np.isfinite(scale) and scale > 0:
            noise = rng.normal(0, scale, size=int(dup.sum()))
            X_res.loc[dup, col] = (X_res.loc[dup, col] + noise).clip(lower=0)
    y_res = y_res.copy().astype(float)
    y_scale = 0.05 * y.std()
    y_res.loc[dup] = y_res.loc[dup] + rng.normal(0, y_scale, size=int(dup.sum()))
    return X_res, y_res.clip(lower=0)


STRATEGIES = {
    "none": none,
    "random_over": random_over,
    "smotenc": smotenc,
    "gaussian_jitter": gaussian_jitter,
}
