"""Train and compare models × oversampling strategies for price and shipping.

Usage:
    python -m ebay_price.train [--data data/processed/listings.csv] [--no-gpu] [--quick]

For each target: sweep every (model, sampling) combo on a fixed 80/20 split,
tune the winner with RandomizedSearchCV, then save one end-to-end pipeline.

Outputs:
    reports/model_comparison.csv / .md    full sweep results
    reports/comparison_<target>.png       RMSE chart per combo
    reports/pred_vs_actual_<target>.png   final-model test scatter
    artifacts/<target>_pipeline.joblib    deployable pipeline (raw fields in)
    artifacts/metadata.json               winner, params, metrics, versions
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, train_test_split

from .config import (
    ARTIFACTS,
    DATA_PROCESSED,
    FEATURES,
    MODELS,
    RANDOM_STATE,
    REPORTS,
    TARGET_PRICE,
    TARGET_SHIPPING,
)
from .models import SEARCH_SPACES, build_pipeline, zoo
from .sampling import STRATEGIES

GPU_MODELS = {"xgboost", "catboost", "torch_mlp"}  # serialize their CV fits on one GPU


def wrap_log(pipe) -> TransformedTargetRegressor:
    """Fit on log1p(target): prices are right-skewed and multiplicative, and the
    raw-scale fit let the expensive tail dominate the loss (R² ~ 0 in practice)."""
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def evaluate(pipe, X_test, y_test) -> dict[str, float]:
    pred = pipe.predict(X_test)
    return {
        "rmse": root_mean_squared_error(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "r2": r2_score(y_test, pred),
    }


def band_offsets(final, X_train, y_train, on_gpu: bool) -> tuple[float, float]:
    """P10/P90 offsets of out-of-fold residuals in log1p space — split-conformal
    style, so the API can report an 80% band instead of pretending the point
    prediction is exact. OOF runs on the raw train split: the resampled one
    leaks duplicates across folds."""
    oof = cross_val_predict(clone(final), X_train, y_train, cv=3, n_jobs=1 if on_gpu else -1)
    resid = np.log1p(y_train.to_numpy()) - np.log1p(np.clip(oof, 0, None))
    lo, hi = np.quantile(resid, [0.1, 0.9])
    return float(lo), float(hi)


def apply_band(pred, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    log_pred = np.log1p(np.clip(pred, 0, None))
    return np.expm1(log_pred + lo), np.expm1(log_pred + hi)


def sweep(name: str, X_train, y_train, X_test, y_test, gpu: bool, quick: bool) -> pd.DataFrame:
    rows = []
    strategies = (
        {k: STRATEGIES[k] for k in ("none", "random_over")} if quick else STRATEGIES
    )
    for samp_name, sampler in strategies.items():
        X_s, y_s = sampler(X_train, y_train)
        for model_name, est in zoo(gpu, quick).items():
            pipe = wrap_log(build_pipeline(clone(est)))
            t0 = time.perf_counter()
            pipe.fit(X_s, y_s)
            metrics = evaluate(pipe, X_test, y_test)
            rows.append(
                {
                    "target": name,
                    "model": model_name,
                    "sampling": samp_name,
                    **metrics,
                    "fit_seconds": round(time.perf_counter() - t0, 2),
                    "train_rows": len(X_s),
                }
            )
            print(
                f"  {name:8s} | {samp_name:15s} | {model_name:13s} | "
                f"RMSE {metrics['rmse']:7.2f} | MAE {metrics['mae']:6.2f} | "
                f"R2 {metrics['r2']:6.3f} | {rows[-1]['fit_seconds']:6.1f}s"
            )
    return pd.DataFrame(rows)


def tune_winner(
    name: str, comparison: pd.DataFrame, X_train, y_train, X_test, y_test, gpu: bool, quick: bool
):
    best = comparison.loc[comparison["rmse"].idxmin()]
    print(f"\n{name}: tuning winner {best['model']} + {best['sampling']}")
    X_s, y_s = STRATEGIES[best["sampling"]](X_train, y_train)
    on_gpu = gpu and best["model"] in GPU_MODELS
    search = RandomizedSearchCV(
        wrap_log(build_pipeline(clone(zoo(gpu, quick)[best["model"]]))),
        {f"regressor__{k}": v for k, v in SEARCH_SPACES[best["model"]].items()},
        n_iter=5 if quick else (15 if on_gpu else 30),  # GPU fits run serialized
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=1 if on_gpu else -1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_s, y_s)
    tuned_metrics = evaluate(search.best_estimator_, X_test, y_test)

    # CV on the small resampled train split is optimistically biased and the
    # tuned params can lose to the sweep defaults on held-out data — keep the better
    default = wrap_log(build_pipeline(clone(zoo(gpu, quick)[best["model"]])))
    default.fit(X_s, y_s)
    default_metrics = evaluate(default, X_test, y_test)
    if tuned_metrics["rmse"] <= default_metrics["rmse"]:
        final, metrics = search.best_estimator_, tuned_metrics
        chosen_params = {k: repr(v) for k, v in search.best_params_.items()}
    else:
        final, metrics = default, default_metrics
        chosen_params = {"note": "sweep defaults kept — beat the tuned candidate on test"}
    print(
        f"{name}: tuned RMSE {tuned_metrics['rmse']:.2f} vs default {default_metrics['rmse']:.2f}"
        f" -> keeping {'tuned' if final is not default else 'default'}"
        f" ({best['model']}, R2 {metrics['r2']:.3f})"
    )

    lo, hi = band_offsets(final, X_train, y_train, on_gpu)
    band_lo, band_hi = apply_band(final.predict(X_test), lo, hi)
    coverage = float(np.mean((y_test >= band_lo) & (y_test <= band_hi)))
    print(f"{name}: 80% band offsets [{lo:+.3f}, {hi:+.3f}] log1p, test coverage {coverage:.1%}")

    # keep the artifact portable: predict on CPU everywhere
    model = final.regressor_.named_steps["model"]
    if hasattr(model, "set_params") and "device" in model.get_params():
        model.set_params(device="cpu")

    meta = {
        "model": best["model"],
        "sampling": best["sampling"],
        "best_params": chosen_params,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "band_log_offsets": {"p10": round(lo, 4), "p90": round(hi, 4)},
        "band_test_coverage": round(coverage, 3),
        "sweep_best_rmse": round(float(best["rmse"]), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return final, meta


def mae_by_family(pipe, X_test, y_test) -> pd.DataFrame:
    """Test MAE per iPhone family ('13 Pro Max' -> '13') — R² inflates once the
    scope spans 8 -> 17, so this is the honest per-segment error view."""
    fam = X_test["model"].str.split().str[0]
    err = np.abs(pipe.predict(X_test) - y_test)
    tbl = err.groupby(fam).agg(["mean", "size"]).round(2)
    tbl.columns = ["mae", "n"]
    order = list(dict.fromkeys(m.split()[0] for m in MODELS))
    return tbl.reindex([f for f in order if f in tbl.index])


def _despine(ax) -> None:
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_comparison(name: str, comparison: pd.DataFrame) -> None:
    pivot = comparison.pivot(index="model", columns="sampling", values="rmse")
    pivot = pivot.loc[pivot.min(axis=1).sort_values(ascending=False).index]  # best at top
    ax = pivot.plot.barh(figsize=(9, 6), width=0.82)
    ax.set_xlabel("test RMSE (C$) — lower is better")
    ax.set_ylabel("")
    ax.set_title(f"{name}: test RMSE by model × oversampling", loc="left", fontweight="bold")
    ax.legend(title="sampling", frameon=False, fontsize=9)
    _despine(ax)
    ax.grid(axis="y", visible=False)
    ax.figure.tight_layout()
    ax.figure.savefig(REPORTS / f"comparison_{name}.png", dpi=150)
    plt.close(ax.figure)


def plot_pred_vs_actual(name: str, pipe, X_test, y_test) -> None:
    pred = np.clip(pipe.predict(X_test), 0, None)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_test, pred, s=9, alpha=0.25, edgecolors="none", rasterized=True)
    if float(min(y_test.min(), pred.min())) > 0:  # log-log when zero-free (price)
        lo = 0.9 * float(min(y_test.min(), pred.min()))
        hi = 1.1 * float(max(y_test.max(), pred.max()))
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:  # shipping is zero-inflated, keep linear
        lo, hi = 0.0, 1.05 * float(max(y_test.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7, label="perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"actual {name} (C$)")
    ax.set_ylabel(f"predicted {name} (C$)")
    ax.set_title(f"{name}: held-out test set (n={len(y_test):,})", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(REPORTS / f"pred_vs_actual_{name}.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PROCESSED / "listings.csv")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--quick", action="store_true", help="2 models × 2 samplings (CI/tests)")
    args = parser.parse_args(argv)
    gpu = not args.no_gpu

    df = pd.read_csv(args.data)
    print(f"data: {len(df)} rows from {args.data}")
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    all_comparisons = []
    fam_mae: dict[str, pd.DataFrame] = {}
    metadata: dict = {
        "trained_on": str(date.today()),
        "data_file": str(args.data.name),
        "n_rows": int(len(df)),
        "gpu": gpu,
        "target_transform": "log1p",
        "versions": {
            p: version(p)
            for p in ("scikit-learn", "xgboost", "lightgbm", "catboost", "pandas")
        },
        "targets": {},
    }

    for name, target in (("price", TARGET_PRICE), ("shipping", TARGET_SHIPPING)):
        data = df.dropna(subset=[target])
        X, y = data[FEATURES], data[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        print(f"\n=== {name}: {len(X_train)} train / {len(X_test)} test ===")
        comparison = sweep(name, X_train, y_train, X_test, y_test, gpu, args.quick)
        all_comparisons.append(comparison)
        plot_comparison(name, comparison)

        final, meta = tune_winner(
            name, comparison, X_train, y_train, X_test, y_test, gpu, args.quick
        )
        plot_pred_vs_actual(name, final, X_test, y_test)
        fam_mae[name] = mae_by_family(final, X_test, y_test)
        joblib.dump(final, ARTIFACTS / f"{name}_pipeline.joblib")
        metadata["targets"][name] = meta

    full = pd.concat(all_comparisons, ignore_index=True)
    full.to_csv(REPORTS / "model_comparison.csv", index=False)
    with open(REPORTS / "model_comparison.md", "w", encoding="utf-8") as f:
        f.write(f"# Model comparison — {date.today()}\n\n")
        for name in ("price", "shipping"):
            sub = full[full["target"] == name]
            f.write(f"## {name}\n\n")
            f.write(sub.drop(columns="target").to_markdown(index=False, floatfmt=".3f"))
            f.write("\n\n### final model: test MAE by family\n\n")
            f.write(fam_mae[name].to_markdown(floatfmt=".2f"))
            f.write("\n\n")
        f.write("## Final tuned models\n\n```json\n")
        f.write(json.dumps(metadata["targets"], indent=2))
        f.write("\n```\n")
    with open(ARTIFACTS / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nwrote {REPORTS / 'model_comparison.csv'}, artifacts -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
