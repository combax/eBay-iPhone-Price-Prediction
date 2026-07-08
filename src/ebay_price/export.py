"""Export trained pipelines for the Go serving path (deploy/go).

Usage:
    python -m ebay_price.export

Reads artifacts/<target>_pipeline.joblib + artifacts/metadata.json and writes
artifacts/serving/:
    <target>.onnx   the regressor only — float feature vector in, log1p-scale
                    prediction out (the Go server applies expm1 and the bands)
    serving.json    fitted preprocessing constants (imputer medians, scaler
                    mean/std, one-hot categories), band offsets, serving
                    defaults, and parity vectors with expected predictions
                    from the joblib pipelines

Everything the Go server needs is generated here from the FITTED sklearn
objects — nothing is hand-maintained — and the server refuses to start if its
predictions drift from the recorded parity vectors.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .config import (
    ARTIFACTS,
    CATEGORICAL,
    CONDITIONS,
    DATA_PROCESSED,
    FEATURES,
    MODELS,
    NUMERIC,
    STORAGES,
)

N_PARITY = 25


def prep_params(prep) -> dict:
    """Fitted preprocessing constants; the Go feature vector is
    [(numeric - mean) / std ..., one-hot(categorical) ...] in this order."""
    num = prep.named_transformers_["num"]
    ohe = prep.named_transformers_["cat"]
    return {
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "impute_median": num.named_steps["impute"].statistics_.tolist(),
        "scale_mean": num.named_steps["scale"].mean_.tolist(),
        "scale_std": num.named_steps["scale"].scale_.tolist(),
        "categories": {
            col: [str(v) for v in cats]
            for col, cats in zip(CATEGORICAL, ohe.categories_, strict=True)
        },
    }


def to_onnx(model, n_features: int) -> bytes:
    kind = type(model).__name__
    if kind == "LGBMRegressor":
        from onnxmltools import convert_lightgbm as convert
        from onnxmltools.convert.common.data_types import FloatTensorType
    elif kind == "XGBRegressor":
        from onnxmltools import convert_xgboost as convert
        from onnxmltools.convert.common.data_types import FloatTensorType
    else:  # sklearn natives (ridge, hist_gb, random_forest); torch_mlp has no exporter
        from skl2onnx import convert_sklearn as convert
        from skl2onnx.common.data_types import FloatTensorType

    onx = convert(model, initial_types=[("input", FloatTensorType([None, n_features]))])
    return onx.SerializeToString()


def export_target(name: str, sample: pd.DataFrame, meta_t: dict, out_dir) -> dict:
    ttr = joblib.load(ARTIFACTS / f"{name}_pipeline.joblib")
    pipe = ttr.regressor_  # Pipeline(prep, model); predicts on the log1p scale
    params = prep_params(pipe.named_steps["prep"])
    n_features = len(NUMERIC) + sum(len(c) for c in params["categories"].values())
    (out_dir / f"{name}.onnx").write_bytes(to_onnx(pipe.named_steps["model"], n_features))

    expected = np.clip(ttr.predict(sample), 0.0, None)
    parity = [
        {
            "input": {
                k: (None if pd.isna(v) else (float(v) if k in NUMERIC else str(v)))
                for k, v in row.items()
            },
            "expected": round(float(pred), 4),
        }
        for (_, row), pred in zip(sample.iterrows(), expected, strict=True)
    ]
    print(f"{name}: {n_features}-feature ONNX + {len(parity)} parity vectors")
    return {
        "onnx": f"{name}.onnx",
        "prep": params,
        "band_log_offsets": meta_t["band_log_offsets"],
        "parity": parity,
    }


def main() -> None:
    meta = json.loads((ARTIFACTS / "metadata.json").read_text())
    df = pd.read_csv(DATA_PROCESSED / "listings.csv")
    sample = df[FEATURES].sample(N_PARITY, random_state=0)
    out_dir = ARTIFACTS / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)

    serving = {
        "trained_on": meta["trained_on"],
        # request-validation vocabulary — keeps the Go contract identical to api.py
        "vocab": {
            "condition": CONDITIONS,
            "model": MODELS,
            "carrier_status": ["Locked", "Unlocked"],
            "storage_gb": STORAGES,
        },
        "targets": {
            name: export_target(name, sample, meta["targets"][name], out_dir)
            for name in meta["targets"]
        },
    }
    (out_dir / "serving.json").write_text(json.dumps(serving, indent=1))
    print(f"wrote {out_dir / 'serving.json'}")


if __name__ == "__main__":
    main()
