"""Parity guards for the Go serving path: the exported prep constants and the
ONNX graph must reproduce the sklearn pipeline. The Go server mirrors
manual_vector() below and self-checks against the recorded parity vectors."""

import numpy as np
import pytest
from lightgbm import LGBMRegressor

from ebay_price.config import FEATURES
from ebay_price.export import prep_params, to_onnx
from ebay_price.models import build_pipeline
from ebay_price.train import wrap_log

onnxruntime = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxmltools")


def manual_vector(params: dict, row: dict) -> np.ndarray:
    """Feature vector exactly as deploy/go/main.go builds it."""
    out = []
    for i, col in enumerate(params["numeric"]):
        v = row.get(col)
        v = params["impute_median"][i] if v is None or (isinstance(v, float) and np.isnan(v)) else v
        out.append((v - params["scale_mean"][i]) / params["scale_std"][i])
    for col in params["categorical"]:
        cats = params["categories"][col]
        onehot = [0.0] * len(cats)
        if str(row.get(col)) in cats:  # unknown category -> all zeros
            onehot[cats.index(str(row.get(col)))] = 1.0
        out.extend(onehot)
    return np.asarray(out, dtype=np.float32)


@pytest.fixture
def fitted(listings):
    pipe = wrap_log(build_pipeline(LGBMRegressor(n_estimators=30, verbose=-1)))
    pipe.fit(listings[FEATURES], listings["price_cad"])
    return pipe, listings


def test_prep_params_reproduce_transform(fitted):
    pipe, listings = fitted
    prep = pipe.regressor_.named_steps["prep"]
    params = prep_params(prep)
    X = listings[FEATURES].head(30)
    want = prep.transform(X).astype(np.float32)
    got = np.vstack([manual_vector(params, row.to_dict()) for _, row in X.iterrows()])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_onnx_matches_model(fitted):
    pipe, listings = fitted
    prep = pipe.regressor_.named_steps["prep"]
    model = pipe.regressor_.named_steps["model"]
    X_t = prep.transform(listings[FEATURES]).astype(np.float32)
    sess = onnxruntime.InferenceSession(to_onnx(model, X_t.shape[1]))
    got = sess.run(None, {sess.get_inputs()[0].name: X_t})[0].ravel()
    np.testing.assert_allclose(got, model.predict(X_t), rtol=1e-4, atol=1e-4)
