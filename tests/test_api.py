import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge

import ebay_price.api as api
from ebay_price.config import FEATURES
from ebay_price.models import build_pipeline


@pytest.fixture
def client(listings, tmp_path, monkeypatch):
    """API backed by small pipelines trained on the synthetic table."""
    for name, target in (("price", "price_cad"), ("shipping", "shipping_cad")):
        pipe = build_pipeline(Ridge())
        pipe.fit(listings[FEATURES], listings[target])
        joblib.dump(pipe, tmp_path / f"{name}_pipeline.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "trained_on": "2026-01-01",
                "targets": {
                    "price": {
                        "model": "ridge",
                        "band_log_offsets": {"p10": -0.3, "p90": 0.35},
                    }
                },
            }
        )
    )
    monkeypatch.setattr(api, "ARTIFACTS", tmp_path)
    with TestClient(api.app) as c:
        yield c


GOOD = {
    "condition": "Pre-Owned",
    "model": "11 Pro",
    "storage_gb": 128,
    "carrier_status": "Unlocked",
    "location": "Canada",
    "seller_feedback_pct": 99.1,
    "seller_feedback_count": 1500,
    "product_stars": 4.5,
    "product_ratings_count": 20,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["pipelines"] == ["price", "shipping"]


def test_predict(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_price_cad"] >= 0
    assert body["predicted_shipping_cad"] >= 0
    assert body["trained_on"] == "2026-01-01"
    lo, hi = body["price_range_cad"]  # from the fixture's band offsets
    assert lo <= body["predicted_price_cad"] <= hi
    assert body["shipping_range_cad"] is None  # shipping metadata has no offsets


def test_predict_defaults_and_missing_stars(client):
    minimal = {k: GOOD[k] for k in ("condition", "model", "storage_gb", "carrier_status")}
    r = client.post("/predict", json=minimal)
    assert r.status_code == 200  # optional fields default, stars imputed by pipeline


def test_predict_new_scope_models(client):
    r = client.post("/predict", json={**GOOD, "model": "17 Pro Max", "storage_gb": 2048})
    assert r.status_code == 200
    r = client.post("/predict", json={**GOOD, "model": "Air", "battery_health_pct": 92})
    assert r.status_code == 200


def test_predict_unseen_location_ok(client):
    r = client.post("/predict", json={**GOOD, "location": "Atlantis"})
    assert r.status_code == 200  # OneHot(handle_unknown="ignore")


def test_predict_bad_condition(client):
    r = client.post("/predict", json={**GOOD, "condition": "Slightly Chewed"})
    assert r.status_code == 422
    r = client.post("/predict", json={**GOOD, "storage_gb": 33})
    assert r.status_code == 422
