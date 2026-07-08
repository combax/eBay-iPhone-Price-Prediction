"""FastAPI service for used-iPhone price & shipping prediction (models 8 -> 17).

Run:
    uvicorn ebay_price.api:app --host 0.0.0.0 --port 8000

Loads the end-to-end pipelines from artifacts/ at startup; requests carry raw
listing fields and go straight into the pipelines — no hand-encoding here.
Trained on SOLD listings, so predictions are market values, not asking prices.
"""

from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import ARTIFACTS, CONDITIONS, MODELS, NUMERIC, STORAGES

Condition = StrEnum("Condition", {c: c for c in CONDITIONS})
ModelName = StrEnum("ModelName", {m: m for m in MODELS})

PIPELINES: dict[str, object] = {}
METADATA: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    artifacts = ARTIFACTS  # module-level so tests can monkeypatch ebay_price.api.ARTIFACTS
    for name in ("price", "shipping"):
        path = artifacts / f"{name}_pipeline.joblib"
        if path.exists():
            PIPELINES[name] = joblib.load(path)
    meta_path = artifacts / "metadata.json"
    if meta_path.exists():
        METADATA.update(json.loads(meta_path.read_text()))
    if not PIPELINES:
        raise RuntimeError(
            f"no pipelines in {artifacts} — run `python -m ebay_price.train` first"
        )
    yield
    PIPELINES.clear()
    METADATA.clear()


app = FastAPI(
    title="Used iPhone price API",
    description="Predicts sold price and shipping cost (C$) for used iPhones (8 -> 17) on eBay.ca",
    version=__version__,
    lifespan=lifespan,
)


class Listing(BaseModel):
    condition: Condition
    model: ModelName
    storage_gb: int
    carrier_status: Literal["Locked", "Unlocked"]
    location: str = Field("Canada", description="Seller country; unseen values are fine")
    # omitted seller stats become NaN and the pipelines impute training medians,
    # so the default prediction prices a typical sale, not a 0-feedback seller
    seller_feedback_pct: float | None = Field(None, ge=0, le=100)
    seller_feedback_count: int | None = Field(None, ge=0)
    product_stars: float | None = Field(None, ge=0, le=5)
    product_ratings_count: int = Field(0, ge=0)
    sealed: bool = False
    battery_health_pct: float | None = Field(None, ge=50, le=100)

    @field_validator("storage_gb")
    @classmethod
    def _known_storage(cls, v: int) -> int:
        if v not in STORAGES:
            raise ValueError(f"storage_gb must be one of {STORAGES}")
        return v


class Prediction(BaseModel):
    predicted_price_cad: float | None = None
    price_range_cad: tuple[float, float] | None = None
    predicted_shipping_cad: float | None = None
    shipping_range_cad: tuple[float, float] | None = None
    trained_on: str | None = None


@app.post("/predict")
def predict(listing: Listing) -> Prediction:
    row = pd.DataFrame([listing.model_dump()])
    row[NUMERIC] = row[NUMERIC].apply(pd.to_numeric)  # None -> NaN, pipeline imputes
    out: dict = {}
    for name, pipe in PIPELINES.items():
        pred = round(max(float(pipe.predict(row)[0]), 0.0), 2)
        out[f"predicted_{name}_cad"] = pred
        # 80% band from train-time out-of-fold residual quantiles (log1p space)
        if offsets := METADATA.get("targets", {}).get(name, {}).get("band_log_offsets"):
            log_pred = math.log1p(pred)
            out[f"{name}_range_cad"] = (
                round(max(math.expm1(log_pred + offsets["p10"]), 0.0), 2),
                round(math.expm1(log_pred + offsets["p90"]), 2),
            )
    return Prediction(**out, trained_on=METADATA.get("trained_on"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipelines": sorted(PIPELINES)}


@app.get("/")
def root() -> dict:
    return {
        "service": "Used iPhone price API",
        "version": __version__,
        "docs": "/docs",
        "models": {
            name: meta.get("model")
            for name, meta in METADATA.get("targets", {}).items()
        },
    }
