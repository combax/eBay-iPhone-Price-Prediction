"""Project paths and constants."""

import os
from pathlib import Path

# EBAY_PRICE_HOME lets Docker/installed copies point at a data dir outside site-packages
ROOT = Path(os.environ.get("EBAY_PRICE_HOME", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

RANDOM_STATE = 42

# modeling table schema (produced by clean.py, consumed by train.py / api.py)
CATEGORICAL = ["condition", "location", "model", "carrier_status"]
NUMERIC = [
    "storage_gb",
    "seller_feedback_pct",
    "seller_feedback_count",
    "product_stars",
    "product_ratings_count",
    "sealed",
    "battery_health_pct",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET_PRICE = "price_cad"
TARGET_SHIPPING = "shipping_cad"

CONDITIONS = [
    "Parts Only",
    "Pre-Owned",
    "Good - Refurbished",
    "Very Good - Refurbished",
    "Excellent - Refurbished",
    "Certified - Refurbished",
    "Refurbished",
    "Open Box",
    "New (Other)",
    "Brand New",
]
# every iPhone family sold used in 2026: 8 through 17, the X generation, and Air
MODELS = [
    "8", "8 Plus",
    "X", "XR", "XS", "XS Max",
    "11", "11 Pro", "11 Pro Max",
    "12 Mini", "12", "12 Pro", "12 Pro Max",
    "13 Mini", "13", "13 Pro", "13 Pro Max",
    "14", "14 Plus", "14 Pro", "14 Pro Max",
    "15", "15 Plus", "15 Pro", "15 Pro Max",
    "16e", "16", "16 Plus", "16 Pro", "16 Pro Max",
    "17e", "17", "17 Pro", "17 Pro Max",
    "Air",
]
STORAGES = [64, 128, 256, 512, 1024, 2048]
