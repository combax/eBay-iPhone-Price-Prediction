import numpy as np
import pandas as pd
import pytest

from ebay_price.config import CONDITIONS, FEATURES, MODELS, STORAGES


@pytest.fixture
def listings() -> pd.DataFrame:
    """Synthetic modeling table matching the clean.py output schema."""
    rng, n = np.random.default_rng(0), 150
    df = pd.DataFrame(
        {
            "condition": rng.choice(CONDITIONS, n),
            "location": rng.choice(["Canada", "United States", "China"], n),
            "model": rng.choice(MODELS, n),
            "carrier_status": rng.choice(["Locked", "Unlocked"], n),
            "storage_gb": rng.choice(STORAGES, n),
            "seller_feedback_pct": rng.uniform(80, 100, n).round(1),
            "seller_feedback_count": rng.integers(0, 5000, n),
            "product_stars": rng.choice([np.nan, 3.5, 4.0, 4.5, 5.0], n),
            "product_ratings_count": rng.integers(0, 300, n),
            "sealed": rng.integers(0, 2, n),
            "battery_health_pct": rng.choice([np.nan, 82.0, 88.0, 95.0, 100.0], n),
            # lognormal: right-skewed like real prices, so rare-bin oversampling engages
            "price_cad": (80 + rng.lognormal(4.5, 0.8, n)).round(2),
            "shipping_cad": rng.choice([0.0, 0.0, 0.0, 15.0, 25.0, 90.0], n),
        }
    )
    assert list(df.columns) == [*FEATURES, "price_cad", "shipping_cad"]
    return df
