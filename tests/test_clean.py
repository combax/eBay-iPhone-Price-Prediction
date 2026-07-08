import numpy as np
import pandas as pd

from ebay_price.clean import (
    clean,
    parse_battery,
    parse_carrier,
    parse_model,
    parse_price,
    parse_shipping,
    parse_storage,
)
from ebay_price.config import FEATURES
from ebay_price.scrape import FIELDS


def test_parse_price():
    assert parse_price("C $379.99") == 379.99
    assert parse_price("C $1,234.56") == 1234.56
    assert parse_price("C $249.96 to C $399.99") is None  # multi-variant range
    assert parse_price(np.nan) is None


def test_parse_shipping():
    assert parse_shipping("Free shipping") == 0.0
    assert parse_shipping("+C $40.00 shipping") == 40.0
    assert parse_shipping("+C $16.58 shipping estimate") == 16.58
    assert parse_shipping("Shipping not specified") is None
    assert parse_shipping("") is None


def test_parse_storage():
    assert parse_storage("Apple iPhone 11 64GB Black") == 64
    assert parse_storage("iPhone 11 Pro 256 GB Gold") == 256
    assert parse_storage("iPhone 15 Pro Max 1TB Natural") == 1024
    # multi-variant listings show the cheapest variant's price -> min storage
    assert parse_storage("iPhone 11 64GB 128GB 256GB - All") == 64
    assert parse_storage("Apple iPhone 11 great condition") is None


def test_parse_model_and_carrier():
    assert parse_model("Apple iPhone 11 Pro Max 256GB") == "11 Pro Max"
    assert parse_model("Apple iPhone 11 Pro 64GB") == "11 Pro"
    assert parse_model("Apple iPhone 11 64GB") == "11"
    assert parse_model("Apple iPhone XS Max 64GB Gold") == "XS Max"
    assert parse_model("Apple iPhone XR 128GB") == "XR"
    assert parse_model("iPhone 14 Plus 128GB Midnight") == "14 Plus"
    assert parse_model("Apple iPhone 16e 128GB") == "16e"
    assert parse_model("Apple iPhone Air 256GB Sky Blue") == "Air"
    assert parse_model("iPhone 12 mini 64GB") == "12 Mini"
    assert parse_model("Apple iPhone 12/12 Pro 128GB") is None  # ambiguous multi-model
    assert parse_model("Samsung Galaxy S24 128GB") is None
    assert parse_carrier("iPhone 11 Unlocked 64GB") == "Unlocked"
    assert parse_carrier("iPhone 11 64GB (Bell)") == "Locked"


def test_parse_battery():
    assert parse_battery("iPhone 13 128GB 87% Battery Health") == 87
    assert parse_battery("iPhone 13 Battery Health 92%") == 92
    assert parse_battery("iPhone 13 30% battery") is None  # <50% = junk/typo
    assert parse_battery("iPhone 13 128GB Blue") is None


def _raw_row(**overrides) -> dict:
    row = dict.fromkeys(FIELDS, "")
    row.update(
        listing_id="1",
        title="Apple iPhone 11 64GB Black Unlocked",
        condition="Pre-Owned",
        price="C $200.00",
        shipping="Free shipping",
        seller_name="seller",
        seller_feedback_pct="99.5",
        seller_feedback_count="1200",
        location="",
        page="1",
    )
    row.update(overrides)
    return row


def test_clean_end_to_end():
    raw = pd.DataFrame(
        [
            _raw_row(sold_date="7 Jul 2026"),
            _raw_row(listing_id="2", title="Case for Apple iPhone 11 64GB"),  # accessory
            _raw_row(listing_id="3", title="Lot of 5 iPhone 11 64GB"),  # multi-device lot
            _raw_row(listing_id="4", price="C $100 to C $300"),  # price range
            _raw_row(listing_id="5", title="iPhone 11 64GB cracked screen"),  # damaged
            _raw_row(listing_id="6", title="Samsung Galaxy S10 64GB"),  # wrong phone
            _raw_row(listing_id="7", price="C $99,999.00"),  # out of bounds
            _raw_row(listing_id="1"),  # duplicate id
        ]
    )
    df = clean(raw)
    assert len(df) == 1
    got = df.iloc[0]
    assert got["model"] == "11"
    assert got["storage_gb"] == 64
    assert got["carrier_status"] == "Unlocked"
    assert got["location"] == "Canada"  # blank -> domestic default
    assert got["price_cad"] == 200.0
    assert got["shipping_cad"] == 0.0
    assert got["sealed"] == 0
    assert got["sold_date"] == pd.Timestamp("2026-07-07")
    assert list(df.columns) == [*FEATURES, "price_cad", "shipping_cad", "sold_date"]
