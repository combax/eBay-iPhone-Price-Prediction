"""Clean raw scraped listings into the modeling table.

Usage:
    python -m ebay_price.clean [--raw data/raw/a.csv data/raw/b.csv]
                               [--out data/processed/listings.csv]

Defaults to ALL data/raw/ebay_iphone_sold_*.csv files concatenated — repeated
scrape runs accumulate history and listing_id dedup removes the overlap.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from .config import CONDITIONS, DATA_PROCESSED, DATA_RAW, FEATURES, MODELS, STORAGES

# accessories and parts that match the search but aren't phones
ACCESSORY_RE = re.compile(
    r"\b(?:case|cover|screen protector|protector|lens|charger|cable|box only|"
    r"empty box|back glass|housing|bezel|digitizer|battery)\b",
    re.I,
)
# defect listings the price model shouldn't learn from (Parts Only condition stays)
DAMAGE_RE = re.compile(r"crack|broken|no face ?id", re.I)
# multi-device lots aren't single-phone prices
LOT_RE = re.compile(r"\b(?:lot|bulk|wholesale|\d+\s*(?:pcs|units|pieces))\b", re.I)
# sold prices are real transactions, so bounds are only junk guards: below C$20 is
# parts/errors, above C$4000 outruns a maxed-out new 17 Pro Max
PRICE_BOUNDS = (20.0, 4000.0)
# a sale is only a current-market price signal for so long; older is drift noise
SOLD_MAX_AGE_DAYS = 365
# 87% of sold listings ship cross-border from the US, so C$35-270 quotes are real;
# above this is freight-quote junk (p99.9 ~ C$630) — null it, the price row stays
SHIPPING_CAP = 500.0

# family after "iphone" (or a slash continuation: "iPhone 12/12 Pro"), optional variant
MODEL_RE = re.compile(
    r"(?:iphone\s*|(?<=/)\s*)(air|xs|xr|x|8|1[1-7]e?)(?![0-9a-z])[\s-]*"
    r"(pro\s*max|pro|plus|max|mini)?",
    re.I,
)
SUFFIXES = {"promax": "Pro Max", "pro": "Pro", "plus": "Plus", "max": "Max", "mini": "Mini"}
BATTERY_RE = re.compile(
    r"(\d{2,3})\s*%\s*(?:battery|batt\b|bh\b)|batt(?:ery)?\s*(?:health)?\s*[:\-]?\s*(\d{2,3})\s*%",
    re.I,
)


def parse_price(text: object) -> float | None:
    """'C $379.99' -> 379.99. Ranges ('C $250 to C $400') and junk -> None."""
    if not isinstance(text, str) or " to " in text.lower():
        return None
    m = re.search(r"([\d,]+\.?\d*)", text)
    return float(m.group(1).replace(",", "")) if m else None


def parse_shipping(text: object) -> float | None:
    """'Free shipping' -> 0.0, '+C $40.00 shipping' -> 40.0, unknown -> None."""
    if not isinstance(text, str) or "shipping" not in text.lower():
        return None
    if "free" in text.lower():
        return 0.0
    m = re.search(r"([\d,]+\.?\d*)", text)
    return float(m.group(1).replace(",", "")) if m else None


def parse_storage(title: str) -> float | None:
    """Smallest plausible capacity in GB, or None when the title names none.

    ponytail: multi-variant listings ("64GB/128GB/256GB") show the cheapest
    variant's price, so min storage is the config that matches the price.
    """
    gbs = {int(m) for m in re.findall(r"(\d+)\s*GB", title, flags=re.I)}
    gbs |= {int(m) * 1024 for m in re.findall(r"(\d+)\s*TB", title, flags=re.I)}
    plausible = gbs & set(STORAGES)
    return float(min(plausible)) if plausible else None


def parse_model(title: str) -> str | None:
    """Canonical model ('13 Pro Max', 'XS Max', '16e', 'Air'), or None when no
    family is named or several distinct ones are (multi-model listings)."""
    found = set()
    for base, suffix in MODEL_RE.findall(title):
        base = base.lower()
        name = "Air" if base == "air" else base.upper() if base.startswith("x") else base
        if suffix:
            name += " " + SUFFIXES[re.sub(r"\s+", "", suffix.lower())]
        found.add(name)
    return found.pop() if len(found) == 1 else None


def parse_carrier(title: str) -> str:
    # sellers advertise "unlocked" when true; listings that don't say it are
    # treated as carrier-locked
    return "Unlocked" if re.search(r"\bunlocked\b", title, re.I) else "Locked"


def parse_battery(title: str) -> float | None:
    """Battery health % from the title ('87% battery health'), else None."""
    if m := BATTERY_RE.search(title):
        pct = float(next(g for g in m.groups() if g))
        if 50 <= pct <= 100:
            return pct
    return None


def clean(raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    funnel: list[tuple[str, int]] = [("raw", len(raw))]

    def step(label: str, frame: pd.DataFrame) -> pd.DataFrame:
        funnel.append((label, len(frame)))
        return frame

    df = step("unique listing_id", raw.drop_duplicates("listing_id").copy())
    df = step("mentions iphone", df[df["title"].str.contains("iphone", case=False, na=False)])
    df = step("not accessory", df[~df["title"].str.contains(ACCESSORY_RE)])
    df = step("not damaged", df[~df["title"].str.contains(DAMAGE_RE)])
    df = step("not multi-device lot", df[~df["title"].str.contains(LOT_RE)])

    df["model"] = df["title"].map(parse_model)
    df["storage_gb"] = df["title"].map(parse_storage)
    df["carrier_status"] = df["title"].map(parse_carrier)
    df["battery_health_pct"] = df["title"].map(parse_battery)
    df["sealed"] = df["title"].str.contains(r"\bsealed\b", case=False).astype(int)
    df["condition"] = df["condition"].replace({"For parts or not working": "Parts Only"})
    df["price_cad"] = df["price"].map(parse_price)
    df["shipping_cad"] = df["shipping"].map(parse_shipping)
    df.loc[df["shipping_cad"] > SHIPPING_CAP, "shipping_cad"] = None
    if "sold_date" not in df:  # pre-2026-07 raw files predate the column
        df["sold_date"] = None
    df["sold_date"] = pd.to_datetime(df["sold_date"], errors="coerce", dayfirst=True)
    # pandas >= 2.1 astype(str) keeps NaN as NaN, so fill before stringifying
    df["location"] = df["location"].fillna("Canada").astype(str).str.strip().replace("", "Canada")
    for col in ("seller_feedback_pct", "product_stars"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("seller_feedback_count", "product_ratings_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = step("known single model", df[df["model"].isin(MODELS)])
    df = step("known condition", df[df["condition"].isin(CONDITIONS)])
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=SOLD_MAX_AGE_DAYS)
    df = step(  # active listings (no sold_date) pass through
        "sold within 12 months",
        df[df["sold_date"].isna() | (df["sold_date"] >= cutoff)],
    )
    df = step("has storage in title", df.dropna(subset=["storage_gb"]))
    df = step("has single price", df.dropna(subset=["price_cad"]))
    df = step(
        "price within bounds",
        df[df["price_cad"].between(*PRICE_BOUNDS)],
    )
    df = step("has seller feedback", df.dropna(subset=["seller_feedback_pct"]))
    if verbose:
        for (label, n), (_, prev) in zip(funnel[1:], funnel, strict=False):
            print(f"  {label:22s} {n:5d}  (-{prev - n})")
    # shipping_cad and product_stars may stay NaN: train.py drops NaN shipping rows
    # for the shipping target only, and pipelines impute the rest
    return df[[*FEATURES, "price_cad", "shipping_cad", "sold_date"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=DATA_PROCESSED / "listings.csv")
    args = parser.parse_args()

    paths = args.raw or sorted(DATA_RAW.glob("ebay_iphone_sold_*.csv"))
    if not paths:
        sys.exit(
            f"no {DATA_RAW / 'ebay_iphone_sold_*.csv'} — "
            "run `python -m ebay_price.scrape` first, or pass --raw"
        )
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = clean(raw, verbose=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    names = ", ".join(p.name for p in paths)
    print(f"{names}: {len(raw)} raw -> {len(df)} clean rows -> {args.out}")
    print(df.describe(include="all").T[["count", "unique", "top", "mean", "min", "max"]])


if __name__ == "__main__":
    main()
