"""Polite scraper for iPhone listings on eBay.ca search results.

Usage:
    python -m ebay_price.scrape [--max-pages 25] [--active] [--out data/raw/my.csv]

Defaults to SOLD listings (last ~90 days) across all iPhone families 8 -> 17:
sold prices are transaction prices, which is what the models train on —
asking prices for old phones are dominated by fantasy listings that never
sell. --active scrapes live Buy-It-Now asking prices instead.

Writes one CSV row per unique listing, flushed after every page, and moves to
the next query when a page yields no new listings (eBay repeats results past
the last real page). If eBay refuses the connection, rows scraped so far are
already on disk.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup
from curl_cffi import requests

from .config import CONDITIONS as CANONICAL_CONDITIONS
from .config import DATA_RAW

SEARCH_URL = "https://www.ebay.ca/sch/i.html"
# one query per iPhone family — eBay's Model filter leaks other models anyway,
# so clean.py's title parsing is the real gate and queries just steer coverage
QUERIES = [
    f"iPhone {fam}"
    for fam in ("8", "X", "XR", "XS", "11", "12", "13", "14", "15", "16", "17", "Air")
]
BASE_PARAMS = {
    "_sacat": "9355",  # Cell Phones & Smartphones
    "_ipg": "240",  # max listings per page
}
# card labels to recognize: the canonical set plus eBay's raw Parts Only wording
CONDITIONS = {*CANONICAL_CONDITIONS, "For parts or not working"}
FEEDBACK_RE = re.compile(r"^([\d.]+)% positive \(([\d.,]+[KM]?)\)$")
STARS_RE = re.compile(r"([\d.]+) out of 5")
SOLD_RE = re.compile(r"^sold\s+(.+)$", re.I)
FIELDS = [
    "listing_id", "title", "condition", "price", "shipping", "seller_name",
    "seller_feedback_pct", "seller_feedback_count", "product_stars",
    "product_ratings_count", "location", "sold_date", "page",
]


def _count(text: str) -> int:
    """'20.1K' -> 20100, '1,584' -> 1584."""
    text = text.replace(",", "")
    mult = {"K": 1_000, "M": 1_000_000}.get(text[-1:], 1)
    return int(float(text.rstrip("KM")) * mult)


def new_session() -> requests.Session:
    session = requests.Session(impersonate="chrome")
    session.headers.update({"Accept-Language": "en-CA,en;q=0.9"})
    # cookie warm-up on the homepage; the search endpoint 403s cold sessions
    session.get("https://www.ebay.ca/", timeout=30).raise_for_status()
    return session


def parse_card(li) -> dict | None:
    """One search-result card -> raw row dict, or None for promo tiles."""
    listing_id = li.get("data-listingid")
    img = li.select_one("img.s-card__image")
    title = img.get("alt", "").strip() if img else ""
    price_el = li.select_one(".s-card__price")
    if not (listing_id and title and price_el):
        return None
    row = dict.fromkeys(FIELDS, "")
    row.update(listing_id=listing_id, title=title, price=price_el.get_text(strip=True))

    caption = li.select_one(".s-card__caption")
    if caption and (m := SOLD_RE.match(caption.get_text(" ", strip=True))):
        row["sold_date"] = m.group(1)

    for span in li.select("span.su-styled-text"):
        text = span.get_text(" ", strip=True)
        if not row["condition"] and text in CONDITIONS:
            row["condition"] = text
        elif not row["shipping"] and "shipping" in text.lower():
            row["shipping"] = text
        elif not row["location"] and text.startswith("from "):
            row["location"] = text[5:]

    for attr_row in li.select("div.s-card__attribute-row"):
        spans = attr_row.find_all("span", recursive=False)
        if len(spans) == 2 and (m := FEEDBACK_RE.match(spans[1].get_text(" ", strip=True))):
            row["seller_name"] = spans[0].get_text(" ", strip=True)
            row["seller_feedback_pct"] = m.group(1)
            row["seller_feedback_count"] = _count(m.group(2))
            break

    stars_el = li.select_one("div.x-star-rating")
    if stars_el and (m := STARS_RE.search(stars_el.get_text(" ", strip=True))):
        row["product_stars"] = m.group(1)
        # the count lives in a class-less span; span.string is often None, so match text
        for span in li.find_all("span"):
            if m2 := re.match(r"^([\d,]+) product ratings?$", span.get_text(" ", strip=True)):
                row["product_ratings_count"] = m2.group(1).replace(",", "")
                break
    return row


def scrape(
    max_pages: int = 25,
    out_path: Path | None = None,
    sold: bool = True,
    queries: list[str] | None = None,
) -> Path:
    queries = queries or QUERIES
    mode = "sold" if sold else "active"
    out_path = out_path or DATA_RAW / f"ebay_iphone_{mode}_{date.today():%Y-%m-%d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # sold results include ended auctions, whose final bid is a real price;
    # live auctions carry no final price, so the active mode stays Buy-It-Now only
    mode_params = {"LH_Sold": "1", "LH_Complete": "1"} if sold else {"LH_BIN": "1"}
    session = new_session()
    seen: set[str] = set()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for query in queries:
            for page in range(1, max_pages + 1):
                params = BASE_PARAMS | mode_params | {"_nkw": query, "_pgn": str(page)}
                for attempt in range(1, 4):
                    resp = session.get(SEARCH_URL, params=params, timeout=30)
                    if resp.status_code == 200:
                        break
                    print(
                        f"{query} page {page}: HTTP {resp.status_code}, retry {attempt}/3",
                        file=sys.stderr,
                    )
                    time.sleep(10 * attempt)
                    session = new_session()
                else:
                    raise RuntimeError(
                        f"eBay refused {query!r} page {page} after 3 attempts — "
                        f"the {len(seen)} rows scraped so far are safe in {out_path}"
                    )

                cards = BeautifulSoup(resp.text, "lxml").select("li.s-card")
                new = [r for c in cards if (r := parse_card(c)) and r["listing_id"] not in seen]
                for r in new:
                    r["page"] = page
                seen.update(r["listing_id"] for r in new)
                writer.writerows(new)
                f.flush()
                print(
                    f"{query} page {page}: {len(cards)} cards, {len(new)} new "
                    f"(total {len(seen)})"
                )
                if not new:
                    break
                time.sleep(random.uniform(1.5, 3.0))
            time.sleep(random.uniform(2.0, 4.0))

    print(f"wrote {len(seen)} listings -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=25, help="per query")
    parser.add_argument(
        "--active", action="store_true", help="live asking prices instead of sold prices"
    )
    parser.add_argument("--queries", nargs="+", default=None, metavar="QUERY")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    scrape(
        max_pages=args.max_pages, out_path=args.out, sold=not args.active, queries=args.queries
    )


if __name__ == "__main__":
    main()
