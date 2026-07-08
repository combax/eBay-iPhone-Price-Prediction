from ebay_price.config import FEATURES
from ebay_price.sampling import STRATEGIES, _bins


def test_bins_merge_tiny(listings):
    bins = _bins(listings["price_cad"])
    counts = bins.value_counts()
    assert counts.min() >= 8  # SMOTE needs same-bin neighbours
    assert bins.notna().all()


def test_all_strategies_shapes_and_sanity(listings):
    X, y = listings[FEATURES], listings["price_cad"]
    for name, strategy in STRATEGIES.items():
        Xs, ys = strategy(X, y)
        assert len(Xs) == len(ys), name
        assert list(Xs.columns) == FEATURES, name
        assert not ys.isna().any(), name
        assert (ys >= 0).all(), name
        if name == "none":
            assert len(Xs) == len(X)
        else:  # skewed target -> rare bins exist -> strategies add rows
            assert len(Xs) > len(X), name


def test_zero_inflated_target(listings):
    X, y = listings[FEATURES], listings["shipping_cad"]
    for name, strategy in STRATEGIES.items():
        Xs, ys = strategy(X, y)
        assert len(Xs) >= len(X), name
        assert not ys.isna().any(), name
