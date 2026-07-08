import json

import ebay_price.train as train


def test_train_quick_end_to_end(listings, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(train, "REPORTS", tmp_path / "reports")
    data = tmp_path / "listings.csv"
    listings.to_csv(data, index=False)

    train.main(["--data", str(data), "--quick", "--no-gpu"])

    meta = json.loads((tmp_path / "artifacts" / "metadata.json").read_text())
    assert set(meta["targets"]) == {"price", "shipping"}
    for name in ("price", "shipping"):
        assert (tmp_path / "artifacts" / f"{name}_pipeline.joblib").exists()
        assert meta["targets"][name]["metrics"]["rmse"] > 0
    assert (tmp_path / "reports" / "model_comparison.csv").exists()
    assert (tmp_path / "reports" / "model_comparison.md").exists()
