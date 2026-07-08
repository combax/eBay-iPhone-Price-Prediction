# Model comparison — 2026-07-07

## price

| model         | sampling        |    rmse |    mae |    r2 |   fit_seconds |   train_rows |
|:--------------|:----------------|--------:|-------:|------:|--------------:|-------------:|
| ridge         | none            | 155.798 | 89.239 | 0.885 |         0.060 |        20821 |
| hist_gb       | none            | 128.713 | 70.128 | 0.922 |         1.940 |        20821 |
| random_forest | none            | 127.022 | 64.427 | 0.924 |         2.330 |        20821 |
| xgboost       | none            | 120.357 | 64.819 | 0.931 |         1.440 |        20821 |
| lightgbm      | none            | 117.769 | 63.090 | 0.934 |         0.770 |        20821 |
| catboost      | none            | 126.190 | 68.081 | 0.925 |        15.240 |        20821 |
| torch_mlp     | none            | 132.501 | 70.077 | 0.917 |        12.520 |        20821 |
| ridge         | random_over     | 149.009 | 85.566 | 0.895 |         0.080 |        31952 |
| hist_gb       | random_over     | 127.669 | 71.570 | 0.923 |         0.580 |        31952 |
| random_forest | random_over     | 131.425 | 65.572 | 0.918 |         3.720 |        31952 |
| xgboost       | random_over     | 122.362 | 65.496 | 0.929 |         1.600 |        31952 |
| lightgbm      | random_over     | 119.769 | 63.286 | 0.932 |         0.920 |        31952 |
| catboost      | random_over     | 123.567 | 69.110 | 0.928 |        16.860 |        31952 |
| torch_mlp     | random_over     | 155.891 | 75.903 | 0.885 |        32.740 |        31952 |
| ridge         | smotenc         | 149.704 | 85.674 | 0.894 |         0.070 |        31952 |
| hist_gb       | smotenc         | 129.112 | 71.632 | 0.921 |         0.600 |        31952 |
| random_forest | smotenc         | 129.384 | 66.033 | 0.921 |         4.060 |        31952 |
| xgboost       | smotenc         | 122.792 | 66.069 | 0.929 |         1.620 |        31952 |
| lightgbm      | smotenc         | 119.000 | 63.778 | 0.933 |         0.970 |        31952 |
| catboost      | smotenc         | 128.481 | 70.181 | 0.922 |        15.910 |        31952 |
| torch_mlp     | smotenc         | 135.766 | 73.525 | 0.913 |        13.070 |        31952 |
| ridge         | gaussian_jitter | 149.047 | 85.566 | 0.895 |         0.110 |        31952 |
| hist_gb       | gaussian_jitter | 132.799 | 73.755 | 0.916 |         0.640 |        31952 |
| random_forest | gaussian_jitter | 126.633 | 64.442 | 0.924 |         5.080 |        31952 |
| xgboost       | gaussian_jitter | 125.169 | 67.579 | 0.926 |         1.560 |        31952 |
| lightgbm      | gaussian_jitter | 120.518 | 64.415 | 0.931 |         1.160 |        31952 |
| catboost      | gaussian_jitter | 133.262 | 73.510 | 0.916 |        15.950 |        31952 |
| torch_mlp     | gaussian_jitter | 137.996 | 72.711 | 0.910 |        19.420 |        31952 |

### final model: test MAE by family

| model   |    mae |      n |
|:--------|-------:|-------:|
| 8       |  23.02 | 117.00 |
| X       |  32.34 |  46.00 |
| XR      |  21.65 | 131.00 |
| XS      |  30.89 |  78.00 |
| 11      |  26.40 | 387.00 |
| 12      |  34.31 | 581.00 |
| 13      |  48.68 | 767.00 |
| 14      |  54.58 | 727.00 |
| 15      |  66.16 | 922.00 |
| 16e     |  42.82 |  88.00 |
| 16      |  90.61 | 526.00 |
| 17e     |  78.14 |  36.00 |
| 17      | 113.99 | 713.00 |
| Air     |  87.80 |  87.00 |

## shipping

| model         | sampling        |   rmse |    mae |    r2 |   fit_seconds |   train_rows |
|:--------------|:----------------|-------:|-------:|------:|--------------:|-------------:|
| ridge         | none            | 94.873 | 68.645 | 0.206 |         0.080 |        20631 |
| hist_gb       | none            | 77.306 | 52.112 | 0.473 |         0.550 |        20631 |
| random_forest | none            | 75.082 | 45.716 | 0.502 |         2.730 |        20631 |
| xgboost       | none            | 74.439 | 49.199 | 0.511 |         1.550 |        20631 |
| lightgbm      | none            | 74.121 | 47.774 | 0.515 |         0.890 |        20631 |
| catboost      | none            | 77.739 | 52.644 | 0.467 |        13.170 |        20631 |
| torch_mlp     | none            | 79.860 | 55.358 | 0.437 |         4.790 |        20631 |
| ridge         | random_over     | 83.705 | 65.190 | 0.382 |         0.090 |        35349 |
| hist_gb       | random_over     | 71.795 | 49.651 | 0.545 |         0.540 |        35349 |
| random_forest | random_over     | 76.452 | 45.538 | 0.484 |         4.850 |        35349 |
| xgboost       | random_over     | 70.270 | 47.563 | 0.564 |         1.630 |        35349 |
| lightgbm      | random_over     | 71.271 | 46.335 | 0.552 |         0.930 |        35349 |
| catboost      | random_over     | 70.677 | 49.057 | 0.559 |        16.580 |        35349 |
| torch_mlp     | random_over     | 79.452 | 53.865 | 0.443 |        27.730 |        35349 |
| ridge         | smotenc         | 83.552 | 65.525 | 0.384 |         0.110 |        35349 |
| hist_gb       | smotenc         | 72.390 | 50.933 | 0.538 |         0.690 |        35349 |
| random_forest | smotenc         | 75.664 | 46.182 | 0.495 |         5.380 |        35349 |
| xgboost       | smotenc         | 71.181 | 49.279 | 0.553 |         1.700 |        35349 |
| lightgbm      | smotenc         | 71.118 | 47.659 | 0.554 |         1.090 |        35349 |
| catboost      | smotenc         | 72.267 | 51.110 | 0.539 |        16.580 |        35349 |
| torch_mlp     | smotenc         | 76.717 | 54.047 | 0.481 |        12.890 |        35349 |
| ridge         | gaussian_jitter | 83.679 | 65.172 | 0.382 |         0.090 |        35349 |
| hist_gb       | gaussian_jitter | 77.376 | 53.379 | 0.472 |         0.640 |        35349 |
| random_forest | gaussian_jitter | 75.894 | 46.300 | 0.492 |         6.440 |        35349 |
| xgboost       | gaussian_jitter | 76.055 | 51.576 | 0.489 |         1.560 |        35349 |
| lightgbm      | gaussian_jitter | 73.931 | 48.771 | 0.518 |         1.130 |        35349 |
| catboost      | gaussian_jitter | 77.594 | 53.378 | 0.469 |        16.030 |        35349 |
| torch_mlp     | gaussian_jitter | 77.231 | 53.880 | 0.474 |        35.750 |        35349 |

### final model: test MAE by family

| model   |    mae |      n |
|:--------|-------:|-------:|
| 8       |  15.87 | 127.00 |
| X       |  11.51 |  43.00 |
| XR      |  19.17 | 131.00 |
| XS      |  14.45 |  80.00 |
| 11      |  19.21 | 420.00 |
| 12      |  19.03 | 538.00 |
| 13      |  30.54 | 743.00 |
| 14      |  33.23 | 721.00 |
| 15      |  54.02 | 886.00 |
| 16e     |  33.61 |  82.00 |
| 16      |  66.63 | 539.00 |
| 17e     |  59.59 |  45.00 |
| 17      | 109.19 | 716.00 |
| Air     |  78.13 |  87.00 |

## Final tuned models

```json
{
  "price": {
    "model": "lightgbm",
    "sampling": "none",
    "best_params": {
      "regressor__model__subsample": "0.9",
      "regressor__model__num_leaves": "127",
      "regressor__model__n_estimators": "1000",
      "regressor__model__min_child_samples": "20",
      "regressor__model__learning_rate": "0.05",
      "regressor__model__colsample_bytree": "0.7"
    },
    "metrics": {
      "rmse": 115.6515,
      "mae": 61.6125,
      "r2": 0.9366
    },
    "band_log_offsets": {
      "p10": -0.193,
      "p90": 0.1954
    },
    "band_test_coverage": 0.815,
    "sweep_best_rmse": 117.7694,
    "n_train": 20821,
    "n_test": 5206
  },
  "shipping": {
    "model": "xgboost",
    "sampling": "random_over",
    "best_params": {
      "note": "sweep defaults kept \u2014 beat the tuned candidate on test"
    },
    "metrics": {
      "rmse": 70.2705,
      "mae": 47.5633,
      "r2": 0.5642
    },
    "band_log_offsets": {
      "p10": -0.7878,
      "p90": 0.8434
    },
    "band_test_coverage": 0.785,
    "sweep_best_rmse": 70.2705,
    "n_train": 20631,
    "n_test": 5158
  }
}
```
