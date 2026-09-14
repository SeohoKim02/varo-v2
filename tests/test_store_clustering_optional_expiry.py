from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "services"
    / "legacy_adapters"
    / "_local_modules"
    / "store_clustering.py"
)

spec = importlib.util.spec_from_file_location(
    "varo_test_store_clustering",
    MODULE_PATH,
)

assert spec is not None
assert spec.loader is not None

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_store_clustering_accepts_missing_expiry_without_inventing_zero():
    stores = pd.DataFrame(
        [
            {
                "store_name": "A",
                "type": "STORE",
                "latitude": 37.5,
                "longitude": 127.0,
            },
            {
                "store_name": "B",
                "type": "STORE",
                "latitude": 37.6,
                "longitude": 127.1,
            },
            {
                "store_name": "C",
                "type": "STORE",
                "latitude": 37.7,
                "longitude": 127.2,
            },
        ]
    )

    inventory = pd.DataFrame(
        [
            {
                "store_name": "A",
                "product_name": "P1",
                "avg_daily_sales": 10.0,
                "stock_qty": 100.0,
                "dead_stock_qty": 20.0,
            },
            {
                "store_name": "B",
                "product_name": "P1",
                "avg_daily_sales": 20.0,
                "stock_qty": 200.0,
                "dead_stock_qty": 50.0,
            },
            {
                "store_name": "C",
                "product_name": "P2",
                "avg_daily_sales": 5.0,
                "stock_qty": 300.0,
                "dead_stock_qty": 150.0,
            },
        ]
    )

    features = module._build_store_features(
        stores,
        inventory,
    )

    assert "days_to_expiry" in features.columns
    assert features["days_to_expiry"].isna().all()

    analyzed, summary, cluster_map = (
        module.analyze_store_clustering(
            stores,
            inventory,
        )
    )

    assert not analyzed.empty
    assert not summary.empty
    assert cluster_map

    assert analyzed["days_to_expiry"].isna().all()
