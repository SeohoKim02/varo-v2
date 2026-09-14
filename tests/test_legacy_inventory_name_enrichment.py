from __future__ import annotations

import pandas as pd

from services.legacy_adapters.data_adapter import prepare_legacy_data


def _data(inventory: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "stores": pd.DataFrame(
            [
                {"node_id": "S1", "node_name": "Store One", "node_type": "STORE"},
                {"node_id": "S2", "node_name": "Store Two", "node_type": "STORE"},
            ]
        ),
        "products": pd.DataFrame(
            [
                {"product_id": "P1", "product_name": "Product One"},
                {"product_id": "P2", "product_name": "Product Two"},
            ]
        ),
        "inventory": inventory,
        "routes": pd.DataFrame(),
        "config": pd.DataFrame(),
    }


def test_prepare_legacy_data_adds_master_names_to_inventory():
    inventory = pd.DataFrame(
        [
            {"store_id": "S1", "product_id": "P1", "stock_qty": 10},
            {"store_id": "S2", "product_id": "P2", "stock_qty": 20},
        ]
    )

    result = prepare_legacy_data(_data(inventory))["inventory"]

    assert len(result) == 2
    assert result["product_name"].tolist() == ["Product One", "Product Two"]
    assert result["store_name"].tolist() == ["Store One", "Store Two"]


def test_prepare_legacy_data_preserves_existing_names_and_fills_blanks():
    inventory = pd.DataFrame(
        [
            {
                "store_id": "S1",
                "product_id": "P1",
                "stock_qty": 10,
                "store_name": "Existing Store",
                "product_name": "Existing Product",
            },
            {
                "store_id": "S2",
                "product_id": "P2",
                "stock_qty": 20,
                "store_name": "",
                "product_name": None,
            },
        ]
    )

    result = prepare_legacy_data(_data(inventory))["inventory"]

    assert result.loc[0, "store_name"] == "Existing Store"
    assert result.loc[0, "product_name"] == "Existing Product"

    assert result.loc[1, "store_name"] == "Store Two"
    assert result.loc[1, "product_name"] == "Product Two"



def test_real_outbound_proxy_is_preserved_as_daily_proxy_and_30d_projection():
    inventory = pd.DataFrame(
        [
            {
                "store_id": "S1",
                "product_id": "P1",
                "stock_qty": 100,
                "sales_qty": 2,
                "sales_qty_semantics": "demand_proxy_not_retail_sales",
                "sales_qty_provenance": "derived_proxy_from_actual_outbound_qty",
            }
        ]
    )

    result = prepare_legacy_data(_data(inventory))["inventory"]

    assert float(result.loc[0, "avg_daily_sales"]) == 2.0
    assert float(result.loc[0, "sales_30d"]) == 60.0
    assert float(result.loc[0, "sales_30"]) == 60.0
    assert float(result.loc[0, "demand_qty"]) == 60.0

    assert (
        result.loc[0, "avg_daily_sales_provenance"]
        == "derived_daily_demand_proxy_from_actual_outbound"
    )

    assert (
        result.loc[0, "sales_30d_provenance"]
        == "derived_30d_projection_from_daily_demand_proxy"
    )
