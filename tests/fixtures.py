"""In-package test fixtures for Varo V2.

Builds a small, deterministic workbook entirely in code so tests run without the
original folder, the backup ZIP, or any DQN artifact. Includes route_id R001
(DIRECT) and R002 (VIA_DC); R002 is used for the selected_route_id persistence
tests. No reward/loss/model/q-table/policy columns are present.
"""
from __future__ import annotations

from io import BytesIO

import pandas as pd

from services.dqn_guard import dqn_exclusion_report

# Deterministic KPI expectations derived from the recommendation rows below.
EXPECTED_RECOMMENDATION_COUNT = 4
EXPECTED_TOTAL_QTY = 140.0          # 50 + 30 + 40 + 20
EXPECTED_TOTAL_SAVING = 47000.0     # 12000 + 20000 + 9000 + 6000
EXPECTED_AVERAGE_VHS = 68.75        # (78 + 72 + 65 + 60) / 4
DEFAULT_SELECTED_ROUTE_ID = "R001"  # highest VHS
PERSISTED_ROUTE_ID = "R002"


def stores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"node_id": "DC01", "node_name": "중앙물류센터", "store_name": "중앙물류센터", "node_type": "DC"},
            {"node_id": "S001", "node_name": "연신내점", "store_name": "연신내점", "node_type": "STORE"},
            {"node_id": "S002", "node_name": "홍대점", "store_name": "홍대점", "node_type": "STORE"},
            {"node_id": "S003", "node_name": "강남점", "store_name": "강남점", "node_type": "STORE"},
        ]
    )


def products_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"product_id": "P001", "product_name": "냉동만두500g", "unit_price": 4500},
            {"product_id": "P002", "product_name": "아이스크림1L", "unit_price": 6800},
        ]
    )


def inventory_frame() -> pd.DataFrame:
    rows = []
    names = {"P001": "냉동만두500g", "P002": "아이스크림1L"}
    store_names = {"S001": "연신내점", "S002": "홍대점", "S003": "강남점", "DC01": "중앙물류센터"}
    stock = {"S001": (120, 18), "S002": (80, 12), "S003": (60, 9), "DC01": (200, 0)}
    for store_id, (base, sales) in stock.items():
        for product_id in ("P001", "P002"):
            rows.append(
                {
                    "store_id": store_id,
                    "store_name": store_names[store_id],
                    "product_id": product_id,
                    "product_name": names[product_id],
                    "stock_qty": base,
                    "dead_stock_qty": max(0, base - sales * 4),
                    "sales_qty": sales,
                    "avg_daily_sales": sales,
                    "demand_qty": sales * 7,
                    "sales_7d": sales * 7,
                    "sales_30d": sales * 30,
                    "expiry_days": 14,
                    "days_to_expiry": 14,
                    "unit_price": 4500 if product_id == "P001" else 6800,
                    "order_cost": 1500,
                    "demand_std": 3.0,
                    "lead_time_days": 2,
                }
            )
    return pd.DataFrame(rows)


def routes_frame() -> pd.DataFrame:
    pairs = [
        ("S001", "S002", 4.2, 3000, 11.0),
        ("S002", "S001", 4.2, 3000, 11.0),
        ("S001", "DC01", 3.1, 2200, 8.0),
        ("DC01", "S001", 3.1, 2200, 8.0),
        ("DC01", "S003", 4.5, 2600, 9.5),
        ("S003", "DC01", 4.5, 2600, 9.5),
        ("S002", "S003", 5.0, 3300, 12.5),
        ("S003", "S001", 6.1, 3800, 14.0),
    ]
    return pd.DataFrame(
        [
            {
                "source_id": s,
                "target_id": t,
                "distance_km": d,
                "estimated_cost": c,
                "travel_time_min": m,
            }
            for s, t, d, c, m in pairs
        ]
    )


def recommendations_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "route_id": "R001", "product_id": "P001", "product_name": "냉동만두500g",
                "source_id": "S001", "source_name": "연신내점",
                "target_id": "S002", "target_name": "홍대점",
                "dc_id": None, "dc_name": None, "route_type": "DIRECT",
                "recommended_qty": 50, "transport_type": "일반 탑차",
                "distance_km": 4.2, "travel_time_min": 11.0,
                "estimated_cost": 3000, "expected_saving": 12000,
                "vhs_score": 78, "recommendation_grade": "높음",
                "confidence_score": 80, "reason": "재고 과잉과 수요 분포를 반영한 직접 이동 권장.",
            },
            {
                "route_id": "R002", "product_id": "P002", "product_name": "아이스크림1L",
                "source_id": "S001", "source_name": "연신내점",
                "target_id": "S003", "target_name": "강남점",
                "dc_id": "DC01", "dc_name": "중앙물류센터", "route_type": "VIA_DC",
                "recommended_qty": 30, "transport_type": "냉동 탑차",
                "distance_km": 7.6, "travel_time_min": 16.4,
                "estimated_cost": 5000, "expected_saving": 20000,
                "vhs_score": 72, "recommendation_grade": "보통",
                "confidence_score": 70, "reason": "냉동 품목의 DC 경유 합류 이동으로 비용 절감.",
            },
            {
                "route_id": "R003", "product_id": "P001", "product_name": "냉동만두500g",
                "source_id": "S002", "source_name": "홍대점",
                "target_id": "S003", "target_name": "강남점",
                "dc_id": None, "dc_name": None, "route_type": "DIRECT",
                "recommended_qty": 40, "transport_type": "일반 탑차",
                "distance_km": 5.0, "travel_time_min": 12.5,
                "estimated_cost": 2500, "expected_saving": 9000,
                "vhs_score": 65, "recommendation_grade": "보통",
                "confidence_score": 68, "reason": "회전율 차이를 반영한 점포 간 재배치.",
            },
            {
                "route_id": "R004", "product_id": "P002", "product_name": "아이스크림1L",
                "source_id": "S003", "source_name": "강남점",
                "target_id": "S001", "target_name": "연신내점",
                "dc_id": None, "dc_name": None, "route_type": "DIRECT",
                "recommended_qty": 20, "transport_type": "냉동 탑차",
                "distance_km": 6.1, "travel_time_min": 14.0,
                "estimated_cost": 2000, "expected_saving": 6000,
                "vhs_score": 60, "recommendation_grade": "낮음",
                "confidence_score": 62, "reason": "임박 재고 소진을 위한 보조 이동.",
            },
        ]
    )


def sample_workbook() -> dict[str, pd.DataFrame]:
    """A validation-passing workbook keyed like services.data_loader output."""
    return {
        "stores": stores_frame(),
        "products": products_frame(),
        "inventory": inventory_frame(),
        "routes": routes_frame(),
        "recommendations": recommendations_frame(),
    }


# Excel sheet names expected by services.data_loader.
_SHEET_NAMES = {
    "stores": "stores", "products": "products", "inventory": "inventory",
    "routes": "routes", "recommendations": "v2_recommendations",
}


def workbook_excel_bytes(sheets: dict[str, pd.DataFrame] | None = None) -> BytesIO:
    """Write a workbook dict to an in-memory .xlsx (BytesIO) for upload tests."""
    frames = sheets if sheets is not None else sample_workbook()
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for key, frame in frames.items():
            sheet_name = _SHEET_NAMES.get(key, key)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    buffer.seek(0)
    return buffer


def dqn_sample_frames(store_count: int = 2, dc_count: int = 1, *, names_as_endpoints: bool = False) -> dict[str, pd.DataFrame]:
    """DQN 10-pack style frames with a separate dcs sheet and alias columns."""
    stores = pd.DataFrame([
        {"store_id": f"S{i:02d}", "store_name": f"점포 {i:02d}"}
        for i in range(1, store_count + 1)
    ])
    dcs = pd.DataFrame([
        {"dc_id": f"DC{i:02d}", "dc_name": f"물류센터 {i:02d}"}
        for i in range(1, dc_count + 1)
    ])
    products = pd.DataFrame([{"item_id": "P01", "item_name": "신선 간편식", "price": 8500}])
    inventory = pd.DataFrame([
        {
            "store_id": f"S{i:02d}", "item_id": "P01", "quantity": 40 + i,
            "sales": 5 + i, "days_to_expiry": 9,
        }
        for i in range(1, store_count + 1)
    ])
    route_rows = []
    for i in range(1, store_count + 1):
        source_id = f"S{i:02d}"
        target_id = f"S{(i % store_count) + 1:02d}"
        route_rows.append({"source": source_id, "target": target_id, "distance": 2 + i, "cost": 900 + i * 50, "time": 7 + i})
        route_rows.append({"source": source_id, "target": f"DC{((i - 1) % dc_count) + 1:02d}", "distance": 1.5 + i, "cost": 800 + i * 40, "time": 5 + i})
        route_rows.append({"source": f"DC{((i - 1) % dc_count) + 1:02d}", "target": target_id, "distance": 1.8 + i, "cost": 820 + i * 40, "time": 6 + i})
    if store_count == 2:
        route_rows.append(dict(route_rows[0]))  # warning only: duplicate source/target route pair
    routes = pd.DataFrame(route_rows)

    name_by_id = {f"S{i:02d}": f"점포 {i:02d}" for i in range(1, store_count + 1)}
    rec_rows = []
    for i in range(max(4, min(store_count, 6))):
        source_id = f"S{(i % store_count) + 1:02d}"
        target_id = f"S{((i + 1) % store_count) + 1:02d}"
        via = i % 2 == 1
        endpoint_source = name_by_id[source_id] if names_as_endpoints else source_id
        endpoint_target = name_by_id[target_id] if names_as_endpoints else target_id
        common = {
            "recommendation_id": "DUP" if i in (1, 2) else None,
            "item_id": "P01", "item_name": "신선 간편식",
            "route_type": "DC 경유" if via else "직접 이동",
            "move_qty": 8 + i, "transport_mode": "냉장 탑차",
            "expected_savings_amount": 5000 + i * 500,
            "confidence": 70 + i,
        }
        if names_as_endpoints:
            common.update({"source": endpoint_source, "target": endpoint_target})
        else:
            common.update({
                "from_store_id": endpoint_source, "from_store_name": name_by_id[source_id],
                "to_store_id": endpoint_target, "to_store_name": name_by_id[target_id],
            })
        if via:
            dc_index = (i % dc_count) + 1
            common.update({"dc_id": f"DC{dc_index:02d}", "dc_name": f"물류센터 {dc_index:02d}"})
        rec_rows.append(common)
    return {
        "stores": stores, "dcs": dcs, "products": products,
        "inventory": inventory, "routes": routes, "recommendations": pd.DataFrame(rec_rows),
    }


def dqn_sample_excel_bytes(store_count: int = 2, dc_count: int = 1, *, names_as_endpoints: bool = False) -> BytesIO:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in dqn_sample_frames(store_count, dc_count, names_as_endpoints=names_as_endpoints).items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    buffer.seek(0)
    return buffer


def korean_recommendations_frame() -> pd.DataFrame:
    """Recommendations sheet using Korean column names and messy numbers."""
    return pd.DataFrame([
        {
            "추천id": "R001", "상품코드": "P001", "상품명": "냉동만두500g",
            "출발점포ID": "S001", "출발점포": "연신내점",
            "도착점포ID": "S002", "도착점포": "홍대점",
            "경로유형": "DIRECT", "추천수량": "50", "이동수단": "일반 탑차",
            "이동비용": "3,000원", "절감액": "12,000원", "기존VHS": "78",
            "추천등급": "높음", "신뢰도": "80", "사유": "재고 과잉",
        },
        {
            "추천id": "R002", "상품코드": "P002", "상품명": "아이스크림1L",
            "출발점포ID": "S001", "출발점포": "연신내점",
            "도착점포ID": "S003", "도착점포": "강남점",
            "물류센터id": "DC01", "물류센터": "중앙물류센터",
            "경로유형": "VIA_DC", "추천수량": "30", "이동수단": "냉동 탑차",
            "이동비용": "5,000원", "절감액": "20,000원", "기존VHS": "72",
            "추천등급": "보통", "신뢰도": "70", "사유": "DC 경유",
        },
    ])


def synthetic_pipeline_result(recommendations: list[dict]) -> dict:
    """Pipeline-shaped dict built in-package for state-machine tests.

    Mirrors a connected self-contained recompute (success) without invoking the
    analysis pipeline, so state tests stay deterministic regardless of which
    algorithms are present in _local_modules. The only deferrals are the two
    design-level history/DQN-excluded analyses.
    """
    return {
        "status": "success",
        "result_basis": "실제 V2 내부 알고리즘 재계산 결과 기준",
        "summary": {
            "total_recommended_qty": EXPECTED_TOTAL_QTY,
            "active_route_count": len(recommendations),
            "total_expected_saving": EXPECTED_TOTAL_SAVING,
            "average_vhs_score": EXPECTED_AVERAGE_VHS,
            "data_quality": "통과",
        },
        "recommendations": list(recommendations),
        "connected_algorithms": [
            "varo_hybrid_score.calculate_varo_hybrid_score",
            "heuristic_optimizer.add_heuristic_scores",
        ],
        "deferred_algorithms": [
            {"algorithm": "varo_sensitivity.run_hybrid_score_sensitivity_analysis",
             "reason": "이력 보정 그룹 입력 전제라 보류"},
            {"algorithm": "vhs_reason.get_reason_sentences",
             "reason": "이력 보정 그룹 전제라 연결 VHS 상황·기여도로 대체"},
        ],
        "warnings": [],
        "excluded_dqn_artifacts": dqn_exclusion_report(),
    }
