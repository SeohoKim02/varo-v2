"""
최소비용 네트워크 알고리즘 (Minimum Cost Network Flow)
───────────────────────────────────────────────────────────
재고 과잉 점포 → DC → 재고 부족 점포 경로에서
총 이동비용을 최소화하는 흐름을 계산한다.

  네트워크 구성
    공급 노드  : 재고 과잉 점포 (excess > 0)
    수요 노드  : 재고 부족 점포 (excess < 0)
    중간 노드  : DC (허브 역할)
    엣지 비용  : routes 시트의 transport_cost 사용

  알고리즘
    Successive Shortest Path (SSP) — 음수 사이클 없는 그리디 최단경로 반복
    scipy.sparse.csgraph.shortest_path 기반 (Dijkstra)

  반환
    flow_df   : 각 (from→to) 엣지의 최적 흐름량·비용
    node_df   : 각 점포의 공급/수요/처리 결과
    total_cost: 전체 네트워크 최소 이동비용
    network_score: 0~100 (재고 불균형 해소율)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import heapq

# ─── 상수 ────────────────────────────────────────────────
_SUPPLY_30D_DAYS  = 30       # 30일치 수요를 기준으로 excess 계산
_MIN_EXCESS_QTY   = 50       # 최소 공급 가능 수량 (너무 작은 excess 제외)
_MIN_SHORTAGE_QTY = 50       # 최소 수요 수량
_MAX_NODES        = 50       # 계산 성능을 위해 공급/수요 상위 N개만 사용
_INF              = float("inf")


def _safe(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


# ─── 노드 생성 ────────────────────────────────────────────

def _build_nodes(inventory_df: pd.DataFrame,
                 stores_df: pd.DataFrame) -> pd.DataFrame:
    """
    점포별 공급/수요 판단.
    SUPPLY : 악성재고(dead_stock_qty)가 많은 점포 — 재고를 내보내야 함
    DEMAND : 활성재고(total_stock - dead_stock)가 수요에 비해 부족한 점포
    DC는 중간 허브 노드 (excess = 0)
    """
    grp = inventory_df.groupby("store_name").agg(
        total_stock   =("stock_qty",       "sum"),
        total_dead    =("dead_stock_qty",   "sum"),
        demand_qty    =("demand_qty",       "sum"),
        avg_daily     =("avg_daily_sales",  "sum"),
    ).reset_index()

    # 활성 재고 = total_stock - dead_stock
    grp["active_stock"] = (grp["total_stock"] - grp["total_dead"]).clip(lower=0)
    grp["dead_ratio"]   = (grp["total_dead"] / grp["total_stock"].replace(0, 1)).round(3)

    # 공급 가능량 = 악성재고 (최대 500개로 캡 — 계산 현실화)
    grp["excess"]   = grp["total_dead"].clip(upper=500).round(0)
    # 수요 부족량 = demand_qty - active_stock (최대 500개 캡)
    grp["shortage"] = (grp["demand_qty"] - grp["active_stock"]).clip(lower=0, upper=500).round(0)

    # 역할 분류: dead_ratio 중앙값 기준 — 상위 절반 SUPPLY, 하위 절반 DEMAND
    dead_med = grp["dead_ratio"].median()
    grp["role"] = grp["dead_ratio"].apply(
        lambda r: "SUPPLY" if r >= dead_med else "DEMAND"
    )

    # DC 노드 추가
    store_col = next((c for c in ["type","store_type"] if c in stores_df.columns), None)
    if store_col:
        dc_rows = stores_df[stores_df[store_col].str.upper() == "DC"][["store_name"]].copy()
        dc_rows["total_stock"] = 0; dc_rows["total_dead"] = 0
        dc_rows["avg_daily_demand"] = 0; dc_rows["supply_30d"] = 0
        dc_rows["excess"] = 0; dc_rows["shortage"] = 0; dc_rows["role"] = "DC"
        grp = pd.concat([grp, dc_rows], ignore_index=True)

    return grp.reset_index(drop=True)


# ─── 그래프 구성 ──────────────────────────────────────────

def _build_graph(routes_df: pd.DataFrame,
                 nodes: list[str]) -> dict:
    """
    nodes 목록에 있는 점포만 포함하는 방향 그래프.
    반환: {from_name: [(cost, to_name, capacity)]}
    """
    node_set = set(nodes)
    graph = defaultdict(list)

    for _, r in routes_df.iterrows():
        frm = r.get("from_name") or r.get("from_store", "")
        to  = r.get("to_name")   or r.get("to_store", "")
        if frm not in node_set or to not in node_set:
            continue
        if str(r.get("route_available", "Y")).upper() != "Y":
            continue
        cost = float(_safe(pd.Series([r.get("transport_cost", 5000)]), 5000).iloc[0])
        graph[frm].append((cost, to))

    return graph


# ─── Dijkstra 최단경로 ───────────────────────────────────

def _dijkstra(graph: dict, source: str) -> dict:
    dist = defaultdict(lambda: _INF)
    dist[source] = 0.0
    prev = {}
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for cost, v in graph.get(u, []):
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dict(dist), prev


def _get_path(prev: dict, src: str, dst: str) -> list[str]:
    path = []
    cur = dst
    while cur != src:
        path.append(cur)
        if cur not in prev:
            return []
        cur = prev[cur]
    path.append(src)
    return list(reversed(path))


# ─── SSP (Successive Shortest Path) ─────────────────────

def _successive_shortest_path(
    graph: dict,
    supply_nodes: dict,   # {name: supply_qty}
    demand_nodes: dict,   # {name: demand_qty}
) -> list[dict]:
    """
    공급 노드에서 수요 노드로 최소비용 경로를 반복 탐색.
    반환: flow_records list
    """
    remaining_supply = dict(supply_nodes)
    remaining_demand = dict(demand_nodes)
    flows = []

    supply_list = sorted(remaining_supply.keys(),
                         key=lambda x: -remaining_supply[x])
    demand_list = sorted(remaining_demand.keys(),
                         key=lambda x: -remaining_demand[x])

    for src in supply_list:
        if remaining_supply[src] <= 0:
            continue

        dist, prev = _dijkstra(graph, src)

        # 수요 노드를 비용 오름차순으로 시도
        demand_by_cost = sorted(
            [(dist.get(d, _INF), d) for d in demand_list if remaining_demand.get(d, 0) > 0],
        )

        for cost, dst in demand_by_cost:
            if cost == _INF:
                continue
            if remaining_supply[src] <= 0:
                break

            path = _get_path(prev, src, dst)
            if not path:
                continue

            qty = min(remaining_supply[src], remaining_demand.get(dst, 0))
            if qty <= 0:
                continue

            remaining_supply[src]   -= qty
            remaining_demand[dst]    = remaining_demand.get(dst, 0) - qty

            flows.append({
                "from_store":    src,
                "to_store":      dst,
                "flow_qty":      round(qty, 1),
                "unit_cost":     round(cost, 0),
                "total_cost":    round(cost * qty, 0),
                "path":          " → ".join(path),
                "path_length":   len(path) - 1,
            })

    return flows


# ─── 메인 함수 ───────────────────────────────────────────

def analyze_min_cost_network(
    inventory_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    max_nodes: int = _MAX_NODES,
) -> tuple:
    """
    최소비용 네트워크 흐름 분석.

    Returns
    -------
    flow_df      : 최적 이동 흐름 DataFrame
    node_df      : 점포별 공급/수요 처리 결과
    summary      : {total_cost, total_flow_qty, solved_rate, network_score}
    """
    if inventory_df is None or inventory_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # 노드 생성
    node_df = _build_nodes(inventory_df, stores_df)

    supply_df = node_df[node_df["role"] == "SUPPLY"].nlargest(max_nodes // 2, "excess")
    demand_df = node_df[node_df["role"] == "DEMAND"].nlargest(max_nodes // 2, "shortage") if "shortage" in node_df.columns else node_df[node_df["role"] == "DEMAND"]
    dc_df     = node_df[node_df["role"] == "DC"]

    if supply_df.empty or demand_df.empty:
        return pd.DataFrame(), node_df, {
            "total_cost": 0, "total_flow_qty": 0,
            "solved_rate": 0, "network_score": 0,
        }

    all_nodes = (
        supply_df["store_name"].tolist()
        + demand_df["store_name"].tolist()
        + dc_df["store_name"].tolist()
    )

    # 그래프 구성
    graph = _build_graph(routes_df, all_nodes)

    # 공급·수요 딕셔너리
    supply_nodes = dict(zip(supply_df["store_name"], supply_df["excess"].clip(lower=0)))
    # DEMAND 흡수용량 = 7일치 일평균 판매량 (얼마나 재고를 흡수할 수 있는지)
    absorb_col = "avg_daily" if "avg_daily" in demand_df.columns else None
    if absorb_col:
        demand_vals = (demand_df[absorb_col] * 7).clip(lower=_MIN_SHORTAGE_QTY, upper=500).round(0)
    else:
        demand_vals = pd.Series([200.0] * len(demand_df), index=demand_df.index)
    demand_nodes = dict(zip(demand_df["store_name"], demand_vals))

    # SSP 실행
    flows = _successive_shortest_path(graph, supply_nodes, demand_nodes)

    if not flows:
        return pd.DataFrame(), node_df, {
            "total_cost": 0, "total_flow_qty": 0,
            "solved_rate": 0, "network_score": 0,
        }

    flow_df = pd.DataFrame(flows)

    # 요약 통계
    total_cost     = flow_df["total_cost"].sum()
    total_flow_qty = flow_df["flow_qty"].sum()
    total_demand   = sum(demand_nodes.values())
    solved_qty     = min(total_flow_qty, total_demand)
    solved_rate    = (solved_qty / total_demand * 100) if total_demand > 0 else 0
    network_score  = round(solved_rate, 1)

    summary = {
        "total_cost":     round(total_cost, 0),
        "total_flow_qty": round(total_flow_qty, 1),
        "solved_rate":    round(solved_rate, 1),
        "network_score":  network_score,
        "n_supply":       len(supply_df),
        "n_demand":       len(demand_df),
        "n_flows":        len(flow_df),
        "avg_path_len":   round(flow_df["path_length"].mean(), 1),
    }

    return flow_df, node_df, summary


def add_network_score_to_recommendations(
    final_recommendations: pd.DataFrame,
    flow_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    final_recommendations 에 network_cost_score 추가.
    최소비용 경로 상에 있는 추천이면 가점.
    """
    if final_recommendations is None or flow_df is None or flow_df.empty:
        return final_recommendations

    df = final_recommendations.copy()

    # 최소비용 경로 상 (source→target) 조합을 set으로 구성
    optimal_pairs = set(
        zip(flow_df["from_store"], flow_df["to_store"])
    )

    df["is_optimal_route"] = df.apply(
        lambda r: (r.get("source_store", ""), r.get("target_store", "")) in optimal_pairs,
        axis=1,
    )

    # 최소비용 흐름 단위비용을 join
    cost_map = flow_df.groupby(["from_store", "to_store"])["unit_cost"].min().to_dict()
    df["network_unit_cost"] = df.apply(
        lambda r: cost_map.get((r.get("source_store"), r.get("target_store")), None),
        axis=1,
    )

    # 네트워크 점수: 최적 경로면 가점
    max_cost = flow_df["unit_cost"].max() if not flow_df.empty else 1
    df["network_cost_score"] = df.apply(
        lambda r: round((1 - r["network_unit_cost"] / max_cost) * 100, 1)
        if pd.notna(r.get("network_unit_cost")) else 50.0,
        axis=1,
    )

    return df
