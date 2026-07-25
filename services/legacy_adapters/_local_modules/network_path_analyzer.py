import heapq
from datetime import time

import pandas as pd


def time_to_minutes(value):
    if pd.isna(value):
        return None

    if isinstance(value, time):
        return value.hour * 60 + value.minute

    if isinstance(value, (int, float)):
        numeric = float(value)

        # 엑셀 시간값: 0.3333 = 08:00, 0.9166 = 22:00
        if 0 <= numeric < 1:
            return int(round(numeric * 24 * 60))

        # 8.5 = 08:30
        if 1 <= numeric < 24:
            hour = int(numeric)
            minute = int(round((numeric - hour) * 60))
            return hour * 60 + minute

        # 이미 분 단위로 들어온 값
        return int(round(numeric))

    text = str(value).strip()
    parts = text.split(":")

    if len(parts) >= 2:
        try:
            return int(float(parts[0])) * 60 + int(float(parts[1]))
        except Exception:
            return None

    try:
        numeric = float(text)

        if 0 <= numeric < 1:
            return int(round(numeric * 24 * 60))

        if 1 <= numeric < 24:
            hour = int(numeric)
            minute = int(round((numeric - hour) * 60))
            return hour * 60 + minute

        return int(round(numeric))
    except Exception:
        return None


def is_within_window(target_min, start_min, end_min):
    if start_min is None or end_min is None:
        return False

    target_min = int(target_min) % (24 * 60)
    start_min = int(start_min) % (24 * 60)
    end_min = int(end_min) % (24 * 60)

    if start_min <= end_min:
        return start_min <= target_min <= end_min

    # 예: 22:00 ~ 02:00
    return target_min >= start_min or target_min <= end_min


def minutes_to_time_text(minutes):
    if minutes is None or pd.isna(minutes):
        return "계산불가"

    minutes = int(minutes)

    if minutes >= 24 * 60:
        return "당일 초과"

    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _num(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def build_graph(routes):
    graph = {}

    if routes is None or routes.empty:
        return graph

    for row in routes.itertuples(index=False):
        r = row._asdict()
        from_id = str(r.get("from_id", r.get("source_id", ""))).strip()
        to_id = str(r.get("to_id", r.get("target_id", ""))).strip()

        if not from_id or not to_id:
            continue

        distance = _num(r.get("distance_km", r.get("route_distance_km", 0)), 0)
        time_min = _num(r.get("travel_time_min", r.get("time_min", distance / 30 * 60)), distance / 30 * 60)

        if "transport_cost" in r and not pd.isna(r.get("transport_cost")):
            cost = _num(r.get("transport_cost"), distance * _num(r.get("cost_per_km", 900), 900))
        else:
            cost = distance * _num(r.get("cost_per_km", 900), 900) + _num(r.get("fixed_cost", 0), 0)

        edge = {"to": to_id, "distance": distance, "time": time_min, "cost": cost}
        reverse_edge = {"to": from_id, "distance": distance, "time": time_min, "cost": cost}

        graph.setdefault(from_id, []).append(edge)
        graph.setdefault(to_id, []).append(reverse_edge)

    return graph


def dijkstra_lowest_cost(graph, start, end):
    start = str(start)
    end = str(end)

    if start == end:
        return {"path": [start], "total_distance": 0, "total_time": 0, "total_cost": 0}

    queue = [(0, start)]
    costs = {start: 0}
    previous = {start: None}
    edge_info = {}

    while queue:
        current_cost, current_node = heapq.heappop(queue)

        if current_node == end:
            break

        if current_cost > costs.get(current_node, float("inf")):
            continue

        for edge in graph.get(current_node, []):
            next_node = edge["to"]
            new_cost = current_cost + edge["cost"]

            if new_cost < costs.get(next_node, float("inf")):
                costs[next_node] = new_cost
                previous[next_node] = current_node
                edge_info[(current_node, next_node)] = edge
                heapq.heappush(queue, (new_cost, next_node))

    if end not in previous:
        return None

    path = []
    node = end

    while node is not None:
        path.append(node)
        node = previous[node]

    path.reverse()

    total_distance = 0
    total_time = 0
    total_cost = 0

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        edge = edge_info[(a, b)]
        total_distance += edge["distance"]
        total_time += edge["time"]
        total_cost += edge["cost"]

    return {
        "path": path,
        "total_distance": total_distance,
        "total_time": total_time,
        "total_cost": total_cost,
    }


def check_path_time_window(path_result, start_map, end_map, departure_time):
    departure_min = departure_time.hour * 60 + departure_time.minute
    path = path_result["path"]

    start_node = path[0]
    if not is_within_window(departure_min, start_map.get(start_node), end_map.get(start_node)):
        return False, "출발 점포 거래가능시간 불만족", None

    arrival_min = departure_min + path_result["total_time"]

    if arrival_min > 24 * 60:
        return False, "도착 시간이 당일 범위를 초과", arrival_min

    end_node = path[-1]
    if not is_within_window(arrival_min, start_map.get(end_node), end_map.get(end_node)):
        return False, "도착 점포 거래가능시간 불만족", arrival_min

    return True, "거래가능시간 만족", arrival_min


def analyze_multi_store_network_paths(
    stores,
    products,
    routes,
    transfer_path_result,
    departure_time,
):
    """
    속도 개선판.
    기존에는 transfer 결과 전체에 대해 Dijkstra를 반복해서 느렸음.
    이제는 비용이 낮은 후보 상위 120개만 다중경로 검토하고,
    같은 출발-도착 조합은 캐시해서 한 번만 계산함.
    """

    if transfer_path_result is None or transfer_path_result.empty:
        return pd.DataFrame(), "점포 간 이동 후보가 없습니다."

    stores_data = stores.copy()
    products_data = products.copy()
    transfer_data = transfer_path_result.copy()

    # 다중경로는 상세 검토 성격이라 상위 후보만 봐도 충분함
    sort_col = "estimated_cost" if "estimated_cost" in transfer_data.columns else None
    if sort_col:
        transfer_data["_sort_cost"] = pd.to_numeric(transfer_data[sort_col], errors="coerce")
        transfer_data = transfer_data.sort_values("_sort_cost", na_position="last").head(120)
    else:
        transfer_data = transfer_data.head(120)

    graph = build_graph(routes)

    if stores_data is None or stores_data.empty:
        return pd.DataFrame(), "점포 데이터가 없습니다."

    store_name_to_id = dict(zip(stores_data["store_name"], stores_data["store_id"]))
    store_id_to_name = dict(zip(stores_data["store_id"], stores_data["store_name"]))

    if "available_start" in stores_data.columns:
        start_map = dict(zip(stores_data["store_id"], stores_data["available_start"].apply(time_to_minutes)))
    else:
        start_map = {}

    if "available_end" in stores_data.columns:
        end_map = dict(zip(stores_data["store_id"], stores_data["available_end"].apply(time_to_minutes)))
    else:
        end_map = {}

    if products_data is not None and not products_data.empty and "product_id" in products_data.columns:
        if "distance_cutline_km" in products_data.columns:
            cutline_map = dict(zip(products_data["product_id"], products_data["distance_cutline_km"]))
        else:
            cutline_map = {}
    else:
        cutline_map = {}

    result_rows = []
    path_cache = {}

    for row in transfer_data.itertuples(index=False):
        r = row._asdict()

        product_id = r.get("product_id")
        product_name = r.get("product_name", "-")
        source_store_name = r.get("source_store", "-")
        target_store_name = r.get("target_store", "-")

        source_id = r.get("source_store_id") or store_name_to_id.get(source_store_name)
        target_id = r.get("target_store_id") or store_name_to_id.get(target_store_name)

        if source_id is None or target_id is None:
            continue

        source_id = str(source_id)
        target_id = str(target_id)

        cache_key = (source_id, target_id)
        if cache_key in path_cache:
            path_result = path_cache[cache_key]
        else:
            path_result = dijkstra_lowest_cost(graph, source_id, target_id)
            path_cache[cache_key] = path_result

        if path_result is None:
            result_rows.append(
                {
                    "product_name": product_name,
                    "source_store": source_store_name,
                    "target_store": target_store_name,
                    "network_path": "경로 없음",
                    "network_distance_km": None,
                    "network_time_min": None,
                    "network_cost": None,
                    "arrival_time": "계산불가",
                    "cutline_status": "불가능",
                    "time_status": "불가능",
                    "network_status": "불가능",
                    "network_recommendation": "경로 없음",
                    "reason": "연결 가능한 경로가 없습니다.",
                }
            )
            continue

        cutline = _num(cutline_map.get(product_id, 999999), 999999)
        cutline_ok = path_result["total_distance"] <= cutline

        time_ok, time_reason, arrival_min = check_path_time_window(
            path_result,
            start_map,
            end_map,
            departure_time,
        )

        network_available = cutline_ok and time_ok

        path_names = [store_id_to_name.get(store_id, store_id) for store_id in path_result["path"]]

        direct_cost = r.get("direct_cost")
        via_cost = r.get("via_cost")

        available_costs = []

        if not pd.isna(direct_cost):
            available_costs.append(float(direct_cost))

        if not pd.isna(via_cost):
            available_costs.append(float(via_cost))

        existing_best_cost = min(available_costs) if available_costs else None

        if not network_available:
            network_recommendation = "다중 경로 비추천"
            reason = "제품별 거리 컷라인을 초과합니다." if not cutline_ok else time_reason
        elif existing_best_cost is None:
            network_recommendation = "다중 경로 추천"
            reason = "기존 직접/DC 경유 이동이 어렵고, 다중 연결 경로가 가능합니다."
        elif path_result["total_cost"] < existing_best_cost:
            network_recommendation = "다중 경로 추천"
            reason = "다중 연결 경로가 기존 직접/DC 경유 방식보다 비용이 낮습니다."
        else:
            network_recommendation = "기존 경로 유지"
            reason = "기존 직접/DC 경유 방식이 다중 연결 경로보다 비용이 낮거나 같습니다."

        result_rows.append(
            {
                "product_name": product_name,
                "source_store": source_store_name,
                "target_store": target_store_name,
                "network_path": " → ".join(path_names),
                "network_distance_km": round(path_result["total_distance"], 1),
                "network_time_min": round(path_result["total_time"], 1),
                "network_cost": round(path_result["total_cost"], 1),
                "arrival_time": minutes_to_time_text(arrival_min),
                "distance_cutline_km": cutline,
                "cutline_status": "가능" if cutline_ok else "불가능",
                "time_status": "가능" if time_ok else "불가능",
                "network_status": "가능" if network_available else "불가능",
                "network_recommendation": network_recommendation,
                "reason": reason,
            }
        )

    return pd.DataFrame(result_rows), None
