"""Kakao Maps JavaScript rendering helpers for Varo V2.

The SDK is only rendered from the route-detail page. This module never embeds
or persists an API key; it reads from Streamlit secrets or the process
environment only when explicitly requested by a page.
"""
from __future__ import annotations

import html
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

KAKAO_ENV_KEY = "KAKAO_JAVASCRIPT_KEY"
LAT_KEYS = ("latitude", "lat", "위도", "store_lat", "node_lat", "dc_lat")
LON_KEYS = ("longitude", "lon", "lng", "경도", "store_lng", "node_lng", "dc_lng")
ID_KEYS = ("node_id", "store_id", "dc_id", "id")
NAME_KEYS = ("node_name", "store_name", "dc_name", "name")
TYPE_KEYS = ("node_type", "store_type", "type")


@dataclass(frozen=True)
class KakaoPoint:
    role: str
    node_id: str
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class KakaoRoutePayload:
    ok: bool
    message: str = ""
    route_id: str | None = None
    route_type: str = "DIRECT"
    points: list[KakaoPoint] = field(default_factory=list)


def get_kakao_key(secrets: Mapping[str, Any] | None = None) -> Optional[str]:
    """Return a JavaScript key from Streamlit secrets first, then env."""
    if secrets is not None:
        try:
            value = secrets.get(KAKAO_ENV_KEY)
        except Exception:
            value = None
        if value:
            return str(value)
    value = os.environ.get(KAKAO_ENV_KEY)
    return str(value) if value else None


def get_kakao_key_from_env() -> Optional[str]:
    return os.environ.get(KAKAO_ENV_KEY)


def get_kakao_key_from_sources(secrets: Mapping[str, Any] | None = None) -> Optional[str]:
    return get_kakao_key(secrets)


def has_kakao_key(secrets: Mapping[str, Any] | None = None) -> bool:
    return bool(get_kakao_key(secrets))


def kakao_status_label(secrets: Mapping[str, Any] | None = None) -> str:
    return "카카오 연결" if has_kakao_key(secrets) else "카카오 미연결"


def _blank(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "nan", "none", "<na>", "nat"}


def _as_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.where(pd.notna(value), None).to_dict("records")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and not _blank(row.get(key)):
            return row.get(key)
    return None


def _number(value: Any) -> float | None:
    if _blank(value):
        return None
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_coordinate(lat: Any, lon: Any) -> tuple[float, float] | None:
    latitude = _number(lat)
    longitude = _number(lon)
    if latitude is None or longitude is None:
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    return latitude, longitude


def _node_id(row: Mapping[str, Any]) -> str:
    return str(_first(row, ID_KEYS) or "")


def _node_name(row: Mapping[str, Any]) -> str:
    return str(_first(row, NAME_KEYS) or _node_id(row))


def _is_dc(row: Mapping[str, Any]) -> bool:
    type_text = str(_first(row, TYPE_KEYS) or "").upper()
    node_id = _node_id(row).upper()
    name = _node_name(row).upper()
    return type_text == "DC" or node_id.startswith("DC") or "DC" in name or "센터" in _node_name(row) or "물류" in _node_name(row)


def _records(data: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    data = data or {}
    source = data.get("stores")
    if source is None:
        source = data.get("nodes")
    return _as_records(source)


def _node_lookup(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for raw in records:
        row = dict(raw)
        for key in (_node_id(row), _node_name(row)):
            if not _blank(key):
                lookup[str(key)] = row
    return lookup


def _point(role: str, node: Mapping[str, Any] | None, fallback_id: Any = None, fallback_name: Any = None) -> KakaoPoint | None:
    if not node:
        return None
    coord = _valid_coordinate(_first(node, LAT_KEYS), _first(node, LON_KEYS))
    if coord is None:
        return None
    node_id = str(_first(node, ID_KEYS) or fallback_id or "")
    name = str(_first(node, NAME_KEYS) or fallback_name or node_id)
    return KakaoPoint(role=role, node_id=node_id, name=name, latitude=coord[0], longitude=coord[1])


def _find_node(lookup: Mapping[str, dict[str, Any]], node_id: Any = None, node_name: Any = None) -> dict[str, Any] | None:
    if not _blank(node_id) and str(node_id) in lookup:
        return lookup[str(node_id)]
    if not _blank(node_name) and str(node_name) in lookup:
        return lookup[str(node_name)]
    return None


def _distance(a: KakaoPoint, b: KakaoPoint) -> float:
    return (a.latitude - b.latitude) ** 2 + (a.longitude - b.longitude) ** 2


def _deterministic_dc_fallback(
    records: Sequence[Mapping[str, Any]],
    source_point: KakaoPoint,
    target_point: KakaoPoint,
) -> dict[str, Any] | None:
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for raw in records:
        row = dict(raw)
        if not _is_dc(row):
            continue
        point = _point("DC", row)
        if not point:
            continue
        score = _distance(source_point, point) + _distance(point, target_point)
        candidates.append((score, point.node_id or point.name, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def resolve_route_points(data: Mapping[str, Any] | None, route: Mapping[str, Any] | None) -> KakaoRoutePayload:
    """Build marker/polyline input for the selected recommendation."""
    if not route:
        return KakaoRoutePayload(False, "추천 경로가 선택되지 않았습니다.")
    records = _records(data)
    lookup = _node_lookup(records)
    source = _find_node(lookup, route.get("source_id"), route.get("source_name"))
    target = _find_node(lookup, route.get("target_id"), route.get("target_name"))
    route_type = str(route.get("route_type") or "DIRECT").upper()

    source_point = _point("출발", source, route.get("source_id"), route.get("source_name"))
    target_point = _point("도착", target, route.get("target_id"), route.get("target_name"))
    if not source_point or not target_point:
        return KakaoRoutePayload(False, "좌표 데이터가 부족합니다.", str(route.get("route_id") or ""), route_type)

    points = [source_point]
    if route_type == "VIA_DC":
        requested_dc = not _blank(route.get("dc_id")) or not _blank(route.get("dc_name"))
        dc = _find_node(lookup, route.get("dc_id"), route.get("dc_name"))
        if dc is None and not requested_dc:
            dc = _deterministic_dc_fallback(records, source_point, target_point)
        dc_point = _point("DC", dc, route.get("dc_id"), route.get("dc_name"))
        if not dc_point:
            message = "DC 좌표 없음" if requested_dc else "좌표 데이터가 부족합니다."
            return KakaoRoutePayload(False, message, str(route.get("route_id") or ""), route_type)
        points.append(dc_point)
    points.append(target_point)
    return KakaoRoutePayload(True, route_id=str(route.get("route_id") or ""), route_type=route_type, points=points)


def build_route_payload(data: Mapping[str, Any] | None, route: Mapping[str, Any] | None) -> KakaoRoutePayload:
    return resolve_route_points(data, route)


def _js_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_kakao_map_html(payload: KakaoRoutePayload, javascript_key: str, height: int = 460) -> str:
    """Return standalone HTML for Streamlit components."""
    safe_key = html.escape(str(javascript_key), quote=True)
    marker_js = ",\n".join(
        "{"
        f"role: {_js_string(point.role)}, name: {_js_string(point.name)}, "
        f"lat: {point.latitude:.8f}, lng: {point.longitude:.8f}"
        "}"
        for point in payload.points
    )
    stroke_color = "#3b82f6" if payload.route_type == "VIA_DC" else "#1f766d"
    stroke_style = "dashed" if payload.route_type == "VIA_DC" else "solid"
    return f"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <style>
    html, body, #map {{ width: 100%; height: {int(height)}px; margin: 0; padding: 0; }}
    .label {{ display:inline-block; padding:4px 8px; border-radius:999px; background:#ffffff; border:1px solid #dfe5ec; color:#1b2533; font-size:12px; font-weight:700; box-shadow:0 2px 8px rgba(0,0,0,.10); }}
  </style>
  <script src="https://dapi.kakao.com/v2/maps/sdk.js?appkey={safe_key}&autoload=false"></script>
</head>
<body>
  <div id="map"></div>
  <script>
    const routePoints = [{marker_js}];
    kakao.maps.load(function() {{
      const map = new kakao.maps.Map(document.getElementById('map'), {{
        center: new kakao.maps.LatLng(routePoints[0].lat, routePoints[0].lng),
        level: 6
      }});
      const bounds = new kakao.maps.LatLngBounds();
      const linePath = [];
      routePoints.forEach(function(point) {{
        const latLng = new kakao.maps.LatLng(point.lat, point.lng);
        bounds.extend(latLng);
        linePath.push(latLng);
        new kakao.maps.Marker({{ map: map, position: latLng, title: point.name }});
        const label = new kakao.maps.CustomOverlay({{
          position: latLng,
          yAnchor: 1.75,
          content: '<span class="label">' + point.role + '</span>'
        }});
        label.setMap(map);
      }});
      new kakao.maps.Polyline({{
        map: map,
        path: linePath,
        strokeWeight: 4,
        strokeColor: '{stroke_color}',
        strokeOpacity: 0.86,
        strokeStyle: '{stroke_style}'
      }});
      map.setBounds(bounds, 52, 52, 52, 52);
    }});
  </script>
</body>
</html>
"""


def render_kakao_route_map(data: Mapping[str, Any], route: Mapping[str, Any], key: str, height: int = 460) -> str:
    """Build HTML for a route map. The caller renders it only on route detail."""
    payload = resolve_route_points(data, route)
    if not payload.ok:
        return ""
    return build_kakao_map_html(payload, key, height=height)
