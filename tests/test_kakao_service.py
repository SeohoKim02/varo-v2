"""Kakao map service tests.

These tests do not call Kakao or load external scripts; they only validate the
HTML/payload builder contract used by the route-detail page.
"""
from __future__ import annotations

import unittest

import pandas as pd

from services.kakao_service import (
    KakaoRoutePayload,
    build_kakao_map_html,
    build_route_payload,
    get_kakao_key_from_sources,
    kakao_status_label,
    resolve_route_points,
)


class KakaoServiceTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "stores": pd.DataFrame(
                [
                    {"node_id": "DC01", "node_name": "서울 서북권 물류센터", "node_type": "DC", "latitude": 37.61, "longitude": 126.92},
                    {"node_id": "DC02", "node_name": "서울 도심권 물류센터", "node_type": "DC", "latitude": 37.56, "longitude": 127.00},
                    {"node_id": "S01", "node_name": "연신내점", "node_type": "STORE", "latitude": 37.62, "longitude": 126.91},
                    {"node_id": "S02", "node_name": "종로점", "node_type": "STORE", "latitude": 37.57, "longitude": 126.99},
                ]
            )
        }

    def test_key_missing_is_safe(self):
        self.assertIsNone(get_kakao_key_from_sources({}))
        self.assertEqual(kakao_status_label({}), "카카오 미연결")

    def test_key_present_generates_connected_label(self):
        self.assertEqual(kakao_status_label({"KAKAO_JAVASCRIPT_KEY": "abc"}), "카카오 연결")

    def test_direct_route_payload_has_source_and_target(self):
        payload = build_route_payload(
            self.data,
            {"route_id": "R001", "route_type": "DIRECT", "source_id": "S01", "target_id": "S02"},
        )
        self.assertTrue(payload.ok)
        self.assertEqual([point.role for point in payload.points], ["출발", "도착"])

    def test_via_dc_uses_row_specific_dc(self):
        payload = build_route_payload(
            self.data,
            {"route_id": "R002", "route_type": "VIA_DC", "source_id": "S01", "target_id": "S02", "dc_id": "DC02"},
        )
        self.assertTrue(payload.ok)
        self.assertEqual([point.role for point in payload.points], ["출발", "DC", "도착"])
        self.assertEqual(payload.points[1].node_id, "DC02")

    def test_via_dc_does_not_fallback_to_arbitrary_dc(self):
        payload = build_route_payload(
            self.data,
            {"route_id": "R003", "route_type": "VIA_DC", "source_id": "S01", "target_id": "S02", "dc_id": "DC99"},
        )
        self.assertFalse(payload.ok)
        self.assertIn("DC", payload.message)

    def test_missing_coordinates_return_message_not_exception(self):
        data = {"stores": pd.DataFrame([{"node_id": "S01", "node_type": "STORE"}, {"node_id": "S02", "node_type": "STORE"}])}
        payload = build_route_payload(data, {"route_id": "R001", "route_type": "DIRECT", "source_id": "S01", "target_id": "S02"})
        self.assertFalse(payload.ok)
        self.assertEqual(payload.message, "좌표 데이터가 부족합니다.")

    def test_via_dc_falls_back_deterministically_only_when_dc_missing(self):
        payload = resolve_route_points(
            self.data,
            {"route_id": "R004", "route_type": "VIA_DC", "source_id": "S01", "target_id": "S02"},
        )
        self.assertTrue(payload.ok)
        self.assertEqual(payload.points[1].role, "DC")
        self.assertIn(payload.points[1].node_id, {"DC01", "DC02"})

    def test_via_dc_can_find_dc_by_name(self):
        payload = resolve_route_points(
            self.data,
            {
                "route_id": "R005",
                "route_type": "VIA_DC",
                "source_id": "S01",
                "target_id": "S02",
                "dc_name": "서울 도심권 물류센터",
            },
        )
        self.assertTrue(payload.ok)
        self.assertEqual(payload.points[1].node_id, "DC02")

    def test_html_contains_sdk_markers_and_polyline(self):
        payload = KakaoRoutePayload(
            ok=True,
            route_id="R001",
            route_type="DIRECT",
            points=build_route_payload(
                self.data,
                {"route_id": "R001", "route_type": "DIRECT", "source_id": "S01", "target_id": "S02"},
            ).points,
        )
        html = build_kakao_map_html(payload, "test-key", height=420)
        self.assertIn("dapi.kakao.com/v2/maps/sdk.js", html)
        self.assertIn("appkey=test-key", html)
        self.assertIn("출발", html)
        self.assertIn("도착", html)
        self.assertIn("kakao.maps.Polyline", html)


if __name__ == "__main__":
    unittest.main()
