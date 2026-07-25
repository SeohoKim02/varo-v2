"""Tests for the Varo V2 standard recommendation contract."""
import unittest

import pandas as pd

from services.recommendation_adapter import (
    StandardRecommendation,
    normalize_standard_recommendation,
    recommendations_from_dataframe,
    validate_standard_recommendation,
)


def _base_item(**overrides):
    item = StandardRecommendation(
        route_id="R001",
        product_id="P001",
        product_name="냉동만두",
        source_id="S01",
        source_name="연신내점",
        target_id="S04",
        target_name="홍제점",
        route_type="DIRECT",
        recommended_qty=12,
        transport_type="냉동/냉장 탑차",
        distance_km=2.4,
        travel_time_min=14,
        estimated_cost=8400,
        expected_saving=21600,
        vhs_score=84.3,
        recommendation_grade="최적",
        confidence_score=91.0,
        greedy_action="재고 이동",
        reason="폐기 위험과 수요 적합도를 고려한 추천",
    ).to_dict()
    item.update(overrides)
    return item


class RecommendationAdapterTests(unittest.TestCase):
    def test_standard_recommendation_is_valid(self):
        item = _base_item()
        validate_standard_recommendation(item)
        self.assertEqual(item["route_type"], "DIRECT")

    def test_via_dc_requires_dc_id(self):
        item = _base_item(route_id="R002", route_type="VIA_DC", dc_id=None, dc_name=None)
        with self.assertRaises(ValueError):
            validate_standard_recommendation(item)

    def test_normalize_rejects_unknown_route_type(self):
        item = _base_item(route_id="R003", route_type="UNKNOWN")
        with self.assertRaises(ValueError):
            normalize_standard_recommendation(item)

    def test_direct_recommendation_from_dataframe(self):
        df = pd.DataFrame([_base_item()])
        rows = recommendations_from_dataframe(df)
        self.assertEqual(rows[0]["route_type"], "DIRECT")
        self.assertIsNone(rows[0]["dc_id"])

    def test_via_dc_recommendation_from_dataframe(self):
        df = pd.DataFrame([_base_item(route_id="R002", route_type="VIA_DC", dc_id="DC01", dc_name="서울 서북권 물류센터")])
        rows = recommendations_from_dataframe(df)
        self.assertEqual(rows[0]["route_type"], "VIA_DC")
        self.assertEqual(rows[0]["dc_id"], "DC01")

    def test_duplicate_route_id_rejected(self):
        df = pd.DataFrame([_base_item(), _base_item(product_id="P002")])
        with self.assertRaises(ValueError):
            recommendations_from_dataframe(df)


if __name__ == "__main__":
    unittest.main()