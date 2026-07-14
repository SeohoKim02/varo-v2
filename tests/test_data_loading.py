"""Tests for actual Varo V2 sample loading and validation."""
import unittest
from pathlib import Path

from services.analysis_pipeline import calculate_overview_kpis, find_recommendation, top_recommendations
from services.data_loader import SAMPLE_FILENAME, load_excel_data
from services.data_validator import ERROR, validate_workbook_data
from services.recommendation_adapter import recommendations_from_dataframe


class SampleDataLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = Path(__file__).resolve().parents[1] / "data" / SAMPLE_FILENAME
        if not cls.sample_path.exists():
            raise AssertionError(f"sample workbook not found: {cls.sample_path}")
        cls.data = load_excel_data(cls.sample_path)
        cls.report = validate_workbook_data(cls.data)
        cls.recommendations = recommendations_from_dataframe(cls.data["recommendations"])

    def test_required_sheets_loaded(self):
        for key in ("stores", "products", "inventory", "routes", "recommendations"):
            self.assertIn(key, self.data)

    def test_dc_exactly_one_and_multiple_stores(self):
        self.assertEqual(self.report.summary["dc_count"], 1)
        self.assertGreaterEqual(self.report.summary["store_count"], 2)

    def test_sample_row_counts(self):
        self.assertEqual(len(self.data["stores"]), 11)
        self.assertEqual(len(self.data["products"]), 8)
        self.assertEqual(len(self.data["inventory"]), 80)
        self.assertEqual(len(self.data["routes"]), 110)
        self.assertEqual(len(self.data["recommendations"]), 8)

    def test_route_source_target_exist(self):
        self.assertFalse(any(message.level == ERROR and "stores.node_id" in message.message for message in self.report.messages))

    def test_direct_and_via_dc_counts(self):
        self.assertEqual(self.report.summary["direct_count"], 5)
        self.assertEqual(self.report.summary["via_dc_count"], 3)

    def test_actual_sample_has_no_error(self):
        self.assertFalse(self.report.has_errors, [message.to_dict() for message in self.report.messages])

    def test_top5_sorting(self):
        top5 = top_recommendations(self.recommendations, limit=5)
        self.assertEqual(len(top5), min(5, len(self.recommendations)))
        scores = [item["vhs_score"] for item in top5]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_kpi_aggregation(self):
        kpis = calculate_overview_kpis(self.recommendations, self.report)
        self.assertEqual(kpis["active_route_count"], 8)
        self.assertGreater(kpis["total_recommended_qty"], 0)
        self.assertGreater(kpis["total_expected_saving"], 0)
        self.assertEqual(kpis["data_quality"], self.report.status)

    def test_selected_route_id_lookup(self):
        route = find_recommendation(self.recommendations, "R002")
        self.assertIsNotNone(route)
        self.assertEqual(route["route_id"], "R002")

    def test_duplicate_node_id_validation(self):
        data = {key: value.copy() for key, value in self.data.items()}
        data["stores"].loc[1, "node_id"] = data["stores"].loc[0, "node_id"]
        report = validate_workbook_data(data)
        self.assertTrue(report.has_errors)

    def test_invalid_route_type_validation(self):
        data = {key: value.copy() for key, value in self.data.items()}
        data["recommendations"].loc[0, "route_type"] = "INVALID"
        report = validate_workbook_data(data)
        self.assertTrue(report.has_errors)

    def test_via_dc_requires_dc_id(self):
        data = {key: value.copy() for key, value in self.data.items()}
        via_idx = data["recommendations"].index[data["recommendations"]["route_type"] == "VIA_DC"][0]
        data["recommendations"].loc[via_idx, "dc_id"] = None
        report = validate_workbook_data(data)
        self.assertTrue(report.has_errors)

    def test_duplicate_route_id_validation(self):
        data = {key: value.copy() for key, value in self.data.items()}
        data["recommendations"].loc[1, "route_id"] = data["recommendations"].loc[0, "route_id"]
        report = validate_workbook_data(data)
        self.assertTrue(report.has_errors)

    def test_negative_cost_and_distance_validation(self):
        data = {key: value.copy() for key, value in self.data.items()}
        data["recommendations"].loc[0, "estimated_cost"] = -1
        data["recommendations"].loc[0, "distance_km"] = -1
        report = validate_workbook_data(data)
        self.assertTrue(report.has_errors)


if __name__ == "__main__":
    unittest.main()
