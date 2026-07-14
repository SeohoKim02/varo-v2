"""Tests for download/export builders and DQN-free guarantees.

These tests are self-contained within varo_v2: recommendations come from the
pure in-package adapter and the pipeline structure is synthesized here, so no
legacy/backup module is imported and no DQN artifact is read.
"""
from __future__ import annotations

import unittest
from io import BytesIO

import pandas as pd

from services import export_service
from services.data_loader import get_default_sample_path, load_excel_data
from services.data_validator import validate_workbook_data
from services.dqn_guard import dqn_exclusion_report
from services.recommendation_adapter import recommendations_from_dataframe

_DQN_VALUE_TOKENS = (
    "reward", "loss", "q_table", "qtable", "policy_table",
    "replay_buffer", "model_path", "training_log",
)


def _sheet_frames(xlsx_bytes: bytes) -> dict[str, pd.DataFrame]:
    excel = pd.ExcelFile(BytesIO(xlsx_bytes))
    return {name: pd.read_excel(excel, sheet_name=name) for name in excel.sheet_names}


def _synthetic_pipeline(recommendations: list[dict]) -> dict:
    """A pipeline-shaped dict built from in-package data only.

    A reward/loss column is deliberately injected into the greedy rows to prove
    the exporter strips DQN-derived columns.
    """
    return {
        "status": "partial",
        "result_basis": "테스트 합성 결과 기준",
        "connected_algorithms": [
            "varo_hybrid_score.calculate_varo_hybrid_score",
            "heuristic_optimizer.add_heuristic_scores",
        ],
        "deferred_algorithms": [
            {"algorithm": "varo_sensitivity.run_hybrid_score_sensitivity_analysis", "reason": "보류"},
        ],
        "warnings": ["민감도 분석 보류"],
        "excluded_dqn_artifacts": dqn_exclusion_report(),
        "vhs_analysis": {
            "comparison_rows": [
                {
                    "route_id": rec["route_id"],
                    "product_name": rec["product_name"],
                    "uploaded_vhs_score": rec.get("uploaded_vhs_score"),
                    "recalculated_vhs_score": rec.get("vhs_score"),
                    "score_difference": 0.0,
                }
                for rec in recommendations
            ],
        },
        "greedy_analysis": {
            "rows": [
                {
                    "route_id": rec["route_id"],
                    "greedy_action": rec.get("greedy_action"),
                    "reward": 999999,  # must be stripped
                    "loss": 1.0,       # must be stripped
                }
                for rec in recommendations
            ],
        },
        "confidence_analysis": {"average": 66.0, "score_range": [66.0, 66.0]},
        "validation_report": {
            "optimality_gap": {
                "status": "계산 가능",
                "gap_str": "16.4%",
                "match_rate": 80.0,
                "varo_total": 13973.0,
                "opt_total": 12000.0,
                "comparable_candidate_count": 8,
                "opt_method": "milp",
                "calculation_function": "varo_optimality_gap.calculate_optimality_gap",
                "formula": "(Varo TOP-K 비용 - 최소비용 TOP-K) / 최소비용 TOP-K × 100",
            },
        },
    }


class ExportServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_excel_data(get_default_sample_path())
        cls.recommendations = recommendations_from_dataframe(cls.data["recommendations"])
        cls.pipeline = _synthetic_pipeline(cls.recommendations)
        cls.validation = validate_workbook_data(cls.data)

    def test_recommendation_csv_has_bom_and_rows(self):
        csv = export_service.recommendations_csv_bytes(self.recommendations)
        self.assertTrue(csv.startswith(b"\xef\xbb\xbf"))
        text = csv.decode("utf-8-sig")
        self.assertIn("route_id", text)
        self.assertIn("VHS(재계산)", text)
        self.assertEqual(len([line for line in text.splitlines() if line.strip()]), 9)

    def test_recommendation_excel_columns_and_dqn_sentinel(self):
        frames = _sheet_frames(export_service.recommendations_excel_bytes(self.recommendations))
        self.assertEqual(list(frames.keys()), ["추천결과"])
        frame = frames["추천결과"]
        self.assertEqual(len(frame), 8)
        for column in ("순위", "route_id", "상품", "VHS(재계산)", "업로드 VHS", "DQN 상태", "추천 이유"):
            self.assertIn(column, frame.columns)
        self.assertEqual(sorted(frame["DQN 상태"].unique().tolist()), ["미연결"])

    def test_analysis_workbook_sheets(self):
        frames = _sheet_frames(
            export_service.analysis_result_excel_bytes(self.pipeline, self.recommendations)
        )
        for sheet in ("추천결과", "VHS분석", "VHS비교", "Greedy분석", "신뢰도", "최적성검증", "알고리즘상태", "검증요약", "DQN제외"):
            self.assertIn(sheet, frames)
        self.assertEqual(len(frames["추천결과"]), 8)

    def test_analysis_workbook_has_v2_summary_sheets(self):
        frames = _sheet_frames(
            export_service.analysis_result_excel_bytes(self.pipeline, self.recommendations)
        )
        for sheet in ("VHS중립값", "민감도요약", "추천사유"):
            self.assertIn(sheet, frames)
        reasons = frames["추천사유"]
        self.assertIn("추천 사유", reasons.columns)
        self.assertIn("DQN 안내", reasons.columns)  # benign note kept
        sensitivity = frames["민감도요약"]
        self.assertIn("종합 민감도", sensitivity.columns)
        for frame in (reasons, sensitivity, frames["VHS중립값"]):
            for column in frame.columns:
                lowered = str(column).lower()
                for token in ("reward", "loss", "q_table", "policy_table", "dqn_correction"):
                    self.assertNotIn(token, lowered)

    def test_generated_candidate_sheets_present(self):
        upload_report = {
            "filename": "g.xlsx", "recognized_sheets": ["stores"], "mapped_column_count": 0,
            "column_mappings": [], "recommendation_source": "generated",
            "recommendation_source_label": "V2 생성 후보 사용", "missing_required_count": 0,
            "numeric_failed_total": 0, "blank_removed_total": 0, "date_success": 2, "date_failed": 1,
            "date_columns": ["만료일"], "validation_status": "통과", "analyzable": True,
            "candidate_info": {
                "generated": True, "count": 1, "direct_count": 0, "via_dc_count": 1,
                "route_deferred": 0, "negative_saving_excluded": 0, "duplicate_removed": 0,
                "qty_excluded": 0, "score_components": ["유통기한 긴급도"],
                "candidates": [{
                    "route_id": "V2C001", "product_name": "상품", "source_name": "A", "target_name": "B",
                    "route_type": "VIA_DC", "transfer_qty": 10, "expected_saving": 5000.0,
                    "candidate_score": 81.0, "score_reason": "유통기한 반영", "direct_available": False,
                    "via_dc_available": True, "selected_route_basis": "직접 경로 없음",
                    "days_to_expiry_source": 5, "recommendation_source": "V2 생성 후보", "dqn_status": "미연결",
                }],
            },
        }
        frames = _sheet_frames(export_service.analysis_result_excel_bytes(
            self.pipeline, self.recommendations, upload_report))
        for sheet in ("생성후보", "후보점수", "후보생성요약", "날짜환산"):
            self.assertIn(sheet, frames)
        generated = frames["생성후보"]
        self.assertIn("후보 점수", generated.columns)
        self.assertIn("경로 선택 근거", generated.columns)
        for frame in frames.values():
            for column in frame.columns:
                lowered = str(column).lower()
                for token in ("reward", "loss", "q_table", "policy_table", "dqn_correction"):
                    self.assertNotIn(token, lowered)

    def test_vhs_comparison_sheet_has_uploaded_and_recalculated(self):
        frames = _sheet_frames(
            export_service.analysis_result_excel_bytes(self.pipeline, self.recommendations)
        )
        comp = frames["VHS비교"]
        for column in (
            "route_id", "product_name", "source_name", "target_name",
            "uploaded_vhs", "recalculated_vhs", "difference", "basis",
            "neutral_components", "note",
        ):
            self.assertIn(column, comp.columns)
        self.assertEqual(len(comp), 8)

    def test_injected_dqn_action_and_values_never_reach_export(self):
        injected = [dict(rec) for rec in self.recommendations]
        for rec in injected:
            rec["dqn_action"] = "과거 액션값"
            rec["dqn_correction"] = 99.0
            rec["reward"] = 123456
            rec["loss"] = 7.0
        frame = _sheet_frames(export_service.recommendations_excel_bytes(injected))["추천결과"]
        self.assertEqual(sorted(frame["DQN 상태"].unique().tolist()), ["미연결"])
        for column in frame.columns:
            lowered = str(column).lower()
            for token in ("reward", "loss", "q_table", "policy_table", "replay", "correction"):
                self.assertNotIn(token, lowered)
        csv_text = export_service.recommendations_csv_bytes(injected).decode("utf-8-sig")
        self.assertNotIn("123456", csv_text)
        self.assertNotIn("과거 액션값", csv_text)

    def test_validation_workbook_sheets(self):
        frames = _sheet_frames(
            export_service.validation_report_excel_bytes(self.validation, self.pipeline, self.recommendations)
        )
        for sheet in ("검증개요", "검증메시지", "시트요약", "알고리즘상태", "DQN제외"):
            self.assertIn(sheet, frames)

    def test_upload_quality_sheets_present_when_report_provided(self):
        upload_report = {
            "filename": "f.xlsx", "recognized_sheets": ["stores"],
            "mapped_column_count": 1,
            "column_mappings": [{"sheet": "stores", "original": "점포명", "standard": "node_name"}],
            "recommendation_source": "uploaded", "recommendation_source_label": "업로드 추천 사용",
            "missing_required_count": 0, "numeric_failed_total": 0, "blank_removed_total": 1,
            "validation_status": "통과", "analyzable": True, "candidate_info": {},
        }
        frames = _sheet_frames(export_service.validation_report_excel_bytes(
            self.validation, self.pipeline, self.recommendations, upload_report
        ))
        self.assertIn("업로드품질", frames)
        self.assertIn("컬럼매핑", frames)
        analysis = _sheet_frames(export_service.analysis_result_excel_bytes(
            self.pipeline, self.recommendations, upload_report
        ))
        self.assertIn("업로드품질", analysis)

    def test_reward_and_loss_columns_are_stripped_from_greedy_sheet(self):
        frames = _sheet_frames(
            export_service.analysis_result_excel_bytes(self.pipeline, self.recommendations)
        )
        greedy = frames["Greedy분석"]
        self.assertNotIn("reward", [str(c).lower() for c in greedy.columns])
        self.assertNotIn("loss", [str(c).lower() for c in greedy.columns])

    def test_no_dqn_value_columns_leak_anywhere(self):
        workbooks = [
            export_service.recommendations_excel_bytes(self.recommendations),
            export_service.analysis_result_excel_bytes(self.pipeline, self.recommendations),
            export_service.validation_report_excel_bytes(self.validation, self.pipeline, self.recommendations),
        ]
        for xlsx in workbooks:
            for sheet, frame in _sheet_frames(xlsx).items():
                for column in frame.columns:
                    lowered = str(column).strip().lower()
                    for token in _DQN_VALUE_TOKENS:
                        self.assertNotIn(token, lowered, f"{token} leaked into sheet {sheet}")

    def test_dqn_exclusion_sheet_marks_artifacts_unused(self):
        frames = _sheet_frames(
            export_service.analysis_result_excel_bytes(self.pipeline, self.recommendations)
        )
        dqn = frames["DQN제외"]
        kv = dict(zip(dqn["항목"].astype(str), dqn["값"].astype(str)))
        self.assertEqual(kv.get("DQN 상태"), "미연결")
        self.assertEqual(kv.get("과거 아티팩트 사용"), "아니오")
        self.assertEqual(kv.get("학습 실행"), "아니오")
        self.assertEqual(kv.get("추론 실행"), "아니오")

    def test_empty_inputs_do_not_crash(self):
        self.assertTrue(export_service.recommendations_csv_bytes([]).startswith(b"\xef\xbb\xbf"))
        self.assertGreater(len(export_service.recommendations_excel_bytes([])), 0)
        self.assertGreater(len(export_service.analysis_result_excel_bytes({}, [])), 0)
        self.assertGreater(len(export_service.validation_report_excel_bytes(None, {}, [])), 0)


if __name__ == "__main__":
    unittest.main()
