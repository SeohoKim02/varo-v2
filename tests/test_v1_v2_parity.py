"""Guards the V1 → V2 feature parity: the parity doc must cover every compared
feature, and the features flagged as 보완함(newly added) must actually be wired
into the V2 pages (no placeholder buttons)."""
from __future__ import annotations

import unittest
from pathlib import Path

from tests.streamlit_log_silencer import quiet_streamlit_test_logs

quiet_streamlit_test_logs()

ROOT = Path(__file__).resolve().parents[1]

# The features the parity task requires V2 to match or beat.
PARITY_FEATURES = [
    "데이터 업로드", "기본 샘플", "데이터 검증", "악성재고", "Varo Hybrid Score",
    "자동 가중치", "추천 생성", "추천 후보 필터", "최종 추천", "Greedy 비교",
    "DQN 학습", "DQN 결과 저장", "DQN 안정성", "DQN action 비교", "Pareto",
    "민감도", "추천 신뢰도", "최적성 검증", "비용 비교", "직접 이동", "DC 경유",
    "지도", "Kakao", "이동 경로", "시뮬레이션", "운영 KPI", "추천 Top5",
    "현재 실행 경로", "검증 리포트", "Excel 다운로드", "분석 결과 다운로드",
    "학습 결과 다운로드", "원본 데이터 확인", "페이지 이동", "상태 초기화",
    "새 데이터 적용", "오류·경고",
]


class ParityDocTests(unittest.TestCase):
    def setUp(self):
        self.doc = (ROOT / "V1_V2_PARITY.md").read_text(encoding="utf-8")

    def test_parity_doc_exists_and_covers_every_feature(self):
        for feature in PARITY_FEATURES:
            self.assertIn(feature, self.doc, f"parity doc must cover: {feature}")

    def test_parity_doc_records_the_backfilled_items(self):
        # Newly added items must be marked so removals cannot silently pass.
        self.assertIn("보완함", self.doc)
        self.assertIn("DQN 학습 결과 다운로드", self.doc)


class WiredFeatureTests(unittest.TestCase):
    def test_dqn_training_result_download_button_is_wired(self):
        source = (ROOT / "pages" / "validation.py").read_text(encoding="utf-8")
        self.assertIn('"학습 결과 다운로드"', source)
        self.assertIn("dl_dqn_training_result", source)

    def test_best_recommendation_separates_strategy_fields(self):
        source = (ROOT / "pages" / "recommendations.py").read_text(encoding="utf-8")
        for label in ("이동 경로 방식", "Greedy 전략", "최종 처리 전략", "추천 등급"):
            self.assertIn(label, source)

    def test_pages_expose_all_five_menu_renderers(self):
        import router
        self.assertEqual(
            set(router._PAGE_RENDERERS),
            {"운영 현황", "추천 실행", "경로 상세", "분석 및 검증", "데이터 관리"},
        )


if __name__ == "__main__":
    unittest.main()
