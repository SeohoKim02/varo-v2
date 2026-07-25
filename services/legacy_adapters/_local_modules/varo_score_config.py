"""
varo_score_config.py
─────────────────────
Varo Hybrid Score 가중치·등급 기준 중앙 관리 모듈.

- 기존 varo_hybrid_score.py / varo_score_v2.py 계산 로직은 유지.
- 이 모듈은 가중치 관리, 정규화 유틸, 등급 기준, config 시트 연동을 담당.
"""

import math
import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. VHS v2 기본 가중치 (scenario_detector가 상황별로 조정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULT_VHS_WEIGHTS: dict = {
    "재고위험":      0.25,
    "판매가능성":    0.20,
    "점포이동적합":  0.15,
    "비용절감":      0.15,
    "폐기회피이익":  0.10,
    "실행가능성":    0.10,
    "이력보정":      0.05,
}
# 합계 검증용
_VHS_WEIGHT_SUM = sum(DEFAULT_VHS_WEIGHTS.values())  # 1.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 추천 등급 기준 (VHS v2 기준)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADE_THRESHOLDS: dict = {
    "최적": 80,    # 80점 이상
    "권장": 65,    # 65점 이상
    "검토": 50,    # 50점 이상
    "보류":  0,    # 50점 미만
}

def assign_recommendation_grade(score) -> str:
    """
    VHS 점수 → 추천 등급.
    기존 varo_score_v2.py 등급(최적/권장/검토)에 '보류' 추가.
    """
    try:
        s = float(score)
        if math.isnan(s): return "검토"
    except (TypeError, ValueError):
        return "검토"
    if s >= GRADE_THRESHOLDS["최적"]: return "최적"
    if s >= GRADE_THRESHOLDS["권장"]: return "권장"
    if s >= GRADE_THRESHOLDS["검토"]: return "검토"
    return "보류"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 점수 구성 기준표 (UI 표시용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORE_CRITERIA: list = [
    {
        "항목":     "재고 위험",
        "가중치":   "25%",
        "반영 기준":"과잉·악성재고 가능성 (폐기위험·회전율·ABC·노후화)",
        "점수 방향":"높을수록 위험",
        "관련 알고리즘":"disposal_risk, turnover, abc, aging",
        "계산 가능": True,
    },
    {
        "항목":     "판매 가능성",
        "가중치":   "20%",
        "반영 기준":"수요 예측·판매 추세·Newsvendor",
        "점수 방향":"높을수록 판매 기대",
        "관련 알고리즘":"demand_forecast, trend, newsvendor",
        "계산 가능": True,
    },
    {
        "항목":     "점포 이동 적합",
        "가중치":   "15%",
        "반영 기준":"수신 점포 매칭·서비스 수준·우선순위",
        "점수 방향":"높을수록 이동 적합",
        "관련 알고리즘":"match, service_level, priority_queue, queue_capacity",
        "계산 가능": True,
    },
    {
        "항목":     "비용 절감",
        "가중치":   "15%",
        "반영 기준":"이동비용·LP 수송 비용·EOQ 이탈도",
        "점수 방향":"낮은 비용 = 높은 점수",
        "관련 알고리즘":"heuristic(비용), transport_lp, eoq",
        "계산 가능": True,
    },
    {
        "항목":     "폐기 회피 이익",
        "가중치":   "10%",
        "반영 기준":"폐기 회피 가능성·할인 민감도",
        "점수 방향":"높을수록 폐기 회피 가능",
        "관련 알고리즘":"disposal_avoidance, discount_sensitivity",
        "계산 가능": True,
    },
    {
        "항목":     "실행 가능성",
        "가중치":   "10%",
        "반영 기준":"병목·점포 처리 능력·카테고리 균형",
        "점수 방향":"높을수록 실행 용이",
        "관련 알고리즘":"bottleneck, store_capacity, category_balance",
        "계산 가능": True,
    },
    {
        "항목":     "이력 보정",
        "가중치":   "5%",
        "반영 기준":"DQN reward 기반 ±8점 보정",
        "점수 방향":"누적 데이터 기반 보정",
        "관련 알고리즘":"reward (RL 학습 결과)",
        "계산 가능": True,
    },
]

def get_score_criteria_df() -> pd.DataFrame:
    """점수 기준표 DataFrame 반환."""
    cols = ["항목", "가중치", "반영 기준", "점수 방향", "관련 알고리즘"]
    return pd.DataFrame([{c: r[c] for c in cols} for r in SCORE_CRITERIA])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 정규화 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def safe_number(value, default: float = 0.0) -> float:
    """NaN / None / inf → default 반환."""
    try:
        f = float(value)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default

def normalize_score(value, min_value: float = 0.0, max_value: float = 100.0,
                    inverse: bool = False, default: float = 50.0) -> float:
    """
    [min_value, max_value] → [0, 100] 정규화.
    inverse=True: 낮을수록 좋은 값(비용 등)은 역방향 처리.
    """
    v = safe_number(value, default)
    mn = safe_number(min_value, 0.0)
    mx = safe_number(max_value, 100.0)
    if mx <= mn:
        return default
    norm = (v - mn) / (mx - mn) * 100.0
    norm = max(0.0, min(100.0, norm))
    return round(100.0 - norm if inverse else norm, 2)

def normalize_weights(weights: dict, default: dict = None) -> dict:
    """
    가중치 dict → 합계 1.0으로 정규화.
    합계 0 또는 잘못된 값 → default 반환.
    """
    default = default or DEFAULT_VHS_WEIGHTS
    try:
        cleaned = {}
        for k, v in weights.items():
            sv = safe_number(v, -1)
            if sv >= 0:
                cleaned[k] = sv
        total = sum(cleaned.values())
        if total <= 0:
            return dict(default)
        return {k: round(v / total, 6) for k, v in cleaned.items()}
    except Exception:
        return dict(default)

def calculate_weighted_score(score_components: dict, weights: dict,
                              default_score: float = 50.0) -> float:
    """
    항목별 점수 dict + 가중치 dict → 가중 합산 점수.
    누락 항목은 default_score 사용.
    """
    total = 0.0
    w_sum = 0.0
    for k, w in weights.items():
        if k == "이력보정":
            continue  # 이력 보정은 별도 처리
        score = safe_number(score_components.get(k, default_score), default_score)
        w_val = safe_number(w, 0.0)
        total += score * w_val
        w_sum += w_val
    if w_sum <= 0:
        return default_score
    return round(total / w_sum, 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. config 시트 연동
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG_WEIGHT_KEYS = list(DEFAULT_VHS_WEIGHTS.keys())

def load_weights_from_config(config_df: pd.DataFrame = None) -> dict:
    """
    config DataFrame (key|value 구조)에서 가중치 읽기.
    없거나 유효하지 않으면 DEFAULT_VHS_WEIGHTS 반환.

    config 엑셀 시트 예시:
        key                  | value
        재고위험              | 0.25
        판매가능성            | 0.20
        점포이동적합          | 0.15
        비용절감              | 0.15
        폐기회피이익          | 0.10
        실행가능성            | 0.10
        이력보정              | 0.05
    """
    if config_df is None or config_df.empty:
        return dict(DEFAULT_VHS_WEIGHTS)

    try:
        # key/value 컬럼 자동 감지
        col_key = next((c for c in config_df.columns
                        if str(c).lower() in ("key","항목","가중치항목","name")), None)
        col_val = next((c for c in config_df.columns
                        if str(c).lower() in ("value","값","가중치","weight")), None)
        if col_key is None or col_val is None:
            return dict(DEFAULT_VHS_WEIGHTS)

        raw = {}
        for _, row in config_df.iterrows():
            k = str(row[col_key]).strip()
            if k in CONFIG_WEIGHT_KEYS:
                raw[k] = safe_number(row[col_val], -1)

        # 하나도 없으면 기본값
        valid = {k: v for k, v in raw.items() if v >= 0}
        if not valid:
            return dict(DEFAULT_VHS_WEIGHTS)

        # 누락 키는 DEFAULT로 채움
        merged = dict(DEFAULT_VHS_WEIGHTS)
        merged.update(valid)
        return normalize_weights(merged, DEFAULT_VHS_WEIGHTS)

    except Exception:
        return dict(DEFAULT_VHS_WEIGHTS)


def load_config_sheet(excel_sheets: dict) -> dict:
    """
    업로드된 엑셀 시트 dict에서 'config' 시트를 찾아 가중치 로드.
    없으면 기본값 반환.
    """
    cfg_sheet = None
    for name in excel_sheets:
        if str(name).lower() in ("config", "설정", "가중치", "weight", "weights"):
            cfg_sheet = excel_sheets[name]
            break
    return load_weights_from_config(cfg_sheet)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 검증 정보 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_weight_diagnostics(weights: dict, result_df: pd.DataFrame = None) -> dict:
    """
    현재 사용 중인 가중치 진단 정보 반환.
    접힌 영역에서 표시하기 위한 검증 데이터.
    """
    w_sum = round(sum(safe_number(v, 0) for v in weights.values()), 6)
    is_default = (weights == DEFAULT_VHS_WEIGHTS)
    is_normalized = abs(w_sum - 1.0) < 0.001

    fallback_cols = 0
    computable_count = 0
    if result_df is not None and not result_df.empty:
        score_cols = [col for _, algo_list in [
            ("재고위험",     ["disposal_risk_score","turnover_score","abc_score","aging_score"]),
            ("판매가능성",   ["demand_forecast_score","trend_score","newsvendor_score"]),
            ("점포이동적합", ["match_score","service_level_score","priority_queue_score"]),
            ("비용절감",     ["heuristic_score","transport_lp_score","eoq_score"]),
            ("폐기회피이익", ["disposal_avoidance_score","discount_sensitivity_score"]),
            ("실행가능성",   ["bottleneck_score","store_capacity_score","category_balance_score"]),
        ] for col in algo_list]
        fallback_cols = sum(1 for c in score_cols if c not in result_df.columns)
        computable_count = len(result_df)

    return {
        "가중치_합계":        w_sum,
        "정규화_여부":        "✅" if is_normalized else f"⚠️ {w_sum:.4f}",
        "기본값_사용":        "기본값" if is_default else "사용자 정의",
        "추천_등급_기준":     {k: f"{v}점 이상" if v > 0 else f"{v}점 미만"
                              for k, v in GRADE_THRESHOLDS.items()},
        "계산_가능_후보_수":  computable_count,
        "fallback_컬럼_수":  fallback_cols,
        "가중치_상세":        {k: f"{v:.1%}" for k, v in weights.items()},
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 추가: Hybrid Score 수학적 정의 / 수식 구성 요소
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize_inverse_score(value, min_value: float = 0.0, max_value: float = 100.0,
                             default: float = 50.0) -> float:
    """
    낮을수록 좋은 값(비용 등)에 대한 역방향 정규화.
    normalize_score(value, min, max, inverse=True)의 alias.
    """
    return normalize_score(value, min_value, max_value, inverse=True, default=default)


def get_hybrid_score_formula_components() -> list:
    """
    VHS(Varo Hybrid Score) 수식 구성 요소 목록 반환.
    각 항목: {symbol, name_ko, name_en, meaning, direction, weight_key,
              normalization, col_hint, implemented}

    S(i) = Σ wk · fk(i),  Σwk = 1,  0 ≤ fk(i) ≤ 1
    """
    return [
        {
            "symbol":       "f₁",
            "name_ko":      "재고 위험도",
            "name_en":      "InventoryRisk",
            "meaning":      "과잉재고 · 악성재고 위험 (회전율, ABC, 노후화, 폐기 위험 포함)",
            "direction":    "높을수록 처리 긴급",
            "weight_key":   "재고위험",
            "weight_default": 0.25,
            "normalization":"0 → 1 (정방향)",
            "col_hint":     "vhs2_group_재고위험",
            "implemented":  True,
        },
        {
            "symbol":       "f₂",
            "name_ko":      "판매 가능성",
            "name_en":      "SalesPotential",
            "meaning":      "수요 예측 · 판매 추세 · Newsvendor 재고 최적화 반영",
            "direction":    "높을수록 판매 기대치 상승",
            "weight_key":   "판매가능성",
            "weight_default": 0.20,
            "normalization":"0 → 1 (정방향)",
            "col_hint":     "vhs2_group_판매가능성",
            "implemented":  True,
        },
        {
            "symbol":       "f₃",
            "name_ko":      "점포 이동 적합",
            "name_en":      "TransferFit",
            "meaning":      "수신 점포 수요 매칭 · 서비스 수준 · 배분 우선순위",
            "direction":    "높을수록 이동 적합",
            "weight_key":   "점포이동적합",
            "weight_default": 0.15,
            "normalization":"0 → 1 (정방향)",
            "col_hint":     "vhs2_group_점포이동적합",
            "implemented":  True,
        },
        {
            "symbol":       "f₄",
            "name_ko":      "비용 절감",
            "name_en":      "CostEfficiency",
            "meaning":      "이동비 · 할인 손실 · EOQ 이탈 비용 반영 (낮을수록 유리)",
            "direction":    "비용 낮을수록 점수 높음 (inverse)",
            "weight_key":   "비용절감",
            "weight_default": 0.15,
            "normalization":"0 → 1 (역방향, inverse)",
            "col_hint":     "vhs2_group_비용절감",
            "implemented":  True,
        },
        {
            "symbol":       "f₅",
            "name_ko":      "폐기 회피 이익",
            "name_en":      "DisposalAvoidance",
            "meaning":      "폐기 가능성 감소 · 할인 민감도 기반 잔여 재고 처리 기대",
            "direction":    "높을수록 폐기 위험 감소",
            "weight_key":   "폐기회피이익",
            "weight_default": 0.10,
            "normalization":"0 → 1 (정방향)",
            "col_hint":     "vhs2_group_폐기회피이익",
            "implemented":  True,
        },
        {
            "symbol":       "f₆",
            "name_ko":      "실행 가능성",
            "name_en":      "Feasibility",
            "meaning":      "병목 해소 · 점포 처리 용량 · 카테고리 균형 적합도",
            "direction":    "높을수록 실행 용이",
            "weight_key":   "실행가능성",
            "weight_default": 0.10,
            "normalization":"0 → 1 (정방향)",
            "col_hint":     "vhs2_group_실행가능성",
            "implemented":  True,
        },
        {
            "symbol":       "f₇",
            "name_ko":      "이력 보정",
            "name_en":      "HistoryCorrection",
            "meaning":      "DQN reward 기반 누적 이력으로 ±점수 보정",
            "direction":    "누적 이력 기반 보정 (양수=상승, 음수=하락)",
            "weight_key":   "이력보정",
            "weight_default": 0.05,
            "normalization":"보정값 (±8점 범위)",
            "col_hint":     "vhs2_history_correction",
            "implemented":  True,
        },
    ]


def get_hybrid_score_math_definition() -> str:
    """
    VHS 수식의 LaTeX/텍스트 표현 반환.
    앱 표시용 — 접힌 영역에서만 사용.
    """
    return (
        "S(i) = w₁·f₁(i) + w₂·f₂(i) + w₃·f₃(i) + w₄·f₄(i)\n"
        "     + w₅·f₅(i) + w₆·f₆(i) + w₇·f₇(i)\n"
        "\n"
        "조건:\n"
        "  • 0 ≤ fk(i) ≤ 1   (각 항목 정규화)\n"
        "  • wk ≥ 0           (가중치 비음수)\n"
        "  • Σ wk = 1         (가중치 합계)\n"
        "  • S(i) ∈ [0, 100]  (최종 점수 범위)\n"
        "  • 비용 항목(f₄): inverse 정규화 적용\n"
        "  • 이력 보정(f₇): DQN reward 기반 ±8점 보정\n"
        "\n"
        "추천 등급:\n"
        "  80 ≤ S < 100 → 최적   65 ≤ S < 80 → 권장\n"
        "  50 ≤ S <  65 → 검토   S < 50      → 보류"
    )


def build_hybrid_score_explanation_table() -> pd.DataFrame:
    """
    수식 구성 요소 상세 표 (앱 접힌 영역 표시용).
    컬럼: 기호, 항목, 의미, 점수 방향, 기본 가중치, 정규화 방식
    """
    components = get_hybrid_score_formula_components()
    rows = []
    for c in components:
        rows.append({
            "기호":        c["symbol"],
            "항목":        c["name_ko"],
            "의미":        c["meaning"],
            "점수 방향":   c["direction"],
            "기본 가중치": f'{c["weight_default"]:.0%}',
            "정규화 방식": c["normalization"],
        })
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pareto dominance 일관성 정의 · 증명 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_pareto_dominance_definition() -> str:
    """
    Varo 기준 Pareto dominance 정의 텍스트.
    접힌 영역에서만 표시.
    """
    return (
        "후보 A가 후보 B를 Pareto dominate 하는 조건:\n"
        "\n"
        "  (1) 모든 항목 k에 대해  fk(A) ≥ fk(B)\n"
        "  (2) 적어도 하나의 항목 m에서  fm(A) > fm(B)\n"
        "\n"
        "이때 wk ≥ 0 이고 wm > 0 이면:\n"
        "\n"
        "  S(A) - S(B)\n"
        "  = Σ wk · [fk(A) - fk(B)]\n"
        "  ≥ wm · [fm(A) - fm(B)]\n"
        "  > 0\n"
        "\n"
        "따라서  S(A) > S(B)  성립."
    )


def get_pareto_dominance_proof_text() -> str:
    """
    증명 스케치 전문 텍스트.
    """
    return (
        "【증명 요약】\n"
        "\n"
        "S(A) - S(B)\n"
        "  = Σk wk · fk(A)  -  Σk wk · fk(B)\n"
        "  = Σk wk · [fk(A) - fk(B)]          ... (선형성)\n"
        "\n"
        "Pareto dominance 조건에 의해:\n"
        "  • 모든 k:  fk(A) - fk(B) ≥ 0\n"
        "  • 항목 m:  fm(A) - fm(B) > 0\n"
        "\n"
        "가중치 조건 wk ≥ 0, wm > 0 에 의해:\n"
        "  Σk wk · [fk(A) - fk(B)]\n"
        "  ≥ wm · [fm(A) - fm(B)]  > 0\n"
        "\n"
        "∴  S(A) > S(B)                        □"
    )


def build_pareto_condition_table() -> pd.DataFrame:
    """
    Pareto dominance 성립 조건 표.
    """
    rows = [
        {
            "조건":           "fk(A) ≥ fk(B)  (모든 k)",
            "의미":           "모든 항목에서 A가 B 이상",
            "Varo 적용 방식": "정규화된 점수 항목 직접 비교",
        },
        {
            "조건":           "fm(A) > fm(B)  (하나 이상)",
            "의미":           "적어도 한 항목에서 A가 B보다 우위",
            "Varo 적용 방식": "수요 높음·비용 낮음·긴급도 높음 등 중 하나 이상 우위",
        },
        {
            "조건":           "wk ≥ 0  (모든 k)",
            "의미":           "음수 가중치 없음",
            "Varo 적용 방식": "normalize_weights() 로 보장",
        },
        {
            "조건":           "wm > 0  (개선 항목)",
            "의미":           "개선된 항목의 가중치가 양수",
            "Varo 적용 방식": "DEFAULT_VHS_WEIGHTS 모든 항목 > 0",
        },
        {
            "조건":           "Σ wk = 1",
            "의미":           "가중치 합 = 1",
            "Varo 적용 방식": "normalize_weights() 자동 정규화",
        },
    ]
    return pd.DataFrame(rows)


def build_score_direction_table() -> pd.DataFrame:
    """
    각 점수 항목의 방향 통일 확인 표.
    Pareto 비교를 위해 모든 항목이 '높을수록 좋음' 방향으로 통일.
    """
    rows = [
        {
            "항목":              "재고 위험도  (f₁)",
            "원래 의미":         "과잉·악성재고 위험",
            "점수 방향":         "높을수록 긴급 → 처리 우선",
            "Pareto 비교 가능": "✅",
        },
        {
            "항목":              "판매 가능성  (f₂)",
            "원래 의미":         "수요 예측·판매 추세",
            "점수 방향":         "높을수록 판매 기대 → 이동 우선",
            "Pareto 비교 가능": "✅",
        },
        {
            "항목":              "점포 이동 적합  (f₃)",
            "원래 의미":         "받는 점포 매칭·배분 적합도",
            "점수 방향":         "높을수록 이동 적합",
            "Pareto 비교 가능": "✅",
        },
        {
            "항목":              "비용 절감  (f₄)",
            "원래 의미":         "이동비·할인손실·처리비",
            "점수 방향":         "비용 낮을수록 높은 점수 (inverse 정규화)",
            "Pareto 비교 가능": "✅  (inverse 변환 후)",
        },
        {
            "항목":              "폐기 회피 이익  (f₅)",
            "원래 의미":         "폐기 가능성 감소·할인 민감도",
            "점수 방향":         "높을수록 폐기 감소 기대 → 처리 우선",
            "Pareto 비교 가능": "✅",
        },
        {
            "항목":              "실행 가능성  (f₆)",
            "원래 의미":         "병목·처리 용량·카테고리 균형",
            "점수 방향":         "높을수록 실행 용이",
            "Pareto 비교 가능": "✅",
        },
        {
            "항목":              "이력 보정  (f₇)",
            "원래 의미":         "DQN reward 기반 보정",
            "점수 방향":         "보정값 양수 시 점수 상승 (누적 이력 반영)",
            "Pareto 비교 가능": "✅  (보정 방향 통일)",
        },
    ]
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# argmax 최적성 정의 · 증명 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_argmax_optimality_definition() -> str:
    """
    Varo argmax 선택 기준 정의 텍스트.
    접힌 영역에서만 표시.
    """
    return (
        "【후보 집합 및 선택 기준】\n"
        "\n"
        "  R  : 가능한 모든 재고 처리 후보 집합\n"
        "  i  : 개별 추천 후보 (상품 × 점포 × 전략 조합)\n"
        "  S(i): 후보 i의 Varo Hybrid Score\n"
        "\n"
        "Varo 최종 선택:\n"
        "  i* = argmax S(i),  i ∈ R\n"
        "\n"
        "argmax 정의에 의해 모든 j ∈ R에 대해:\n"
        "  S(i*) ≥ S(j)\n"
        "\n"
        "TOP N 추천:\n"
        "  TOP N = sort(R, key=S(i), desc=True)[0:N]\n"
        "\n"
        "∴ Varo는 정의된 Hybrid Score 기준에서\n"
        "  후보 집합 내 최고 점수 후보를 우선 추천한다."
    )


def get_argmax_optimality_proof_text() -> str:
    """증명 스케치 전문 텍스트."""
    return (
        "【증명 요약】\n"
        "\n"
        "후보 집합 R과 점수 함수 S(i)가 주어졌다고 하자.\n"
        "\n"
        "Varo는 다음 후보를 선택한다:\n"
        "  i* = argmax S(i),  i ∈ R\n"
        "\n"
        "argmax 정의에 의해 모든 j ∈ R에 대해\n"
        "  S(i*) ≥ S(j)\n"
        "가 성립한다.\n"
        "\n"
        "따라서 i*는 후보 집합 R 안에서\n"
        "Hybrid Score 기준으로 가장 높은 점수를 갖는 후보이다.\n"
        "\n"
        "※ 이는 현실 세계의 절대 최적해가 아니라,\n"
        "  정의된 Hybrid Score 기준 안에서의 최적 후보 선택이다.\n"
        "\n"
        "∴ Varo는 정의된 기준에서 후보 집합 내 최적 후보를 선택한다.  □"
    )


def build_argmax_selection_table() -> pd.DataFrame:
    """argmax 선택 기준 표."""
    rows = [
        {
            "기호":  "R",
            "의미":  "가능한 추천 후보 집합",
            "Varo 적용 방식": "상품 × 점포 × 전략 조합으로 생성된 후보 전체",
        },
        {
            "기호":  "i",
            "의미":  "개별 추천 후보",
            "Varo 적용 방식": "하나의 재고 처리 후보 (상품·점포·수량·전략 포함)",
        },
        {
            "기호":  "S(i)",
            "의미":  "후보 i의 Hybrid Score",
            "Varo 적용 방식": "정규화 점수와 가중치로 0~100 범위로 계산",
        },
        {
            "기호":  "i*",
            "의미":  "최종 선택 후보",
            "Varo 적용 방식": "S(i)가 가장 높은 후보 (argmax)",
        },
        {
            "기호":  "TOP N",
            "의미":  "상위 추천 후보 N개",
            "Varo 적용 방식": "S(i) 기준 내림차순 정렬 상위 N개",
        },
    ]
    return pd.DataFrame(rows)


def build_argmax_condition_table() -> pd.DataFrame:
    """argmax 최적성 성립 조건 표."""
    rows = [
        {
            "조건":  "후보 집합 R 정의",
            "의미":  "비교 가능한 후보가 생성되어야 함",
            "상태":  "✅ 적용",
        },
        {
            "조건":  "점수 함수 S(i) 정의",
            "의미":  "모든 후보에 동일한 점수 기준 적용",
            "상태":  "✅ 적용",
        },
        {
            "조건":  "점수 방향 통일",
            "의미":  "높은 점수가 더 좋은 후보",
            "상태":  "✅ inverse 정규화로 보장",
        },
        {
            "조건":  "가중치 정규화",
            "의미":  "가중치 합 = 1",
            "상태":  "✅ normalize_weights() 적용",
        },
        {
            "조건":  "정렬 기준 적용",
            "의미":  "S(i) 기준 내림차순 정렬",
            "상태":  "✅ 적용",
        },
        {
            "조건":  "동점 처리",
            "의미":  "동일 점수 시 기존 정렬 순서",
            "상태":  "✅ DataFrame 순서 유지",
        },
    ]
    return pd.DataFrame(rows)
