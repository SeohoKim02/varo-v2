"""
varo_confidence.py
───────────────────
추천 신뢰도(Confidence) 계산 모듈.

- 기존 vhs_confidence.py(데이터 품질 신뢰도)와는 별도 레이어.
- 최종 추천 후보별 0~100점 신뢰도 + 높음/보통/낮음 등급 + 근거 요약 생성.
- 기존 Varo Hybrid Score, 최종 추천 결과를 수정하지 않음.
"""

import math
import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 기준값 (한 곳에서 관리)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_default_confidence_rules() -> dict:
    return {
        "high_score_threshold":   80,
        "medium_score_threshold": 60,
        "min_move_qty":            1,
        "dqn_normal_bonus":       10,
        "dqn_warning_bonus":       3,
        "model_agreement_bonus":  10,
        "valid_cost_bonus":        8,
        "valid_qty_bonus":         8,
        "high_vhs_bonus":         15,
        "medium_vhs_bonus":        8,
        "data_quality_bonus":      7,
        "missing_data_penalty":   10,
        "zero_qty_penalty":       20,
        "base_score":             50,    # 기본 점수
    }

_RULES = get_default_confidence_rules()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _safe_num(value, default: float = 0.0) -> float:
    try:
        f = float(value)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default

def is_valid_positive_number(value) -> bool:
    try:
        f = float(value)
        return not (math.isnan(f) or math.isinf(f)) and f > 0
    except (TypeError, ValueError):
        return False

def _coerce_series(df: pd.DataFrame, col: str, default=None):
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 단일 행 신뢰도 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calculate_recommendation_confidence(row: "pd.Series",
                                         dqn_status: str = None) -> dict:
    """
    추천 후보 단일 행(row) → {score, level, reason}.
    dqn_status: "정상"/"주의"/"제외"/"데이터 없음" (없으면 무시)
    """
    rules   = _RULES
    score   = float(rules["base_score"])
    reasons = []
    checks  = {}

    # ── VHS2 / Hybrid Score ──────────────────────────────
    vhs = _safe_num(row.get("vhs2") or row.get("heuristic_score"), 0)
    if vhs >= rules["high_score_threshold"]:
        score += rules["high_vhs_bonus"]
        reasons.append("점수 높음")
        checks["vhs_high"] = True
    elif vhs >= rules["medium_score_threshold"]:
        score += rules["medium_vhs_bonus"]
        checks["vhs_medium"] = True
    else:
        checks["vhs_low"] = True

    # ── 추천 등급 ─────────────────────────────────────────
    grade = str(row.get("vhs2_grade") or row.get("heuristic_grade") or "")
    if grade in ("최적", "권장"):
        score += 5
        reasons.append("등급 우수")

    # ── 추천 수량 ─────────────────────────────────────────
    qty_cols = ["suggested_qty","move_qty","recommended_qty","transfer_qty","suggested_transfer_qty"]
    qty = 0.0
    for qc in qty_cols:
        v = row.get(qc)
        if is_valid_positive_number(v):
            qty = float(v); break
    if qty >= rules["min_move_qty"]:
        score += rules["valid_qty_bonus"]
        reasons.append("수량 유효")
        checks["qty_valid"] = True
    else:
        score -= rules["zero_qty_penalty"]
        checks["qty_zero"] = True

    # ── 예상 비용 ─────────────────────────────────────────
    cost = _safe_num(row.get("estimated_cost"), -1)
    if cost > 0:
        score += rules["valid_cost_bonus"]
        checks["cost_valid"] = True
    else:
        checks["cost_invalid"] = True

    # ── 데이터 품질 (기존 vhs2_confidence 활용) ──────────
    existing_conf = str(row.get("vhs2_confidence", ""))
    if existing_conf == "HIGH":
        score += rules["data_quality_bonus"]
        reasons.append("데이터 충실")
    elif existing_conf == "LOW":
        score -= rules["missing_data_penalty"]

    # ── DQN 상태 반영 ─────────────────────────────────────
    # row에 dqn_status 컬럼이 있으면 우선, 아니면 파라미터 사용
    dqn_st = str(row.get("dqn_status", "") or dqn_status or "")
    if "정상" in dqn_st:
        score += rules["dqn_normal_bonus"]
        checks["dqn_normal"] = True
    elif "주의" in dqn_st:
        score += rules["dqn_warning_bonus"]
        checks["dqn_warning"] = True
    # 제외/데이터없음은 가점만 없음 (감점 X)

    # ── 모델 일치 여부 ────────────────────────────────────
    agreement = str(row.get("agreement_status", "") or "")
    if "일치" in agreement and "불일치" not in agreement:
        score += rules["model_agreement_bonus"]
        reasons.append("모델 일치")
        checks["model_agree"] = True

    # ── 수요 적합도 반영 (varo_demand 연동) ────────────────
    demand_st = str(row.get("demand_status", "") or "")
    if "높음" in demand_st:
        score += 8
        reasons.append("수요 높음")
    elif "보통" in demand_st:
        score += 3
    elif "낮음" in demand_st:
        score -= 5
        reasons.append("수요 낮음")
    # "데이터 없음" → 가점/감점 없음

    # ── 유통기한 긴급도 (expiry_days 있을 때) ─────────────
    expiry = _safe_num(row.get("expiry_days") or row.get("days_to_expiry"), -1)
    if 0 < expiry <= 5:
        # 매우 임박 → 신속 처리 권장, 신뢰도에는 neutral
        pass

    # ── 수요 적합도 반영 ──────────────────────────────────
    demand_st = str(row.get("demand_status", "") or "")
    if demand_st == "높음":
        score += 8
        reasons.append("수요 높음")
    elif demand_st == "보통":
        score += 3
    elif demand_st == "낮음":
        score -= 5
    # 데이터 없음은 가점/감점 없음

    # ── 프로모션 현실성 반영 ──────────────────────────────
    promo_st = str(row.get("promotion_status", "") or "")
    if promo_st == "유리":
        score += 7
        reasons.append("프로모션 효과 양호")
    elif promo_st == "보통":
        score += 2
    elif promo_st == "비추천":
        score -= 5
        reasons.append("할인 손실 부담")
    # 데이터 부족은 가점/감점 없음
    # 순효과 음수면 추가 감점
    net_b = _safe_num(row.get("promotion_net_benefit"), 1)
    if promo_st not in ("", "데이터 부족") and net_b < 0:
        score -= 3

    # ── 클램핑 ────────────────────────────────────────────
    score = max(0.0, min(100.0, score))
    score = round(score, 1)

    # ── 요약 근거 (최대 3개) ─────────────────────────────
    reason_str = " · ".join(reasons[:3]) if reasons else "-"

    return {"score": score, "level": assign_confidence_level(score),
            "reason": reason_str, "checks": checks}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 등급 부여
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def assign_confidence_level(score) -> str:
    s = _safe_num(score, 0)
    if s >= _RULES["high_score_threshold"]:   return "높음"
    if s >= _RULES["medium_score_threshold"]: return "보통"
    return "낮음"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. DataFrame 전체에 신뢰도 컬럼 추가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def add_confidence_columns(df: pd.DataFrame,
                           dqn_status: str = None) -> pd.DataFrame:
    """
    최종 추천 DataFrame에 confidence_score / confidence_level / confidence_reason 추가.
    기존 컬럼을 수정하지 않음.
    """
    if df is None or df.empty:
        return df

    # DQN 상태 row-wise 감지 우선
    has_dqn_col = "dqn_status" in df.columns
    out = df.copy()

    scores, levels, reasons = [], [], []
    for _, row in out.iterrows():
        dqn_st = str(row.get("dqn_status", "") or dqn_status or "") if has_dqn_col \
                 else (dqn_status or "")
        try:
            res = calculate_recommendation_confidence(row, dqn_st)
        except Exception:
            res = {"score": 50.0, "level": "보통", "reason": "-"}
        scores.append(res["score"])
        levels.append(res["level"])
        reasons.append(res["reason"])

    out["confidence_score"]  = scores
    out["confidence_level"]  = levels
    out["confidence_reason"] = reasons
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 신뢰도 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_confidence_summary(df: pd.DataFrame) -> dict:
    """confidence_level 집계."""
    if df is None or df.empty or "confidence_level" not in df.columns:
        return {}
    vc = df["confidence_level"].value_counts().to_dict()
    cs = pd.to_numeric(df.get("confidence_score", pd.Series()), errors="coerce")
    return {
        "높음":  int(vc.get("높음", 0)),
        "보통":  int(vc.get("보통", 0)),
        "낮음":  int(vc.get("낮음", 0)),
        "평균":  round(float(cs.mean()), 1) if not cs.empty else 0.0,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 기준표 (UI 표시용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_confidence_criteria_table() -> pd.DataFrame:
    rows = [
        {"기준":"Hybrid Score",  "반영 방식":"점수 반영",   "조건":"80↑ +15 / 60↑ +8",     "영향":"상승"},
        {"기준":"추천 등급",     "반영 방식":"등급 확인",   "조건":"최적·권장 +5",           "영향":"상승"},
        {"기준":"추천 수량",     "반영 방식":"유효성 확인", "조건":"1개 이상 +8 / 0 -20",    "영향":"상승/하락"},
        {"기준":"예상 비용",     "반영 방식":"유효성 확인", "조건":"정상값 +8",              "영향":"상승"},
        {"기준":"데이터 품질",   "반영 방식":"기존 신뢰도", "조건":"HIGH +7 / LOW -10",      "영향":"상승/하락"},
        {"기준":"DQN 상태",      "반영 방식":"보조 반영",   "조건":"정상 +10 / 주의 +3",     "영향":"보조 상승"},
        {"기준":"모델 일치도",   "반영 방식":"비교 반영",   "조건":"추천 일치 +10",           "영향":"상승"},
    ]
    return pd.DataFrame(rows)
