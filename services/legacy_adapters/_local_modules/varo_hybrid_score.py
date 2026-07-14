"""
VARO Hybrid Score (VHS) — 악성재고 통합 의사결정 알고리즘
═══════════════════════════════════════════════════════════════

  구조
  ────
  [10개 알고리즘 점수]
       ↓  상황 감지 (Situation Detector)
  [가중치 자동 조정]
       ↓  가중합 계산
  [Composite Raw Score]
       ↓  DQN 보정
  [VHS  0~100]
       ↓  Action Recommender
  [추천 액션: 재배치/할인/폐기/보류]

  컴포넌트 기본 가중치 (합계 1.0)
  ─────────────────────────────────
    heuristic       0.20  비용·거리·수량 종합
    disposal_risk   0.18  폐기위험도 (긴급성)
    demand_forecast 0.15  수요 예측 (재고 소진 위험)
    turnover        0.12  재고 회전율 (악성 정도)
    abc             0.10  상품 가치 등급
    safety_stock    0.08  안전재고·재주문점
    match           0.07  점포-상품 매칭 적합도
    eoq             0.05  발주량 과잉/과소
    network_cost    0.03  최소비용 경로 적합성
    cluster         0.02  점포 클러스터 친화도

  상황 감지 & 가중치 조정
  ─────────────────────────
    EXPIRY_URGENT   유통기한 5일 이내 → disposal·demand 가중치 ↑
    FROZEN_EXCESS   냉동·냉장 카테고리 + 재고 과잉 → eoq·match ↑
    HIGH_COST       이동비용 상위 20% → network_cost·heuristic ↑
    DEMAND_SURGE    수요 증가 추세 → demand·match ↑
    DEAD_STOCK      회전율 DEAD → turnover ↑ disposal ↑
    REORDER_CRISIS  재주문 CRITICAL → safety_stock ↑ demand ↑

  추천 액션 결정 로직
  ─────────────────────
    폐기       : disposal CRITICAL + turnover DEAD + abc C + demand_risk 낮음
    할인 판매  : disposal HIGH+ + (turnover SLOW+ or 수요 감소)
    재배치 이동: match EXCELLENT/GOOD + 목적지 재주문 위험 or 수요 급증
    보류       : 위 조건 미해당 (모니터링 유지)
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════
#  기본 가중치
# ═══════════════════════════════════════════════════════════
# ── VHS 32개 알고리즘 통합 가중치 ──────────────────────────
# Varo Hybrid Decision Score
# = 재고위험(22%) + 판매가능성(18%) + 점포적합도(18%)
# + 재고균형(12%) + 폐기회피(8%) + 실행가능성(8%)
# + 최적화(8%) + 기존연동(6%)
# - 실패위험 패널티(차감)

_BASE_WEIGHTS = {
    # ── A. 재고 위험 (22%) ──
    "disposal_risk_score":   0.09,   # #3 폐기위험도
    "turnover_score":        0.06,   # #2 재고회전율
    "abc_score":             0.04,   # #1 ABC분석
    "aging_score":           0.03,   # #11 재고 노후화
    # ── B. 판매 가능성 (18%) ──
    "demand_forecast_score": 0.08,   # #6 수요예측
    "trend_score":           0.05,   # #12 판매추세
    "newsvendor_score":      0.05,   # #26 Newsvendor
    # ── C. 점포 수요 적합도 (18%) ──
    "match_score":           0.07,   # #8 점포-상품매칭
    "service_level_score":   0.05,   # #25 서비스수준
    "priority_queue_score":  0.03,   # #19 우선순위큐
    "queue_capacity_score":  0.03,   # #27 대기행렬
    # ── D. 재고 균형 개선 (12%) ──
    "category_balance_score":0.05,   # #14 카테고리균형
    "safety_stock_score":    0.04,   # #4 Safety Stock
    "transport_lp_score":    0.03,   # #22 수송문제LP
    # ── E. 폐기 회피 이익 (8%) ──
    "disposal_avoidance_score":0.05, # #18 폐기회피이익
    "discount_sensitivity_score":0.03,# #17 할인민감도
    # ── F. 실행 가능성 (8%) ──
    "bottleneck_score":      0.03,   # #30 병목분석
    "store_capacity_score":  0.03,   # #15 점포처리능력
    "lp_allocation_score":   0.02,   # #21 LP배분
    # ── G. 최적화 모델 (8%) ──
    "multiobjective_score":  0.03,   # #20/#32 다목적
    "topsis_score":          0.02,   # #24 TOPSIS
    "pareto_score":          0.02,   # #31 Pareto
    "assignment_score":      0.01,   # #23 할당문제
    # ── H. 기존 시스템 연동 (6%) ──
    "heuristic_score":       0.03,   # 휴리스틱
    "greedy_score":          0.02,   # 그리디
    "eoq_score":             0.01,   # #5 EOQ
}

# 실패 위험 패널티 (VHS에서 차감)
_PENALTY_WEIGHTS = {
    "relocation_failure_score":  0.04,  # #16 재배치실패위험
    "substitute_conflict_score": 0.02,  # #13 대체상품충돌
}


# ═══════════════════════════════════════════════════════════
#  상황 감지 → 가중치 배율
# ═══════════════════════════════════════════════════════════
_SITUATION_MODS = {
    "EXPIRY_URGENT": {          # 유통기한 5일 이내
        "disposal_risk_score":   2.0,
        "demand_forecast_score": 1.5,
        "match_score":           1.3,
    },
    "FROZEN_EXCESS": {          # 냉동·냉장 과잉
        "eoq_score":             1.8,
        "match_score":           1.5,
        "network_cost_score":    1.4,
    },
    "HIGH_COST": {              # 이동비용 상위 20%
        "network_cost_score":    2.5,
        "heuristic_score":       1.4,
    },
    "DEMAND_SURGE": {           # 수요 급증 추세
        "demand_forecast_score": 1.8,
        "match_score":           1.4,
        "safety_stock_score":    1.2,
        "greedy_score":          1.2,   # 수요 급증 시 그리디 선택 더 중요
    },
    "DEAD_STOCK": {             # 회전율 DEAD
        "turnover_score":        1.8,
        "disposal_risk_score":   1.4,
        "match_score":           1.2,
        "greedy_score":          1.3,   # 악성재고는 그리디 추천 신뢰도 ↑
    },
    "REORDER_CRISIS": {         # 재주문 CRITICAL
        "safety_stock_score":    2.0,
        "demand_forecast_score": 1.5,
        "match_score":           1.3,
    },
}

# ═══════════════════════════════════════════════════════════
#  DQN 보정 범위
# ═══════════════════════════════════════════════════════════
_DQN_MAX_CORRECTION = 8.0   # DQN이 조정할 수 있는 최대 점수 ±8점


def _safe(s, default=0.0):
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(default)
    try:
        return float(s)
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════
#  상황 감지
# ═══════════════════════════════════════════════════════════

def _detect_situations(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 행의 상황 플래그(bool)를 계산해 반환.
    반환 컬럼: sit_EXPIRY_URGENT, sit_FROZEN_EXCESS, sit_HIGH_COST,
               sit_DEMAND_SURGE, sit_DEAD_STOCK, sit_REORDER_CRISIS
    """
    out = pd.DataFrame(index=df.index)

    # 유통기한 5일 이내
    for col in ["expiry_days", "days_to_expiry"]:
        if col in df.columns:
            out["sit_EXPIRY_URGENT"] = _safe(df[col], 999) <= 5
            break
    else:
        out["sit_EXPIRY_URGENT"] = False

    # 냉동·냉장 카테고리
    cat_col = next((c for c in ["category", "inventory_category"] if c in df.columns), None)
    if cat_col:
        cat = df[cat_col].astype(str)
        out["sit_FROZEN_EXCESS"] = cat.str.contains("냉동|냉장|냉|FROZEN|CHILLED", case=False)
    else:
        out["sit_FROZEN_EXCESS"] = False

    # 이동비용 상위 20%
    cost_col = next((c for c in ["estimated_cost", "direct_cost"] if c in df.columns), None)
    if cost_col:
        cost_s = _safe(df[cost_col], 0.0)
        thr    = cost_s.quantile(0.80) if cost_s.gt(0).any() else 0
        out["sit_HIGH_COST"] = cost_s >= thr
    else:
        out["sit_HIGH_COST"] = False

    # 수요 급증
    if "demand_trend" in df.columns:
        out["sit_DEMAND_SURGE"] = df["demand_trend"] == "INCREASING"
    else:
        out["sit_DEMAND_SURGE"] = False

    # 악성재고 DEAD
    if "turnover_grade" in df.columns:
        out["sit_DEAD_STOCK"] = df["turnover_grade"] == "DEAD"
    else:
        out["sit_DEAD_STOCK"] = False

    # 재주문 CRITICAL
    if "reorder_status" in df.columns:
        out["sit_REORDER_CRISIS"] = df["reorder_status"] == "CRITICAL"
    else:
        out["sit_REORDER_CRISIS"] = False

    return out


# ═══════════════════════════════════════════════════════════
#  행별 가중치 계산
# ═══════════════════════════════════════════════════════════

def _calc_row_weights(situations: pd.Series) -> dict:
    """
    한 행의 상황 플래그 → 최종 가중치 dict 반환.
    배율 적용 후 합계 1.0으로 정규화.
    """
    w = dict(_BASE_WEIGHTS)

    for sit_key, mods in _SITUATION_MODS.items():
        col = f"sit_{sit_key}"
        if situations.get(col, False):
            for comp, mult in mods.items():
                if comp in w:
                    w[comp] = w[comp] * mult

    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def _build_weight_matrix(df: pd.DataFrame, sit_df: pd.DataFrame) -> pd.DataFrame:
    """
    sit_df의 각 행 상황에 맞춰 행별 가중치를 계산.
    반환: 행=샘플, 열=컴포넌트 가중치 DataFrame
    """
    weight_rows = []
    for idx in df.index:
        sit_row = sit_df.loc[idx] if idx in sit_df.index else pd.Series(dtype=bool)
        weight_rows.append(_calc_row_weights(sit_row))
    return pd.DataFrame(weight_rows, index=df.index)


# ═══════════════════════════════════════════════════════════
#  컴포넌트 점수 수집
# ═══════════════════════════════════════════════════════════

def _gather_component_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    필요한 점수 컬럼을 수집 + 없는 컬럼은 50점(중립) 기본값 처리.
    greedy_score  : is_greedy_selected(+40pt 보너스) + greedy_rank 기반 점수
    cluster_score 제거 → is_same_cluster 신호를 match_score에 +5점 보너스로 흡수
    """
    out = pd.DataFrame(index=df.index)

    for col in _BASE_WEIGHTS:
        if col == "greedy_score":
            # greedy_rank: 1위=100, 2위=90, ..., 10위+=20, 없음=50
            if "greedy_rank" in df.columns:
                rank = pd.to_numeric(df["greedy_rank"], errors="coerce").fillna(999)
                rank_score = (100 - (rank - 1) * 10).clip(lower=20)
            else:
                rank_score = pd.Series(50.0, index=df.index)

            # is_greedy_selected: True면 +40점 보너스 (최대 100 캡)
            if "is_greedy_selected" in df.columns:
                greedy_bonus = df["is_greedy_selected"].map(
                    {True: 40.0, False: 0.0}
                ).fillna(0.0)
            else:
                greedy_bonus = pd.Series(0.0, index=df.index)

            out["greedy_score"] = (rank_score + greedy_bonus).clip(0, 100)

        else:
            score = _safe(df[col], 50.0) if col in df.columns else 50.0
            # match_score: 동일 클러스터면 +5점 보너스 흡수
            if col == "match_score" and "is_same_cluster" in df.columns:
                cluster_bonus = df["is_same_cluster"].map({True: 5.0, False: 0.0}).fillna(0.0)
                score = (score + cluster_bonus).clip(0, 100)
            out[col] = score

    return out.clip(0, 100)


# ═══════════════════════════════════════════════════════════
#  DQN 보정
# ═══════════════════════════════════════════════════════════

def _dqn_correction(df: pd.DataFrame, raw_score: pd.Series) -> pd.Series:
    """
    DQN reward를 학습 신호로 사용해 raw_score를 보정.
    reward가 없으면 0 보정.
    """
    if "reward" not in df.columns:
        return pd.Series(0.0, index=df.index)

    reward = _safe(df["reward"], 0.0)
    # reward를 ±8점 범위로 정규화
    r_max = reward.abs().max()
    if r_max > 0:
        corr = (reward / r_max) * _DQN_MAX_CORRECTION
    else:
        corr = pd.Series(0.0, index=df.index)

    return corr.clip(-_DQN_MAX_CORRECTION, _DQN_MAX_CORRECTION).round(2)


# ═══════════════════════════════════════════════════════════
#  액션 추천
# ═══════════════════════════════════════════════════════════

_ACTION_ICONS = {
    "재배치 이동": "🚚",
    "할인 판매":   "🏷️",
    "폐기":        "🗑️",
    "보류":        "⏸️",
}

def _recommend_action(row: pd.Series) -> str:
    """
    컴포넌트 점수·등급 조합으로 최적 처리 액션 결정.
    우선순위: 폐기 > 재배치 > 할인 > 보류
    """
    disposal  = str(row.get("disposal_risk_grade", "LOW"))
    turnover  = str(row.get("turnover_grade",      "NORMAL"))
    abc       = str(row.get("abc_grade",           "C"))
    match_g   = str(row.get("match_grade",         "FAIR"))
    reorder   = str(row.get("reorder_status",      "SAFE"))
    trend     = str(row.get("demand_trend",        "STABLE"))
    vhs       = float(row.get("vhs_raw", 50))
    dr_score  = float(row.get("disposal_risk_score", 50))
    demand_rs = float(row.get("demand_risk_score",  50))

    # ── 폐기: 회전 DEAD + 폐기 CRITICAL + 가치 낮음
    if (disposal == "CRITICAL"
            and turnover == "DEAD"
            and abc == "C"
            and demand_rs < 40):
        return "폐기"

    # ── 재배치: 목적지 수요 있고 매칭 좋음
    if (match_g in ("EXCELLENT", "GOOD")
            and (reorder in ("CRITICAL", "WARNING")
                 or trend == "INCREASING"
                 or vhs >= 70)):
        return "재배치 이동"

    # ── 할인: 폐기 위험 있거나 회전 느림
    if (disposal in ("CRITICAL", "HIGH")
            or turnover in ("SLOW", "DEAD")
            or dr_score >= 60):
        return "할인 판매"

    # ── 보류
    return "보류"


# ═══════════════════════════════════════════════════════════
#  메인 함수
# ═══════════════════════════════════════════════════════════

def calculate_varo_hybrid_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    VARO Hybrid Score (VHS) 계산.

    Parameters
    ----------
    df : pd.DataFrame
        run_all_algorithms() 실행 후 DataFrame.

    Returns
    -------
    pd.DataFrame
        vhs, vhs_grade, vhs_action, vhs_action_icon,
        vhs_dqn_correction, vhs_dominant_situation,
        vhs_weight_* (각 컴포넌트 실제 가중치) 컬럼 추가.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # 1. 상황 감지
    sit_df = _detect_situations(out)
    for col in sit_df.columns:
        out[col] = sit_df[col]

    # 2. 컴포넌트 점수 수집
    comp_df = _gather_component_scores(out)

    # 3. 행별 가중치 계산
    weight_df = _build_weight_matrix(out, sit_df)

    # 4. 가중합 (컴포넌트 점수 × 가중치, 행별)
    raw_score = (comp_df * weight_df).sum(axis=1).clip(0, 100)

    # 4b. 실패위험 패널티 차감
    try:
        penalty = pd.Series(0.0, index=out.index)
        for col, w in _PENALTY_WEIGHTS.items():
            vals = _safe(out[col], 50.0) if col in out.columns else 50.0
            if isinstance(vals, pd.Series):
                penalty += vals * w
            else:
                penalty += float(vals) * w
        raw_score = (raw_score - penalty).clip(0, 100)
    except Exception:
        pass

    # 5. DQN 보정
    dqn_corr  = _dqn_correction(out, raw_score)
    vhs       = (raw_score + dqn_corr).clip(0, 100).round(1)

    out["vhs_raw"]            = raw_score.round(1)
    out["vhs_dqn_correction"] = dqn_corr
    out["vhs"]                = vhs

    # 6. 등급
    def _grade(s):
        if s >= 85: return "최우선 처리"
        if s >= 70: return "우선 처리"
        if s >= 55: return "검토 필요"
        if s >= 40: return "모니터링"
        return "후순위"

    out["vhs_grade"] = vhs.apply(_grade)

    # 7. 액션 추천
    out["vhs_action"]      = out.apply(_recommend_action, axis=1)
    out["vhs_action_icon"] = out["vhs_action"].map(_ACTION_ICONS)

    # 8. 주요 상황 레이블
    sit_labels = {
        "sit_EXPIRY_URGENT":  "유통기한 임박",
        "sit_FROZEN_EXCESS":  "냉동·냉장 과잉",
        "sit_HIGH_COST":      "이동비용 높음",
        "sit_DEMAND_SURGE":   "수요 급증",
        "sit_DEAD_STOCK":     "악성재고",
        "sit_REORDER_CRISIS": "재주문 위기",
    }
    def _dominant_sit(row):
        active = [label for col, label in sit_labels.items()
                  if row.get(col, False)]
        return " · ".join(active) if active else "표준"

    out["vhs_dominant_situation"] = out.apply(_dominant_sit, axis=1)

    # 9. 실제 적용된 가중치 저장 (설명용)
    for i, comp in enumerate(_BASE_WEIGHTS):
        out[f"vhs_w_{comp}"] = weight_df[comp].round(4)

    # 10. 컴포넌트별 기여 점수 저장 (차트용)
    for comp in _BASE_WEIGHTS:
        if comp in comp_df.columns and comp in weight_df.columns:
            out[f"vhs_contrib_{comp}"] = (comp_df[comp] * weight_df[comp]).round(2)
    # 패널티 기여 (음수)
    try:
        for col, w in _PENALTY_WEIGHTS.items():
            vals = _safe(out[col], 50.0) if col in out.columns else 50.0
            out[f"vhs_penalty_{col}"] = -(vals * w).round(2) if isinstance(vals, pd.Series) else -(float(vals) * w)
    except Exception:
        pass

    # 11. 순위
    out = out.sort_values("vhs", ascending=False).reset_index(drop=True)
    out["vhs_rank"] = out.index + 1

    return out


def get_vhs_summary(df: pd.DataFrame) -> dict:
    """대시보드 요약 통계."""
    if df is None or df.empty or "vhs" not in df.columns:
        return {}

    action_cnt = df["vhs_action"].value_counts().to_dict()
    sit_counts = {
        "유통기한 임박": int(df.get("sit_EXPIRY_URGENT",  pd.Series([False]*len(df))).sum()),
        "냉동·냉장 과잉": int(df.get("sit_FROZEN_EXCESS",  pd.Series([False]*len(df))).sum()),
        "수요 급증":      int(df.get("sit_DEMAND_SURGE",   pd.Series([False]*len(df))).sum()),
        "악성재고":       int(df.get("sit_DEAD_STOCK",     pd.Series([False]*len(df))).sum()),
        "재주문 위기":    int(df.get("sit_REORDER_CRISIS", pd.Series([False]*len(df))).sum()),
    }

    return {
        "avg_vhs":       round(df["vhs"].mean(), 1),
        "top_product":   df.iloc[0].get("product_name", "-") if len(df) > 0 else "-",
        "top_vhs":       float(df.iloc[0].get("vhs", 0)) if len(df) > 0 else 0,
        "top_action":    df.iloc[0].get("vhs_action", "-") if len(df) > 0 else "-",
        "action_counts": action_cnt,
        "situation_counts": sit_counts,
        "n_total":       len(df),
        "grade_counts":  df["vhs_grade"].value_counts().to_dict(),
    }
