"""
점포-상품 매칭 최적화 (Store-Product Matching Optimization)
───────────────────────────────────────────────────────────
앞선 알고리즘(ABC, 회전율, 폐기위험도, Safety Stock, 수요예측, 클러스터링)의
결과를 종합해 "이 상품이 이 점포에 얼마나 잘 맞는가"를 0~100점으로 산출한다.

  매칭 점수 구성 (합계 100점)
    ① 수요 적합성  (30점) : 목적 점포 수요 vs 상품 판매속도
    ② 긴급도 보너스(25점) : 폐기위험·Safety Stock CRITICAL → 빠른 이동 필요
    ③ ABC 매칭    (20점) : A등급 상품 → 고회전 점포 우선
    ④ 클러스터 친화(15점) : 동일 클러스터면 이동 시너지 높음
    ⑤ 거리 패널티 (10점) : 거리가 짧을수록 높은 점수

  매칭 등급
    EXCELLENT (80+) : 최적 매칭 — 즉시 실행 권장
    GOOD      (60+) : 양호 매칭 — 우선 검토
    FAIR      (40+) : 보통 매칭 — 조건부 권장
    POOR      (40미만): 낮은 매칭 — 재검토 필요

  입력
    final_recommendations DataFrame (알고리즘 결과 컬럼 포함)

  반환 컬럼
    match_score   : 0~100
    match_grade   : EXCELLENT / GOOD / FAIR / POOR
    match_reason  : 주요 매칭 근거 텍스트
"""

import numpy as np
import pandas as pd


# ─── 상수 ────────────────────────────────────────────────
_W_DEMAND   = 30
_W_URGENCY  = 25
_W_ABC      = 20
_W_CLUSTER  = 15
_W_DISTANCE = 10

_MATCH_GRADES = [
    (80, "EXCELLENT"),
    (60, "GOOD"),
    (40, "FAIR"),
    (0,  "POOR"),
]


def _safe(s, default=0.0):
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(default)
    try:
        return float(s)
    except Exception:
        return default


# ─── ① 수요 적합성 점수 (0~30) ───────────────────────────

def _demand_fit(df: pd.DataFrame) -> pd.Series:
    """
    목적 점포 수요 / 출발 점포 현재 판매량 비율.
    목적 점포가 더 잘 팔릴수록 이동이 유리.
    """
    src_sales = None
    tgt_sales = None

    # 출발 점포 판매량
    for col in ["avg_daily_sales", "sales_30d"]:
        if col in df.columns:
            src_sales = _safe(df[col], 0.0)
            if col == "sales_30d":
                src_sales = src_sales / 30.0
            break
    if src_sales is None:
        for col in ["state_source_sales_30d"]:
            if col in df.columns:
                src_sales = _safe(df[col], 0.0) / 30.0
                break
    if src_sales is None:
        src_sales = pd.Series([1.0] * len(df), index=df.index)

    # 목적 점포 판매량 (state_target_sales_30d)
    for col in ["state_target_sales_30d", "target_sales_30d"]:
        if col in df.columns:
            tgt_sales = _safe(df[col], 0.0) / 30.0
            break
    if tgt_sales is None:
        tgt_sales = src_sales  # 데이터 없으면 같다고 가정 → 중립 점수

    safe_src = src_sales.replace(0, 1e-6)
    ratio = (tgt_sales / safe_src).clip(0, 3)

    # ratio > 1 : 목적지가 더 잘 팔림 → 높은 점수
    score = (ratio / 3.0 * _W_DEMAND).clip(0, _W_DEMAND)
    return score.round(2)


# ─── ② 긴급도 보너스 (0~25) ──────────────────────────────

def _urgency_score(df: pd.DataFrame) -> pd.Series:
    """
    폐기위험 CRITICAL/HIGH + Safety Stock CRITICAL → 빠른 이동 필요 → 높은 점수.
    """
    score = pd.Series(0.0, index=df.index)

    # 폐기 위험도
    disposal_map = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.3, "LOW": 0.0}
    if "disposal_risk_grade" in df.columns:
        d = df["disposal_risk_grade"].map(disposal_map).fillna(0.3)
        score += d * 12

    # Safety Stock 위험
    ss_map = {"CRITICAL": 1.0, "WARNING": 0.6, "MONITOR": 0.2, "SAFE": 0.0}
    if "reorder_status" in df.columns:
        s = df["reorder_status"].map(ss_map).fillna(0.2)
        score += s * 8

    # 수요 예측 위험
    if "demand_risk_score" in df.columns:
        score += _safe(df["demand_risk_score"], 0.0) / 100.0 * 5

    return score.clip(0, _W_URGENCY).round(2)


# ─── ③ ABC 매칭 점수 (0~20) ──────────────────────────────

def _abc_match(df: pd.DataFrame) -> pd.Series:
    """
    A등급 상품 → 고회전(FAST/NORMAL) 점포에서 이동하면 감점,
               고수요 점포로 이동하면 가점.
    C등급 상품 → 어디든 이동해서 처리하는 게 나음 → 중립.
    """
    abc_map = {"A": 1.0, "B": 0.6, "C": 0.3}
    if "abc_grade" in df.columns:
        abc_w = df["abc_grade"].map(abc_map).fillna(0.5)
    else:
        abc_w = pd.Series([0.5] * len(df), index=df.index)

    # A등급 + 목적지 판매량 높으면 높은 점수
    tgt_sales_norm = pd.Series([0.5] * len(df), index=df.index)
    for col in ["state_target_sales_30d"]:
        if col in df.columns:
            ts = _safe(df[col], 0.0)
            max_ts = ts.max()
            if max_ts > 0:
                tgt_sales_norm = (ts / max_ts).clip(0, 1)
            break

    score = abc_w * tgt_sales_norm * _W_ABC
    return score.clip(0, _W_ABC).round(2)


# ─── ④ 클러스터 친화 점수 (0~15) ─────────────────────────

def _cluster_affinity(df: pd.DataFrame) -> pd.Series:
    """동일 클러스터면 만점, 아니면 0."""
    if "is_same_cluster" in df.columns:
        return df["is_same_cluster"].astype(float) * _W_CLUSTER
    return pd.Series([_W_CLUSTER * 0.5] * len(df), index=df.index)


# ─── ⑤ 거리 패널티 점수 (0~10) ───────────────────────────

def _distance_score(df: pd.DataFrame) -> pd.Series:
    """거리가 짧을수록 높은 점수. 최대 10km 기준."""
    for col in ["state_distance_km", "direct_distance_km", "recommended_distance_km"]:
        if col in df.columns:
            dist = _safe(df[col], 5.0).clip(lower=0.1)
            score = (_W_DISTANCE * (1 - (dist / 15.0).clip(0, 1))).clip(0, _W_DISTANCE)
            return score.round(2)
    return pd.Series([_W_DISTANCE * 0.5] * len(df), index=df.index)


# ─── 등급 & 이유 ──────────────────────────────────────────

def _assign_grade(score: float) -> str:
    for threshold, grade in _MATCH_GRADES:
        if score >= threshold:
            return grade
    return "POOR"


def _build_reason(row: pd.Series) -> str:
    parts = []
    g = row.get("match_grade", "FAIR")
    if g == "EXCELLENT":
        parts.append("최적 매칭")
    elif g == "GOOD":
        parts.append("양호 매칭")
    elif g == "FAIR":
        parts.append("보통 매칭")
    else:
        parts.append("낮은 매칭")

    if row.get("disposal_risk_grade") in ("CRITICAL", "HIGH"):
        parts.append("폐기위험 높음")
    if row.get("reorder_status") in ("CRITICAL", "WARNING"):
        parts.append("재주문 위험")
    if row.get("abc_grade") == "A":
        parts.append("핵심(A)상품")
    if row.get("is_same_cluster"):
        parts.append("동일클러스터")

    return " · ".join(parts)


# ─── 메인 함수 ───────────────────────────────────────────

def analyze_store_product_matching(df: pd.DataFrame) -> pd.DataFrame:
    """
    점포-상품 매칭 최적화 분석.

    Parameters
    ----------
    df : pd.DataFrame
        final_recommendations + 알고리즘 결과 컬럼 포함.

    Returns
    -------
    pd.DataFrame
        match_score, match_grade, match_reason 컬럼 추가.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    s_demand   = _demand_fit(out)
    s_urgency  = _urgency_score(out)
    s_abc      = _abc_match(out)
    s_cluster  = _cluster_affinity(out)
    s_distance = _distance_score(out)

    total = (s_demand + s_urgency + s_abc + s_cluster + s_distance).clip(0, 100).round(1)

    out["match_score"]        = total
    out["match_score_demand"] = s_demand
    out["match_score_urgency"]= s_urgency
    out["match_score_abc"]    = s_abc
    out["match_score_cluster"]= s_cluster
    out["match_score_dist"]   = s_distance
    out["match_grade"]        = total.apply(_assign_grade)
    out["match_reason"]       = out.apply(_build_reason, axis=1)

    return out


def get_matching_summary(df: pd.DataFrame) -> pd.DataFrame:
    """매칭 등급별 요약 — 대시보드 표시용."""
    if df is None or df.empty or "match_grade" not in df.columns:
        return pd.DataFrame()

    order = ["EXCELLENT", "GOOD", "FAIR", "POOR"]
    summary = (
        df.groupby("match_grade")
        .agg(
            건수=("product_name", "count"),
            평균매칭점수=("match_score", "mean"),
            평균휴리스틱=("heuristic_score", "mean"),
        )
        .reindex(order)
        .reset_index()
        .rename(columns={"match_grade": "등급"})
    )
    for col in ["평균매칭점수", "평균휴리스틱"]:
        summary[col] = summary[col].round(1)
    return summary
