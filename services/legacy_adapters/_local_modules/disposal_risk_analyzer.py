"""
폐기 위험도 점수 (Disposal Risk Score)
───────────────────────────────────────────────────────────
남은 유통기한·재고량·판매속도·보관기간을 복합적으로 반영해
상품별 폐기 가능성을 0~100 점수로 표현한다.

  점수가 높을수록 폐기 위험이 크다.

  위험 등급
    CRITICAL :  80~100  → 즉시 조치 필요
    HIGH     :  60~79   → 우선 처리 권장
    MEDIUM   :  40~59   → 모니터링 필요
    LOW      :   0~39   → 양호

  점수 구성 (총 100점)
    ① 유통기한 위험도     : 최대 40점  (expiry_days 있을 때 활성)
    ② 재고 소진일 위험도  : 최대 30점  (판매속도 대비 재고)
    ③ 보관 기간 위험도    : 최대 20점  (inbound_days 기반)
    ④ 카테고리 가중치     : 최대 10점  (신선식품·냉장·냉동 가산)

입력 컬럼
  필수 : product_name, stock_qty (or state_source_stock), sales_30d
  선택 : expiry_days     (남은 유통기한 일수 — 없으면 ① 비활성)
         inbound_days    (입고 후 경과일수)
         category        (상품 카테고리 — 신선/냉장/냉동 등)
         unit_cost       (폐기 손실 금액 계산용)

반환 컬럼 추가
  disposal_risk_score  : 0~100
  disposal_risk_grade  : CRITICAL / HIGH / MEDIUM / LOW
  disposal_risk_reason : 주요 위험 원인 텍스트
"""

import pandas as pd
import numpy as np


# ─── 점수 상한 ────────────────────────────────────────────
_W_EXPIRY    = 40   # 유통기한 위험도 최대 점수
_W_TURNOVER  = 30   # 소진일수 위험도 최대 점수
_W_INBOUND   = 20   # 보관기간 위험도 최대 점수
_W_CATEGORY  = 10   # 카테고리 가중치 최대 점수

# ─── 기준값 ───────────────────────────────────────────────
_EXPIRY_CRITICAL_DAYS  = 3    # 3일 이하: 만점
_EXPIRY_HIGH_DAYS      = 7    # 7일 이하: 높음
_EXPIRY_MEDIUM_DAYS    = 14   # 14일 이하: 중간
_EXPIRY_SAFE_DAYS      = 30   # 30일 이상: 0점

_TURNOVER_DEAD_DAYS    = 90   # 소진일수 90일+: 만점
_TURNOVER_SAFE_DAYS    = 30   # 30일 이하: 0점

_INBOUND_DANGER_DAYS   = 90   # 보관 90일+: 만점
_INBOUND_SAFE_DAYS     = 30   # 30일 이하: 0점

# 카테고리 키워드 → 가중치 점수
_CATEGORY_WEIGHTS = {
    "신선": 10, "냉장": 8, "냉동": 6,
    "유제품": 8, "육류": 9, "수산": 9,
    "베이커리": 7, "간편식": 5,
}


def _safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _get_col(df, *candidates, default=0.0):
    for col in candidates:
        if col in df.columns:
            return _safe_numeric(df[col], default)
    return pd.Series([default] * len(df), index=df.index)


# ─── ① 유통기한 위험도 ────────────────────────────────────
def _expiry_risk(expiry_days: pd.Series) -> pd.Series:
    """
    남은 유통기한이 짧을수록 높은 점수 (0 ~ _W_EXPIRY).
    expiry_days = 0(또는 음수)이면 이미 만료 → 만점.
    """
    days = expiry_days.clip(lower=0)

    score = pd.Series(0.0, index=days.index)

    # 이미 만료 또는 당일
    score = score.where(days > 0, _W_EXPIRY)

    # 1 ~ CRITICAL
    mask = (days > 0) & (days <= _EXPIRY_CRITICAL_DAYS)
    score = score.where(~mask, _W_EXPIRY * 0.95)

    # CRITICAL ~ HIGH
    mask = (days > _EXPIRY_CRITICAL_DAYS) & (days <= _EXPIRY_HIGH_DAYS)
    score = score.where(~mask, _W_EXPIRY * 0.75)

    # HIGH ~ MEDIUM
    mask = (days > _EXPIRY_HIGH_DAYS) & (days <= _EXPIRY_MEDIUM_DAYS)
    score = score.where(
        ~mask,
        _W_EXPIRY * (1 - (days - _EXPIRY_HIGH_DAYS) / (_EXPIRY_MEDIUM_DAYS - _EXPIRY_HIGH_DAYS)) * 0.6
    )

    # MEDIUM ~ SAFE
    mask = (days > _EXPIRY_MEDIUM_DAYS) & (days < _EXPIRY_SAFE_DAYS)
    score = score.where(
        ~mask,
        _W_EXPIRY * (1 - (days - _EXPIRY_MEDIUM_DAYS) / (_EXPIRY_SAFE_DAYS - _EXPIRY_MEDIUM_DAYS)) * 0.3
    )

    # SAFE 이상: 0점
    score = score.where(days < _EXPIRY_SAFE_DAYS, 0.0)

    return score.clip(0, _W_EXPIRY).round(2)


# ─── ② 소진일수 위험도 ────────────────────────────────────
def _turnover_risk(stock_qty: pd.Series, sales_30d: pd.Series) -> pd.Series:
    """
    소진 예상일수가 길수록 높은 점수 (0 ~ _W_TURNOVER).
    """
    daily_sales = (sales_30d / 30.0).replace(0, np.nan)
    cover_days  = (stock_qty / daily_sales).fillna(999.0).clip(upper=999.0)

    ratio = (cover_days - _TURNOVER_SAFE_DAYS) / (_TURNOVER_DEAD_DAYS - _TURNOVER_SAFE_DAYS)
    score = ratio.clip(0, 1) * _W_TURNOVER

    return score.round(2)


# ─── ③ 보관기간 위험도 ────────────────────────────────────
def _inbound_risk(inbound_days: pd.Series) -> pd.Series:
    """
    입고 후 경과일이 길수록 높은 점수 (0 ~ _W_INBOUND).
    """
    ratio = (inbound_days - _INBOUND_SAFE_DAYS) / (_INBOUND_DANGER_DAYS - _INBOUND_SAFE_DAYS)
    score = ratio.clip(0, 1) * _W_INBOUND
    return score.round(2)


# ─── ④ 카테고리 가중치 ───────────────────────────────────
def _category_weight(category_series: pd.Series) -> pd.Series:
    def _lookup(cat_val):
        text = str(cat_val).strip()
        for keyword, weight in _CATEGORY_WEIGHTS.items():
            if keyword in text:
                return float(weight)
        return 0.0

    return category_series.apply(_lookup)


# ─── 등급 배정 ────────────────────────────────────────────
def _assign_grade(score: float) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


# ─── 위험 원인 텍스트 생성 ────────────────────────────────
def _build_reason(row: pd.Series) -> str:
    reasons = []

    if row.get("_exp_score", 0) >= _W_EXPIRY * 0.6:
        remaining = row.get("_expiry_days_raw", "?")
        reasons.append(f"유통기한 {remaining:.0f}일 이하" if isinstance(remaining, float) else "유통기한 임박")

    if row.get("_turn_score", 0) >= _W_TURNOVER * 0.6:
        days = row.get("turnover_days_raw", "?")
        reasons.append(f"소진일수 {days:.0f}일 초과" if isinstance(days, float) else "판매속도 저조")

    if row.get("_inb_score", 0) >= _W_INBOUND * 0.6:
        reasons.append("장기 보관")

    if row.get("_cat_score", 0) >= 6:
        reasons.append("신선/냉장 고위험 카테고리")

    return " | ".join(reasons) if reasons else "복합 위험 요소 없음"


# ─── 메인 함수 ────────────────────────────────────────────
def analyze_disposal_risk(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    폐기 위험도 점수를 계산하고 결과 컬럼이 추가된 DataFrame을 반환한다.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        product_name, stock_qty, sales_30d 컬럼 필수.

    Returns
    -------
    pd.DataFrame
        원본에 disposal_risk_score, disposal_risk_grade,
        disposal_risk_reason 컬럼이 추가된 DataFrame.
    """
    if inventory_df is None or inventory_df.empty:
        return inventory_df

    df = inventory_df.copy()

    stock_qty    = _get_col(df, "stock_qty", "state_source_stock", default=0.0)
    sales_30d    = _get_col(df, "sales_30d", "state_source_sales_30d", default=0.0)
    inbound_days = _get_col(df, "inbound_days", "state_inbound_days", default=0.0)

    # 유통기한: expiry_days 또는 days_to_expiry (inventory 시트 원래 컬럼명)
    has_expiry = "expiry_days" in df.columns or "days_to_expiry" in df.columns
    if "expiry_days" in df.columns:
        expiry_days = _safe_numeric(df["expiry_days"], default=999.0)
    elif "days_to_expiry" in df.columns:
        expiry_days = _safe_numeric(df["days_to_expiry"], default=999.0)
    else:
        expiry_days = pd.Series([999.0] * len(df), index=df.index)

    # 카테고리: 컬럼 없으면 ④ 비활성 (0점)
    if "category" in df.columns:
        cat_score = _category_weight(df["category"])
    else:
        cat_score = pd.Series([0.0] * len(df), index=df.index)

    exp_score  = _expiry_risk(expiry_days)
    turn_score = _turnover_risk(stock_qty, sales_30d)
    inb_score  = _inbound_risk(inbound_days)

    # 유통기한 컬럼 없으면 소진일수·보관기간 비중 올려서 100점 맞춤
    if not has_expiry:
        total_base = _W_TURNOVER + _W_INBOUND + _W_CATEGORY
        turn_score = turn_score / total_base * 100
        inb_score  = inb_score  / total_base * 100
        cat_score  = cat_score  / total_base * 100
        total_score = (turn_score + inb_score + cat_score).clip(0, 100).round(1)
    else:
        total_score = (exp_score + turn_score + inb_score + cat_score).clip(0, 100).round(1)

    df["disposal_risk_score"] = total_score
    df["disposal_risk_grade"] = total_score.apply(_assign_grade)

    # 원인 텍스트 계산용 임시 컬럼
    df["_exp_score"]       = exp_score
    df["_turn_score"]      = turn_score
    df["_inb_score"]       = inb_score
    df["_cat_score"]       = cat_score
    df["_expiry_days_raw"] = expiry_days
    df["turnover_days_raw"] = (stock_qty / (sales_30d / 30).replace(0, np.nan)).fillna(999.0).round(1)

    df["disposal_risk_reason"] = df.apply(_build_reason, axis=1)

    # 임시 컬럼 제거
    df.drop(
        columns=["_exp_score", "_turn_score", "_inb_score", "_cat_score",
                 "_expiry_days_raw", "turnover_days_raw"],
        inplace=True,
    )

    return df


def get_disposal_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    폐기 위험 등급별 요약 테이블 반환.
    대시보드 표시용.
    """
    if df is None or df.empty or "disposal_risk_grade" not in df.columns:
        return pd.DataFrame()

    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    summary = (
        df.groupby("disposal_risk_grade")
        .agg(
            상품수=("product_name", "count"),
            평균위험점수=("disposal_risk_score", "mean"),
        )
        .reindex(order)
        .reset_index()
        .rename(columns={"disposal_risk_grade": "등급"})
    )

    summary["평균위험점수"] = summary["평균위험점수"].round(1)

    return summary
