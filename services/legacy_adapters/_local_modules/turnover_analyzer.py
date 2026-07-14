"""
재고 회전율 분석 (Inventory Turnover Analysis)
───────────────────────────────────────────────────────────
판매량과 평균재고를 이용해 상품이 얼마나 빠르게 소진되는지 계산한다.
악성재고(느린 회전) 가능성을 수치로 판단한다.

  회전율 = 연간 판매량 / 평균재고  (연간 기준 표준화)
  재고소진일수 = stock_qty / 일평균판매량

  등급 기준 (소진일수 기준)
    FAST   :  ~30일   → 우수 (빠른 회전)
    NORMAL : 30~60일  → 양호
    SLOW   : 60~90일  → 주의 (악성재고 주의)
    DEAD   :  90일+   → 위험 (악성재고 가능성 높음)

입력 컬럼
  필수 : product_name, sales_30d, stock_qty  (또는 state_source_stock)
  선택 : avg_inventory (없으면 stock_qty 사용),
         sales_7d (있으면 최근 추세 반영)

반환 컬럼 추가
  turnover_days         : 재고소진 예상일수
  turnover_rate_annual  : 연간 회전율
  turnover_grade        : FAST / NORMAL / SLOW / DEAD
  turnover_score        : 0~100 (높을수록 회전 빠름)
  turnover_risk_flag    : True/False (SLOW/DEAD = True)
"""

import pandas as pd
import numpy as np


# ─── 등급 기준 ────────────────────────────────────────────
_GRADE_DAYS = {
    "FAST":   (0,   30),
    "NORMAL": (30,  60),
    "SLOW":   (60,  90),
    "DEAD":   (90,  9999),
}

# 점수: 소진일수 → 100점 기준 (낮을수록 좋음 → 반전 정규화)
_MAX_REFERENCE_DAYS = 180   # 180일 이상이면 0점


def _safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _get_stock_qty(df: pd.DataFrame) -> pd.Series:
    """stock_qty 컬럼 우선, 없으면 state_source_stock fallback"""
    for col in ["stock_qty", "state_source_stock", "source_stock_qty"]:
        if col in df.columns:
            return _safe_numeric(df[col])
    return pd.Series([0.0] * len(df), index=df.index)


def _get_sales_30d(df: pd.DataFrame) -> pd.Series:
    for col in ["sales_30d", "state_source_sales_30d", "source_sales_30d"]:
        if col in df.columns:
            return _safe_numeric(df[col])
    # avg_daily_sales 있으면 × 30 변환
    if "avg_daily_sales" in df.columns:
        return _safe_numeric(df["avg_daily_sales"]) * 30.0
    return pd.Series([0.0] * len(df), index=df.index)


def _get_avg_inventory(df: pd.DataFrame) -> pd.Series:
    """avg_inventory 컬럼 우선, 없으면 stock_qty 사용"""
    if "avg_inventory" in df.columns:
        avg = _safe_numeric(df["avg_inventory"])
        # 0인 경우 stock_qty fallback
        stock = _get_stock_qty(df)
        return avg.where(avg > 0, stock)
    return _get_stock_qty(df)


def _calc_days(stock_qty: pd.Series, sales_30d: pd.Series) -> pd.Series:
    """재고소진 예상일수 계산 (판매 0이면 999 처리)"""
    daily_sales = sales_30d / 30.0
    days = stock_qty / daily_sales.replace(0, np.nan)
    return days.fillna(999.0).clip(upper=999.0).round(1)


def _calc_annual_turnover(sales_30d: pd.Series, avg_inventory: pd.Series) -> pd.Series:
    """
    연간 회전율 = (sales_30d × 12) / avg_inventory
    평균재고가 0이면 0으로 처리
    """
    annual_sales = sales_30d * 12.0
    rate = annual_sales / avg_inventory.replace(0, np.nan)
    return rate.fillna(0.0).round(2)


def _assign_grade(days: float) -> str:
    if days <= 30:
        return "FAST"
    if days <= 60:
        return "NORMAL"
    if days <= 90:
        return "SLOW"
    return "DEAD"


def _calc_score(days: pd.Series) -> pd.Series:
    """
    소진일수가 낮을수록 높은 점수 (0~100).
    _MAX_REFERENCE_DAYS(180일) 이상이면 0점.
    """
    score = (1 - (days.clip(upper=_MAX_REFERENCE_DAYS) / _MAX_REFERENCE_DAYS)) * 100
    return score.round(1)


def analyze_turnover(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    재고 회전율 분석을 수행하고 결과 컬럼이 추가된 DataFrame을 반환한다.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        product_name, sales_30d, stock_qty 컬럼 필수.

    Returns
    -------
    pd.DataFrame
        원본에 turnover_days, turnover_rate_annual, turnover_grade,
        turnover_score, turnover_risk_flag 컬럼이 추가된 DataFrame.
    """
    if inventory_df is None or inventory_df.empty:
        return inventory_df

    df = inventory_df.copy()

    stock_qty    = _get_stock_qty(df)
    sales_30d    = _get_sales_30d(df)
    avg_inventory = _get_avg_inventory(df)

    df["turnover_days"]        = _calc_days(stock_qty, sales_30d)
    df["turnover_rate_annual"] = _calc_annual_turnover(sales_30d, avg_inventory)
    df["turnover_grade"]       = df["turnover_days"].apply(_assign_grade)
    df["turnover_score"]       = _calc_score(df["turnover_days"])
    df["turnover_risk_flag"]   = df["turnover_grade"].isin(["SLOW", "DEAD"])

    return df


def get_turnover_summary(turnover_df: pd.DataFrame) -> pd.DataFrame:
    """
    회전율 등급별 요약 테이블 반환.
    대시보드 표시용.
    """
    if turnover_df is None or turnover_df.empty or "turnover_grade" not in turnover_df.columns:
        return pd.DataFrame()

    order = ["FAST", "NORMAL", "SLOW", "DEAD"]

    summary = (
        turnover_df.groupby("turnover_grade")
        .agg(
            상품수=("product_name", "count"),
            평균소진일수=("turnover_days", "mean"),
            평균회전율=("turnover_rate_annual", "mean"),
            위험상품수=("turnover_risk_flag", "sum"),
        )
        .reindex(order)
        .reset_index()
        .rename(columns={"turnover_grade": "등급"})
    )

    summary["평균소진일수"] = summary["평균소진일수"].round(1)
    summary["평균회전율"]   = summary["평균회전율"].round(2)

    return summary
