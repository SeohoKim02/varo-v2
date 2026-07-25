"""
수요 예측 (Demand Forecasting)
───────────────────────────────────────────────────────────
가용 데이터로 향후 7일 판매량을 예측하고
재고 부족·과잉 위험을 점수화한다.

  사용 방법 (우선순위)
    1) SMA   : sales_7d 있으면 단순이동평균 (7일 창)
    2) WMA   : sales_7d + avg_daily_sales 있으면 가중이동평균
               최근(7일) 가중치 0.6 / 중기(30일 환산) 가중치 0.4
    3) NAIVE : avg_daily_sales × 7 (기본값 fallback)

  추세 판단 (Trend)
    sales_7d / 7 vs avg_daily_sales 비교
      최근 일평균 > 과거 일평균 × 1.10 → INCREASING
      최근 일평균 < 과거 일평균 × 0.90 → DECREASING
      그 외                            → STABLE

  예측 신뢰 구간
    upper = forecast_daily + Z × demand_std  (Z=1.65, 95%)
    lower = max(0, forecast_daily - Z × demand_std)

  재고 충분도 점수 (0~100, 높을수록 재고 부족 위험 크다)
    남은 일수(stock_qty / forecast_daily) vs 위험 임계값 비교
      7일 이하  : 70~100점
      14일 이하 : 40~69점
      30일 이하 : 15~39점
      30일 초과 : 0~14점

  입력 컬럼
    필수 : avg_daily_sales 또는 sales_30d
    선택 : sales_7d, demand_std, stock_qty, lead_time_days

  반환 컬럼
    demand_forecast_7d      : 향후 7일 예측 판매량
    demand_forecast_daily   : 일평균 예측 판매량
    demand_forecast_upper   : 95% 신뢰 상한 (일 기준)
    demand_forecast_lower   : 95% 신뢰 하한 (일 기준)
    demand_trend            : INCREASING / STABLE / DECREASING
    demand_stockout_days    : 현재 재고로 버틸 수 있는 예상 일수
    demand_risk_score       : 0~100 (재고 부족 위험)
    demand_forecast_method  : 사용된 예측 방법 (SMA/WMA/NAIVE)
    demand_forecast_score   : 0~100 (varo_score 연동용)
"""

import numpy as np
import pandas as pd

# ─── 상수 ────────────────────────────────────────────────
_Z_95           = 1.65
_W_RECENT       = 0.60   # 최근 7일 가중치 (WMA)
_W_HISTORY      = 0.40   # 중기 이력 가중치 (WMA)
_TREND_UP_THR   = 1.10   # +10% 이상 → INCREASING
_TREND_DN_THR   = 0.90   # -10% 이하 → DECREASING


def _safe_num(series, default=0.0):
    if isinstance(series, (int, float)):
        return float(series)
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _get_daily_sales(df: pd.DataFrame) -> pd.Series:
    """일평균 판매량 — avg_daily_sales → sales_30d/30 → 0 순서."""
    if "avg_daily_sales" in df.columns:
        return _safe_num(df["avg_daily_sales"], 0.0)
    for col in ["sales_30d", "state_source_sales_30d"]:
        if col in df.columns:
            return _safe_num(df[col], 0.0) / 30.0
    return pd.Series([0.0] * len(df), index=df.index)


def _get_recent_daily(df: pd.DataFrame) -> pd.Series:
    """최근 7일 일평균 = sales_7d / 7."""
    if "sales_7d" in df.columns:
        return (_safe_num(df["sales_7d"], 0.0) / 7.0).clip(lower=0)
    return pd.Series([np.nan] * len(df), index=df.index)


# ─── 예측 메서드 ─────────────────────────────────────────

def _forecast_naive(daily_avg: pd.Series) -> tuple:
    """NAIVE: avg_daily_sales × 7."""
    pred = daily_avg * 7
    return pred.round(1), "NAIVE"


def _forecast_sma(recent_daily: pd.Series) -> tuple:
    """SMA: 최근 7일 일평균 × 7."""
    pred = recent_daily * 7
    return pred.round(1), "SMA"


def _forecast_wma(recent_daily: pd.Series, hist_daily: pd.Series) -> tuple:
    """WMA: 최근 7일 × 0.6 + 중기 이력 × 0.4, 모두 7일 기준."""
    pred = (_W_RECENT * recent_daily + _W_HISTORY * hist_daily) * 7
    return pred.round(1), "WMA"


# ─── 추세 판단 ───────────────────────────────────────────

def _calc_trend(recent_daily: pd.Series, hist_daily: pd.Series) -> pd.Series:
    safe_hist = hist_daily.replace(0, np.nan)
    ratio = recent_daily / safe_hist

    trend = pd.Series("STABLE", index=ratio.index)
    trend = trend.where(ratio.isna() | (ratio >= _TREND_DN_THR), "DECREASING")
    trend = trend.where(ratio.isna() | (ratio <= _TREND_UP_THR), "INCREASING")
    # NaN이면 STABLE
    trend = trend.where(~ratio.isna(), "STABLE")
    return trend


# ─── 신뢰 구간 ───────────────────────────────────────────

def _calc_intervals(
    forecast_daily: pd.Series,
    demand_std: pd.Series,
) -> tuple:
    margin = _Z_95 * demand_std
    upper = (forecast_daily + margin).round(2)
    lower = (forecast_daily - margin).clip(lower=0).round(2)
    return upper, lower


# ─── 재고 충분도 점수 ─────────────────────────────────────

def _calc_risk_score(
    stock_qty: pd.Series,
    forecast_daily: pd.Series,
    lead_time: pd.Series,
) -> pd.Series:
    """
    재고가 예측 수요 대비 얼마나 버티는지 점수화.
    남은 일수 = stock_qty / forecast_daily
    리드타임 이하로 떨어지면 더 위험.
    """
    safe_fc = forecast_daily.replace(0, np.nan)
    stockout_days = (stock_qty / safe_fc).fillna(999).clip(upper=999)

    score = pd.Series(0.0, index=stock_qty.index)

    # 리드타임 이하: 발주해도 재고 소진 → 70~100점
    lt = lead_time.clip(lower=1)
    at_critical = stockout_days <= lt
    crit_ratio = (1 - stockout_days / lt.replace(0, 1)).clip(0, 1)
    score = score.where(~at_critical, 70 + crit_ratio * 30)

    # 7일 이하 (CRITICAL 미만): 40~69점
    at_7 = (~at_critical) & (stockout_days <= 7)
    r7 = (1 - stockout_days / 7).clip(0, 1)
    score = score.where(~at_7, 40 + r7 * 29)

    # 14일 이하: 20~39점
    at_14 = (~at_critical) & (~at_7) & (stockout_days <= 14)
    r14 = (1 - stockout_days / 14).clip(0, 1)
    score = score.where(~at_14, 20 + r14 * 19)

    # 30일 이하: 5~19점
    at_30 = (~at_critical) & (~at_7) & (~at_14) & (stockout_days <= 30)
    r30 = (1 - stockout_days / 30).clip(0, 1)
    score = score.where(~at_30, 5 + r30 * 14)

    # 30일 초과: 0~4점
    safe = (~at_critical) & (~at_7) & (~at_14) & (~at_30)
    excess = ((stockout_days - 30) / 30).clip(0, 1)
    score = score.where(~safe, (1 - excess) * 4)

    return score.clip(0, 100).round(1), stockout_days.round(1)


# ─── 메인 함수 ───────────────────────────────────────────

def analyze_demand_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    수요 예측 분석.

    Parameters
    ----------
    df : pd.DataFrame
        avg_daily_sales 또는 sales_30d 컬럼 필수.
        sales_7d, demand_std, stock_qty, lead_time_days 있으면 더 정확.

    Returns
    -------
    pd.DataFrame
        demand_forecast_7d, demand_forecast_daily, demand_trend,
        demand_risk_score, demand_forecast_score 등 컬럼 추가.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    hist_daily   = _get_daily_sales(out)
    recent_daily = _get_recent_daily(out)

    # 예측 방법 선택
    has_recent  = "sales_7d" in out.columns
    has_history = hist_daily.gt(0).any()

    if has_recent and has_history:
        forecast_7d, method_label = _forecast_wma(recent_daily, hist_daily)
        forecast_daily = forecast_7d / 7
        # recent_daily가 0인 경우 hist로 fallback
        mask_no_recent = recent_daily <= 0
        fc_naive, _ = _forecast_naive(hist_daily)
        forecast_7d    = forecast_7d.where(~mask_no_recent, fc_naive)
        forecast_daily = forecast_daily.where(~mask_no_recent, hist_daily)
        method_series  = pd.Series(method_label, index=out.index)
        method_series  = method_series.where(~mask_no_recent, "NAIVE")
    elif has_recent:
        forecast_7d, method_label = _forecast_sma(recent_daily)
        forecast_daily = forecast_7d / 7
        method_series  = pd.Series(method_label, index=out.index)
    else:
        forecast_7d, method_label = _forecast_naive(hist_daily)
        forecast_daily = hist_daily.copy()
        method_series  = pd.Series(method_label, index=out.index)

    # 수요 표준편차 fallback
    if "demand_std" in out.columns:
        d_std = _safe_num(out["demand_std"], 0.0)
        d_std = d_std.where(d_std > 0, hist_daily * 0.25)
    else:
        d_std = hist_daily * 0.25

    # 추세
    if has_recent:
        trend = _calc_trend(recent_daily, hist_daily)
    else:
        trend = pd.Series("STABLE", index=out.index)

    # 신뢰 구간
    upper, lower = _calc_intervals(forecast_daily, d_std)

    # 재고 충분도
    stock = _safe_num(out["stock_qty"], 0.0) if "stock_qty" in out.columns \
        else _safe_num(out.get("state_source_stock", pd.Series([0.0]*len(out), index=out.index)), 0.0)

    if "lead_time_days" in out.columns:
        lead = _safe_num(out["lead_time_days"], 3.0).clip(lower=1)
    else:
        lead = pd.Series([3.0] * len(out), index=out.index)

    risk_score, stockout_days = _calc_risk_score(stock, forecast_daily, lead)

    # 결과 저장
    out["demand_forecast_7d"]     = forecast_7d
    out["demand_forecast_daily"]  = forecast_daily.round(2)
    out["demand_forecast_upper"]  = upper
    out["demand_forecast_lower"]  = lower
    out["demand_trend"]           = trend
    out["demand_stockout_days"]   = stockout_days
    out["demand_risk_score"]      = risk_score
    out["demand_forecast_method"] = method_series
    out["demand_forecast_score"]  = risk_score   # varo_score 연동용

    return out


def get_demand_forecast_summary(df: pd.DataFrame) -> pd.DataFrame:
    """추세별 요약 테이블 — 대시보드 표시용."""
    if df is None or df.empty or "demand_trend" not in df.columns:
        return pd.DataFrame()

    order = ["INCREASING", "STABLE", "DECREASING"]
    summary = (
        df.groupby("demand_trend")
        .agg(
            상품수=("product_name", "count"),
            평균예측7d=("demand_forecast_7d", "mean"),
            평균위험점수=("demand_risk_score", "mean"),
        )
        .reindex(order)
        .reset_index()
        .rename(columns={"demand_trend": "추세"})
    )
    summary["평균예측7d"]  = summary["평균예측7d"].round(1)
    summary["평균위험점수"] = summary["평균위험점수"].round(1)
    return summary
