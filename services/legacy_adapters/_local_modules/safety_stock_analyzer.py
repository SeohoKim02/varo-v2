"""
Safety Stock / Reorder Point 분석
───────────────────────────────────────────────────────────
수요 변동성과 리드타임을 반영해 안전재고와 재주문점을 계산한다.

  안전재고 (Safety Stock)
    SS = Z × σ_d × √(lead_time)
    - Z     : 서비스 수준 계수 (기본 95% → Z = 1.65)
    - σ_d   : 일 판매량 표준편차 (demand_std)
    - lead_time : 발주~입고 리드타임 (lead_time_days)

  재주문점 (Reorder Point)
    ROP = (avg_daily_sales × lead_time_days) + safety_stock
    → 재고가 ROP 이하로 떨어지면 발주 신호

  재주문 위험 점수 (0~100)
    현재 재고(stock_qty) 대비 ROP 초과 정도를 점수화.
    stock_qty ≤ SS     : 100점 (즉각 발주 필요)
    stock_qty ≤ ROP    :  75~99점 (발주 권장)
    stock_qty ≤ ROP×2  :  30~74점 (모니터링)
    stock_qty >  ROP×2 :   0~29점 (여유 있음)

  입력 컬럼
    필수 : stock_qty (또는 state_source_stock), avg_daily_sales
    선택 : demand_std     (없으면 avg_daily_sales × 0.3 fallback)
           lead_time_days (없으면 3일 기본값)

  반환 컬럼
    safety_stock          : 안전재고 수량
    reorder_point         : 재주문점 수량
    reorder_risk_score    : 0~100 (높을수록 재고 위험)
    reorder_status        : CRITICAL / WARNING / MONITOR / SAFE
    reorder_advice        : 발주 권고 텍스트
    safety_stock_score    : 0~100 (varo_score 연동용, 위험 높을수록 높음)
"""

import numpy as np
import pandas as pd


# ─── 서비스 수준 계수 ─────────────────────────────────────
_SERVICE_LEVEL_Z = {
    0.90: 1.28,
    0.95: 1.65,   # 기본값
    0.99: 2.33,
}
DEFAULT_Z          = 1.65    # 95% 서비스 수준
DEFAULT_LEAD_TIME  = 3       # 리드타임 기본값 (일)
DEFAULT_DEMAND_STD_RATIO = 0.30   # demand_std 없을 때 avg_daily_sales의 30%


def _safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _get_col(df, *candidates, default=0.0):
    for col in candidates:
        if col in df.columns:
            return _safe_numeric(df[col], default)
    return pd.Series([default] * len(df), index=df.index)


# ─── 안전재고 계산 ────────────────────────────────────────
def _calc_safety_stock(
    demand_std: pd.Series,
    lead_time: pd.Series,
    z: float = DEFAULT_Z,
) -> pd.Series:
    """SS = Z × σ_d × √(L)"""
    ss = z * demand_std * np.sqrt(lead_time.clip(lower=1))
    return ss.clip(lower=0).round(1)


# ─── 재주문점 계산 ────────────────────────────────────────
def _calc_reorder_point(
    avg_daily_sales: pd.Series,
    lead_time: pd.Series,
    safety_stock: pd.Series,
) -> pd.Series:
    """ROP = (d̄ × L) + SS"""
    rop = avg_daily_sales * lead_time.clip(lower=1) + safety_stock
    return rop.clip(lower=0).round(1)


# ─── 재주문 위험 점수 ─────────────────────────────────────
def _calc_risk_score(
    stock_qty: pd.Series,
    safety_stock: pd.Series,
    reorder_point: pd.Series,
) -> pd.Series:
    """
    현재 재고와 SS / ROP 비교 → 0~100 점수.
    높을수록 재고 부족 위험이 크다.
    """
    score = pd.Series(0.0, index=stock_qty.index)

    # 재고 ≤ 안전재고: 즉각 위험 (80~100점)
    mask_critical = stock_qty <= safety_stock
    # ROP가 0이면 / SS가 0이면 나누기 오류 방지
    ss_safe = safety_stock.replace(0, 1)
    ratio_critical = ((ss_safe - stock_qty) / ss_safe).clip(0, 1)
    score = score.where(~mask_critical, 80 + ratio_critical * 20)

    # 안전재고 < 재고 ≤ ROP: 발주 권장 (50~79점)
    mask_warning = (~mask_critical) & (stock_qty <= reorder_point)
    rop_safe = reorder_point.replace(0, 1)
    gap = (reorder_point - stock_qty).clip(lower=0)
    band = (reorder_point - safety_stock).replace(0, 1)
    ratio_warn = (gap / band).clip(0, 1)
    score = score.where(~mask_warning, 50 + ratio_warn * 29)

    # ROP < 재고 ≤ ROP×2: 모니터링 (15~49점)
    mask_monitor = (~mask_critical) & (~mask_warning) & (stock_qty <= reorder_point * 2)
    gap2 = (reorder_point * 2 - stock_qty).clip(lower=0)
    band2 = reorder_point.replace(0, 1)
    ratio_mon = (gap2 / band2).clip(0, 1)
    score = score.where(~mask_monitor, 15 + ratio_mon * 34)

    # 나머지: SAFE (0~14점)
    mask_safe = (~mask_critical) & (~mask_warning) & (~mask_monitor)
    excess = (stock_qty - reorder_point * 2).clip(lower=0)
    excess_norm = (excess / (reorder_point * 2).replace(0, 1)).clip(0, 1)
    score = score.where(~mask_safe, (1 - excess_norm) * 14)

    return score.clip(0, 100).round(1)


def _assign_status(score: float) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 50:
        return "WARNING"
    if score >= 15:
        return "MONITOR"
    return "SAFE"


def _build_advice(row: pd.Series) -> str:
    status = row.get("reorder_status", "SAFE")
    ss     = row.get("safety_stock", 0)
    rop    = row.get("reorder_point", 0)
    stock  = row.get("_stock_raw", 0)

    if status == "CRITICAL":
        return f"즉각 발주 필요 — 현재고({stock:.0f})가 안전재고({ss:.0f}) 이하"
    if status == "WARNING":
        return f"발주 권장 — 현재고({stock:.0f})가 재주문점({rop:.0f}) 이하"
    if status == "MONITOR":
        return f"모니터링 — 재주문점({rop:.0f}) 접근 중 (현재고 {stock:.0f})"
    return f"재고 여유 — 현재고({stock:.0f}), 재주문점({rop:.0f})"


# ─── 메인 함수 ────────────────────────────────────────────
def analyze_safety_stock(
    inventory_df: pd.DataFrame,
    service_level: float = 0.95,
) -> pd.DataFrame:
    """
    Safety Stock / Reorder Point 분석.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        stock_qty, avg_daily_sales 필수.
        demand_std, lead_time_days 있으면 더 정확함.
    service_level : float
        목표 서비스 수준 (0.90 / 0.95 / 0.99).

    Returns
    -------
    pd.DataFrame
        safety_stock, reorder_point, reorder_risk_score,
        reorder_status, reorder_advice, safety_stock_score 컬럼 추가.
    """
    if inventory_df is None or inventory_df.empty:
        return inventory_df

    df  = inventory_df.copy()
    z   = _SERVICE_LEVEL_Z.get(service_level, DEFAULT_Z)

    stock      = _get_col(df, "stock_qty", "state_source_stock", default=0.0)

    # 일평균 판매량 fallback 순서
    for _col in ["avg_daily_sales"]:
        if _col in df.columns:
            daily_sale = _safe_numeric(df[_col], 0.0)
            break
    else:
        for _col in ["sales_30d", "state_source_sales_30d", "source_sales_30d"]:
            if _col in df.columns:
                daily_sale = _safe_numeric(df[_col], 0.0) / 30.0
                break
        else:
            daily_sale = pd.Series([0.0] * len(df), index=df.index)

    # demand_std fallback
    if "demand_std" in df.columns:
        d_std = _safe_numeric(df["demand_std"], default=0.0)
        # 0인 경우 fallback
        d_std = d_std.where(d_std > 0, daily_sale * DEFAULT_DEMAND_STD_RATIO)
    else:
        d_std = daily_sale * DEFAULT_DEMAND_STD_RATIO

    # lead_time fallback
    if "lead_time_days" in df.columns:
        lead = _safe_numeric(df["lead_time_days"], default=DEFAULT_LEAD_TIME)
        lead = lead.where(lead > 0, DEFAULT_LEAD_TIME)
    else:
        lead = pd.Series([float(DEFAULT_LEAD_TIME)] * len(df), index=df.index)

    # 계산
    ss  = _calc_safety_stock(d_std, lead, z)
    rop = _calc_reorder_point(daily_sale, lead, ss)
    risk_score = _calc_risk_score(stock, ss, rop)

    df["safety_stock"]       = ss
    df["reorder_point"]      = rop
    df["reorder_risk_score"] = risk_score
    df["reorder_status"]     = risk_score.apply(_assign_status)

    # advice용 임시 컬럼
    df["_stock_raw"] = stock
    df["reorder_advice"] = df.apply(_build_advice, axis=1)
    df.drop(columns=["_stock_raw"], inplace=True)

    # varo_score 연동용 점수 (reorder_risk_score 그대로 사용)
    df["safety_stock_score"] = risk_score

    return df


def get_safety_stock_summary(df: pd.DataFrame) -> pd.DataFrame:
    """재주문 상태별 요약 테이블. 대시보드 표시용."""
    if df is None or df.empty or "reorder_status" not in df.columns:
        return pd.DataFrame()

    order = ["CRITICAL", "WARNING", "MONITOR", "SAFE"]
    stock_col  = next((c for c in ["stock_qty", "state_source_stock"] if c in df.columns), None)

    agg = {"상품수": ("product_name", "count"),
           "평균위험점수": ("reorder_risk_score", "mean")}
    if stock_col:
        agg["평균현재고"] = (stock_col, "mean")

    summary = (
        df.groupby("reorder_status")
        .agg(**agg)
        .reindex(order)
        .reset_index()
        .rename(columns={"reorder_status": "상태"})
    )
    summary["평균위험점수"] = summary["평균위험점수"].round(1)
    if "평균현재고" in summary.columns:
        summary["평균현재고"] = summary["평균현재고"].round(1)

    return summary
