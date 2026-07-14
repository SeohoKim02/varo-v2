"""
EOQ 적정 발주량 (Economic Order Quantity)
───────────────────────────────────────────────────────────
주문비용과 보관비용의 균형점에서 총 비용이 최소가 되는
경제적 주문량과 과잉 발주 위험을 계산한다.

  EOQ 공식 (Wilson Formula)
    EOQ = √(2 × D × S / H)
    - D : 연간 수요량 (avg_daily_sales × 365)
    - S : 1회 주문비용 (order_cost)
    - H : 연간 단위당 보관비용 (unit_cost × holding_rate)

  보관비율 (holding_rate) 기본값
    카테고리별 현실적 보관비율 적용
      신선/유제품 : 35% (빠른 회전, 냉장 비용)
      냉동        : 30% (냉동 전기비)
      일반        : 20%

  과잉 발주 위험도 (0~100)
    현재 발주 패턴 vs EOQ 비교.
    (현재 추정 발주량 = stock_qty - safety_stock) 기준.
    발주량이 EOQ의 2배 이상 → 과잉 위험 높음.
    발주량이 EOQ의 0.5배 이하 → 과소 발주 위험.

  입력 컬럼
    필수 : avg_daily_sales, unit_cost
    선택 : order_cost (없으면 5,000원 기본값)
           safety_stock (safety_stock_analyzer 결과)
           stock_qty, category

  반환 컬럼
    eoq_qty           : 경제적 주문량
    eoq_order_cycle   : EOQ 기준 발주 주기 (일)
    eoq_total_cost    : EOQ 적용 시 연간 총비용 (주문비+보관비)
    eoq_risk_score    : 0~100 (과잉/과소 발주 위험)
    eoq_status        : OVER / OPTIMAL / UNDER / NO_DATA
    eoq_advice        : 발주 권고 텍스트
    eoq_score         : 0~100 (varo_score 연동용, 위험 높을수록 높음)
"""

import numpy as np
import pandas as pd


# ─── 카테고리별 보관비율 ──────────────────────────────────
_HOLDING_RATE_MAP = {
    "신선": 0.35, "유제품": 0.35, "냉장": 0.32,
    "냉동": 0.30, "육류": 0.38, "수산": 0.38,
    "디저트": 0.25, "간편식": 0.22,
    "음료": 0.18, "과자": 0.18,
}
_DEFAULT_HOLDING_RATE = 0.20
_DEFAULT_ORDER_COST   = 5_000   # 원
_MIN_ANNUAL_DEMAND    = 1.0     # 연간 수요 최솟값 (0 방지)


def _safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _get_holding_rate(category_series: pd.Series) -> pd.Series:
    def _lookup(cat):
        c = str(cat)
        for kw, rate in _HOLDING_RATE_MAP.items():
            if kw in c:
                return rate
        return _DEFAULT_HOLDING_RATE
    return category_series.apply(_lookup)


def _calc_eoq(
    annual_demand: pd.Series,
    order_cost: pd.Series,
    holding_cost_per_unit: pd.Series,
) -> pd.Series:
    """EOQ = √(2DS/H) — 모든 값이 양수일 때만 계산."""
    safe_h = holding_cost_per_unit.replace(0, np.nan)
    safe_d = annual_demand.clip(lower=_MIN_ANNUAL_DEMAND)
    eoq = np.sqrt(2 * safe_d * order_cost / safe_h)
    return eoq.fillna(0).round(1)


def _calc_order_cycle(eoq: pd.Series, avg_daily_sales: pd.Series) -> pd.Series:
    """발주 주기 = EOQ / 일평균판매량 (일)"""
    safe_sales = avg_daily_sales.replace(0, np.nan)
    cycle = eoq / safe_sales
    return cycle.fillna(999).clip(upper=365).round(1)


def _calc_annual_total_cost(
    annual_demand: pd.Series,
    eoq: pd.Series,
    order_cost: pd.Series,
    holding_cost_per_unit: pd.Series,
) -> pd.Series:
    """
    연간 총비용 = 주문비용 + 보관비용
    TC = (D/Q)×S + (Q/2)×H
    """
    safe_eoq = eoq.replace(0, np.nan)
    order_part   = (annual_demand / safe_eoq) * order_cost
    holding_part = (safe_eoq / 2) * holding_cost_per_unit
    tc = order_part + holding_part
    return tc.fillna(0).round(0)


def _calc_current_order_qty(
    stock_qty: pd.Series,
    safety_stock: pd.Series,
) -> pd.Series:
    """
    현재 추정 발주량 = stock_qty - safety_stock
    (실제 발주기록 없으므로 현재 재고에서 안전재고를 뺀 값으로 추정)
    최소 0으로 처리.
    """
    est = stock_qty - safety_stock.fillna(0)
    return est.clip(lower=0)


def _calc_eoq_risk_score(
    current_order_qty: pd.Series,
    eoq: pd.Series,
) -> pd.Series:
    """
    EOQ 대비 현재 발주량의 이탈도를 0~100 점수로 변환.
    점수 높을수록 과잉/과소 위험 크다.

    - 현재 발주 = 0  : 데이터 없음 → 50점 중립
    - EOQ = 0        : 계산 불가  → 50점 중립
    - 2배 이상 과잉  : 70~100점
    - 0.5배 이하 과소: 60~80점
    - 0.7~1.3배 적정 : 0~20점
    """
    score = pd.Series(50.0, index=current_order_qty.index)

    safe_eoq = eoq.replace(0, np.nan)
    ratio = current_order_qty / safe_eoq  # current / EOQ

    # 데이터 없는 경우 (current_order_qty = 0)
    no_data = current_order_qty <= 0
    score = score.where(~no_data, 50.0)

    # 과잉 발주 (ratio > 1): ratio가 2이면 100점
    over = (~no_data) & (ratio > 1.0)
    over_score = ((ratio - 1.0) / 1.0).clip(0, 1) * 100
    score = score.where(~over, over_score)

    # 과소 발주 (0 < ratio ≤ 1): ratio가 0.5이면 60점, 0이면 80점
    under = (~no_data) & (ratio > 0) & (ratio <= 1.0)
    under_score = ((1.0 - ratio) / 1.0).clip(0, 1) * 80
    score = score.where(~under, under_score)

    # EOQ 계산 불가 (eoq = 0)
    eoq_zero = eoq <= 0
    score = score.where(~eoq_zero, 50.0)

    return score.clip(0, 100).round(1)


def _assign_status(row: pd.Series) -> str:
    if row.get("_current_order_qty", 0) <= 0 or row.get("eoq_qty", 0) <= 0:
        return "NO_DATA"
    ratio = row["_current_order_qty"] / row["eoq_qty"]
    if ratio > 1.5:
        return "OVER"
    if ratio < 0.6:
        return "UNDER"
    return "OPTIMAL"


def _build_advice(row: pd.Series) -> str:
    status = row.get("eoq_status", "NO_DATA")
    eoq    = row.get("eoq_qty", 0)
    cur    = row.get("_current_order_qty", 0)
    cycle  = row.get("eoq_order_cycle", 0)

    if status == "NO_DATA":
        return f"발주 이력 부족 — EOQ 기준({eoq:.0f}개) 참고 권장"
    if status == "OVER":
        return f"과잉 발주 위험 — 현재({cur:.0f}개) > EOQ({eoq:.0f}개). 발주 주기 {cycle:.0f}일 권장"
    if status == "UNDER":
        return f"과소 발주 위험 — 현재({cur:.0f}개) < EOQ({eoq:.0f}개). 발주량 증가 검토"
    return f"적정 발주 — EOQ({eoq:.0f}개), 권장 발주 주기 {cycle:.0f}일"


# ─── 메인 함수 ────────────────────────────────────────────
def analyze_eoq(
    inventory_df: pd.DataFrame,
    holding_rate: float = None,
) -> pd.DataFrame:
    """
    EOQ 분석을 수행하고 결과 컬럼이 추가된 DataFrame을 반환한다.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        avg_daily_sales, unit_cost 컬럼 필수.
    holding_rate : float, optional
        연간 보관비율 (0~1). None이면 카테고리별 자동 적용.

    Returns
    -------
    pd.DataFrame
        eoq_qty, eoq_order_cycle, eoq_total_cost,
        eoq_risk_score, eoq_status, eoq_advice, eoq_score 컬럼 추가.
    """
    if inventory_df is None or inventory_df.empty:
        return inventory_df

    df = inventory_df.copy()

    # ── 일평균 판매량: avg_daily_sales → sales_30d/30 → state_source_sales_30d/30 순서
    if "avg_daily_sales" in df.columns:
        daily_sale = _safe_numeric(df["avg_daily_sales"], 0.0)
    elif "sales_30d" in df.columns:
        daily_sale = _safe_numeric(df["sales_30d"], 0.0) / 30.0
    elif "state_source_sales_30d" in df.columns:
        daily_sale = _safe_numeric(df["state_source_sales_30d"], 0.0) / 30.0
    else:
        daily_sale = pd.Series([0.0] * len(df), index=df.index)

    # ── 단가: unit_cost → state_unit_cost 순서
    if "unit_cost" in df.columns:
        unit_cost = _safe_numeric(df["unit_cost"], 0.0)
    elif "state_unit_cost" in df.columns:
        unit_cost = _safe_numeric(df["state_unit_cost"], 0.0)
    else:
        unit_cost = pd.Series([0.0] * len(df), index=df.index)

    # ── 현재 재고: stock_qty → state_source_stock 순서
    if "stock_qty" in df.columns:
        stock_qty = _safe_numeric(df["stock_qty"], 0.0)
    elif "state_source_stock" in df.columns:
        stock_qty = _safe_numeric(df["state_source_stock"], 0.0)
    else:
        stock_qty = pd.Series([0.0] * len(df), index=df.index)

    safety_ss = _safe_numeric(
        df["safety_stock"] if "safety_stock" in df.columns
        else pd.Series([0.0] * len(df), index=df.index),
        0.0,
    )

    order_cost_col = _safe_numeric(
        df["order_cost"] if "order_cost" in df.columns
        else pd.Series([float(_DEFAULT_ORDER_COST)] * len(df), index=df.index),
        default=float(_DEFAULT_ORDER_COST),
    )
    order_cost_col = order_cost_col.where(order_cost_col > 0, _DEFAULT_ORDER_COST)

    # 보관비율
    if holding_rate is not None:
        h_rate = pd.Series([holding_rate] * len(df), index=df.index)
    elif "category" in df.columns:
        h_rate = _get_holding_rate(df["category"])
    elif "inventory_category" in df.columns:
        h_rate = _get_holding_rate(df["inventory_category"])
    else:
        h_rate = pd.Series([_DEFAULT_HOLDING_RATE] * len(df), index=df.index)

    # 연간 수요 & 단위 보관비
    annual_demand        = (daily_sale * 365).clip(lower=_MIN_ANNUAL_DEMAND)
    holding_cost_per_unit = unit_cost * h_rate

    # EOQ 계산
    eoq      = _calc_eoq(annual_demand, order_cost_col, holding_cost_per_unit)
    cycle    = _calc_order_cycle(eoq, daily_sale)
    tc       = _calc_annual_total_cost(annual_demand, eoq, order_cost_col, holding_cost_per_unit)
    cur_ord  = _calc_current_order_qty(stock_qty, safety_ss)
    risk     = _calc_eoq_risk_score(cur_ord, eoq)

    df["eoq_qty"]         = eoq
    df["eoq_order_cycle"] = cycle
    df["eoq_total_cost"]  = tc
    df["eoq_risk_score"]  = risk
    df["_current_order_qty"] = cur_ord

    df["eoq_status"]  = df.apply(_assign_status, axis=1)
    df["eoq_advice"]  = df.apply(_build_advice, axis=1)
    df["eoq_score"]   = risk   # varo_score 연동용

    df.drop(columns=["_current_order_qty"], inplace=True)

    return df


def get_eoq_summary(df: pd.DataFrame) -> pd.DataFrame:
    """EOQ 상태별 요약 테이블. 대시보드 표시용."""
    if df is None or df.empty or "eoq_status" not in df.columns:
        return pd.DataFrame()

    order = ["OVER", "OPTIMAL", "UNDER", "NO_DATA"]
    summary = (
        df.groupby("eoq_status")
        .agg(
            상품수=("product_name", "count"),
            평균EOQ=("eoq_qty", "mean"),
            평균위험점수=("eoq_risk_score", "mean"),
        )
        .reindex(order)
        .reset_index()
        .rename(columns={"eoq_status": "상태"})
    )
    summary["평균EOQ"]     = summary["평균EOQ"].round(1)
    summary["평균위험점수"] = summary["평균위험점수"].round(1)
    return summary
