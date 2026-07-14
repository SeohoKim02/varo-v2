"""
ABC 분석 (ABC Analysis)
───────────────────────────────────────────────────────────
매출액·재고가치를 기준으로 상품을 A/B/C 3등급으로 분류한다.

  A 등급 : 누적 매출 상위 80%  → 핵심 상품, 최우선 처리
  B 등급 : 누적 매출 80~95%   → 중요 상품, 우선 검토
  C 등급 : 누적 매출 하위 5%  → 저가치 상품, 후순위

입력 컬럼 (inventory DataFrame)
  필수 : product_name, sales_30d, unit_cost
  선택 : store (없으면 전체 집계), stock_qty (재고가치 fallback)

반환 컬럼 추가
  abc_revenue_value  : 매출가치 (sales_30d * unit_cost)
  abc_cumulative_pct : 누적 매출 비율 (0~1)
  abc_grade          : A / B / C
  abc_priority       : 1(A) / 2(B) / 3(C)  ← 정렬·점수 계산용
"""

import pandas as pd


# ─── 기준값 ───────────────────────────────────────────────
_A_THRESHOLD = 0.80   # 누적 80% 이내 → A
_B_THRESHOLD = 0.95   # 누적 80~95%   → B
                       # 나머지        → C

# 등급별 우선순위 점수 (heuristic/varo_score 연동용)
_GRADE_PRIORITY = {"A": 1, "B": 2, "C": 3}
_GRADE_SCORE    = {"A": 100, "B": 60, "C": 20}   # 0~100 점수


def _safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _calc_revenue_value(df: pd.DataFrame) -> pd.Series:
    """
    매출가치 = sales_30d × unit_cost
    컬럼명이 다른 경우 fallback 순서대로 시도
    """
    # 판매량 fallback 순서
    for col in ["sales_30d", "state_source_sales_30d", "source_sales_30d"]:
        if col in df.columns:
            sales = _safe_numeric(df[col])
            break
    else:
        # avg_daily_sales 있으면 × 30
        if "avg_daily_sales" in df.columns:
            sales = _safe_numeric(df["avg_daily_sales"]) * 30
        else:
            sales = pd.Series([0.0] * len(df), index=df.index)

    # 단가 fallback 순서
    for col in ["unit_cost", "state_unit_cost"]:
        if col in df.columns:
            cost = _safe_numeric(df[col])
            break
    else:
        cost = pd.Series([1.0] * len(df), index=df.index)

    return sales * cost


def analyze_abc(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    ABC 분석을 수행하고 등급 컬럼이 추가된 DataFrame을 반환한다.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        products / inventory 시트 또는 final_recommendations DataFrame.
        product_name, sales_30d, unit_cost 컬럼 필수.

    Returns
    -------
    pd.DataFrame
        원본에 abc_revenue_value, abc_cumulative_pct, abc_grade,
        abc_priority, abc_score 컬럼이 추가된 DataFrame.
    """
    if inventory_df is None or inventory_df.empty:
        return inventory_df

    df = inventory_df.copy()

    # 1. 매출가치 계산
    df["abc_revenue_value"] = _calc_revenue_value(df)

    # 2. 상품별 합산 (store 컬럼이 있으면 상품×점포 조합 유지)
    #    ABC는 '상품 수준' 집계가 기준이지만,
    #    Varo는 상품×점포 행을 유지해야 하므로 상품 집계값을 join 방식으로 추가한다.
    if "product_name" in df.columns:
        product_total = (
            df.groupby("product_name")["abc_revenue_value"]
            .sum()
            .reset_index()
            .rename(columns={"abc_revenue_value": "_abc_product_total"})
        )
        df = df.merge(product_total, on="product_name", how="left")
    else:
        df["_abc_product_total"] = df["abc_revenue_value"]

    # 3. 중복 제거된 상품 목록으로 누적 비율 계산
    # product_rank는 product_name별 1행 → 여기서 sum해야 분모가 올바름
    # (df 전체로 sum하면 상품이 여러 점포에 중복 등장해 분모가 부풀려짐)
    product_rank = (
        df[["product_name", "_abc_product_total"]]
        .drop_duplicates("product_name")
        .sort_values("_abc_product_total", ascending=False)
        .reset_index(drop=True)
    )

    total_revenue = product_rank["_abc_product_total"].sum()

    if total_revenue == 0 or pd.isna(total_revenue):
        df["abc_cumulative_pct"] = 0.0
        df["abc_grade"]          = "C"
        df["abc_priority"]       = 3
        df["abc_score"]          = 20.0
        df.drop(columns=["_abc_product_total"], inplace=True)
        return df

    product_rank["_cumsum"]      = product_rank["_abc_product_total"].cumsum()
    product_rank["_cum_pct"]     = product_rank["_cumsum"] / total_revenue
    product_rank["abc_grade"]    = product_rank["_cum_pct"].apply(_assign_grade)
    product_rank["abc_cumulative_pct"] = product_rank["_cum_pct"]

    grade_map    = product_rank.set_index("product_name")["abc_grade"].to_dict()
    cum_pct_map  = product_rank.set_index("product_name")["abc_cumulative_pct"].to_dict()

    df["abc_grade"]          = df["product_name"].map(grade_map).fillna("C")
    df["abc_cumulative_pct"] = df["product_name"].map(cum_pct_map).fillna(1.0)
    df["abc_priority"]       = df["abc_grade"].map(_GRADE_PRIORITY)
    df["abc_score"]          = df["abc_grade"].map(_GRADE_SCORE).astype(float)

    df.drop(columns=["_abc_product_total"], inplace=True)

    return df


def _assign_grade(cum_pct: float) -> str:
    if cum_pct <= _A_THRESHOLD:
        return "A"
    if cum_pct <= _B_THRESHOLD:
        return "B"
    return "C"


def get_abc_summary(abc_df: pd.DataFrame) -> pd.DataFrame:
    """
    ABC 등급별 요약 테이블을 반환한다.
    대시보드 표시용.
    """
    if abc_df is None or abc_df.empty or "abc_grade" not in abc_df.columns:
        return pd.DataFrame()

    summary = (
        abc_df.groupby("abc_grade")
        .agg(
            상품수=("product_name", "nunique"),
            총매출가치=("abc_revenue_value", "sum"),
        )
        .reindex(["A", "B", "C"])
        .reset_index()
        .rename(columns={"abc_grade": "등급"})
    )

    total = summary["총매출가치"].sum()
    summary["매출비율(%)"] = (summary["총매출가치"] / total * 100).round(1)

    return summary
