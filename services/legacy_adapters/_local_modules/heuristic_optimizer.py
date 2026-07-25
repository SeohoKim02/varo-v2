import pandas as pd


def _to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _normalize_lower_is_better(series, max_score):
    numeric = _to_numeric(series)

    if numeric.notna().sum() == 0:
        return pd.Series([max_score * 0.5] * len(series), index=series.index)

    min_value = numeric.min()
    max_value = numeric.max()

    if min_value == max_value:
        return pd.Series([max_score] * len(series), index=series.index)

    score = (max_value - numeric) / (max_value - min_value) * max_score
    return score.fillna(max_score * 0.4)


def _normalize_higher_is_better(series, max_score):
    numeric = _to_numeric(series)

    if numeric.notna().sum() == 0:
        return pd.Series([max_score * 0.5] * len(series), index=series.index)

    min_value = numeric.min()
    max_value = numeric.max()

    if min_value == max_value:
        return pd.Series([max_score] * len(series), index=series.index)

    score = (numeric - min_value) / (max_value - min_value) * max_score
    return score.fillna(max_score * 0.4)


def _strategy_score(value):
    text = str(value)

    if "다중" in text:
        return 18
    if "직접" in text:
        return 20
    if "DC" in text or "경유" in text:
        return 16
    if "재배치" in text or "이동" in text:
        return 18
    if "프로모션" in text or "할인" in text:
        return 12
    if "폐기" in text:
        return 5

    return 10


def _reason_bonus(value):
    text = str(value)
    bonus = 0

    if "비용" in text and ("낮" in text or "저렴" in text or "절감" in text):
        bonus += 8

    if "판매" in text or "수요" in text:
        bonus += 5

    if "거리" in text or "경로" in text:
        bonus += 4

    if "가능" in text:
        bonus += 3

    return min(bonus, 12)


def _grade(score):
    if score >= 85:
        return "최우선 추천"
    if score >= 70:
        return "우선 추천"
    if score >= 55:
        return "검토 가능"
    return "후순위"


def add_heuristic_scores(final_recommendations):
    """
    최종 추천 후보에 휴리스틱 점수를 추가한다.

    점수 구성:
    - 비용 점수: 낮을수록 높음
    - 수량 점수: 처리 수량이 많을수록 높음
    - 전략 점수: 이동/재배치 전략에 가중치
    - 이유 보너스: 비용 절감, 수요, 거리 등 키워드 반영
    """

    if final_recommendations is None or final_recommendations.empty:
        return final_recommendations

    result = final_recommendations.copy()

    if "estimated_cost" not in result.columns:
        result["estimated_cost"] = 0

    if "suggested_qty" not in result.columns:
        result["suggested_qty"] = 0

    if "final_recommendation" not in result.columns:
        result["final_recommendation"] = ""

    if "reason" not in result.columns:
        result["reason"] = ""

    result["cost_score"] = _normalize_lower_is_better(
        result["estimated_cost"],
        max_score=40
    )

    result["quantity_score"] = _normalize_higher_is_better(
        result["suggested_qty"],
        max_score=25
    )

    result["strategy_score"] = result["final_recommendation"].apply(_strategy_score)
    result["reason_bonus"] = result["reason"].apply(_reason_bonus)

    result["heuristic_score"] = (
        result["cost_score"]
        + result["quantity_score"]
        + result["strategy_score"]
        + result["reason_bonus"]
    ).round(1)

    result["heuristic_grade"] = result["heuristic_score"].apply(_grade)

    result["_estimated_cost_numeric"] = _to_numeric(result["estimated_cost"])
    result["_suggested_qty_numeric"] = _to_numeric(result["suggested_qty"])

    result = result.sort_values(
        by=[
            "heuristic_score",
            "_estimated_cost_numeric",
            "_suggested_qty_numeric",
        ],
        ascending=[False, True, False]
    ).reset_index(drop=True)

    result["greedy_rank"] = result.index + 1
    result["is_greedy_selected"] = result["greedy_rank"] == 1

    result["greedy_reason"] = result.apply(
        lambda row: (
            "휴리스틱 점수가 가장 높아 Greedy 알고리즘에 의해 최적 후보로 선택됨"
            if row["is_greedy_selected"]
            else "Greedy 선택 후보보다 우선순위가 낮음"
        ),
        axis=1
    )

    return result


def select_greedy_best_candidate(final_recommendations):
    if final_recommendations is None or final_recommendations.empty:
        return None

    if "is_greedy_selected" in final_recommendations.columns:
        selected = final_recommendations[
            final_recommendations["is_greedy_selected"] == True
        ]

        if not selected.empty:
            return selected.iloc[0]

    if "heuristic_score" in final_recommendations.columns:
        return final_recommendations.sort_values(
            by="heuristic_score",
            ascending=False
        ).iloc[0]

    return final_recommendations.iloc[0]
