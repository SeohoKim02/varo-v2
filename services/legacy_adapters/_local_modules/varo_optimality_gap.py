"""
varo_optimality_gap.py
───────────────────────
Varo 추천 결과 vs 비용 최소화 최적화 결과 비교 모듈.

- Varo 최종 추천을 대체하지 않음 — 검증용 비교 기능
- 소규모 후보 집합(상위 30개) 기준 계산
- scipy.optimize.milp → brute-force(≤15개) → fallback 순서
- 다운로드 버튼 없음
"""

import math
import itertools
import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_CANDIDATES = 30
BRUTE_FORCE_MAX = 15

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _sn(v, d=0.0):
    try:
        f = float(str(v).replace(",","").replace("원","").replace("%","").strip())
        return d if (math.isnan(f) or math.isinf(f)) else f
    except: return d

def _first(row, *names, default=0.0):
    for n in names:
        try:
            v = row.get(n) if hasattr(row,'get') else None
            if v is not None and not (isinstance(v,float) and math.isnan(v)):
                return v
        except: pass
    return default

def parse_cost_value(value) -> float:
    return _sn(value, 0.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 비용 함수 Cost(i)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calculate_candidate_cost(row) -> float:
    """
    Cost(i) = 이동비용 + 할인손실 + 프로모션비용
              - 폐기회피효과 - 프로모션순효과(양수시)
    우선순위: estimated_cost → 기타 비용 가감
    """
    base   = _sn(_first(row,"estimated_cost","transport_cost"), 0)
    loss   = _sn(_first(row,"discount_loss_cost","discount_loss"), 0)
    promo  = _sn(_first(row,"promotion_fixed_cost"), 0)
    avoid  = _sn(_first(row,"avoided_disposal_cost"), 0)
    net_b  = _sn(_first(row,"promotion_net_benefit"), 0)

    cost = base + loss + promo - avoid
    if net_b > 0:
        cost -= net_b

    # 수량 보정: move_qty가 있으면 단위 비용 × 수량으로 스케일 확인
    # (estimated_cost는 이미 총 비용이므로 추가 곱하지 않음)
    return max(0.0, round(cost, 2))


def add_optimization_cost_column(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 전체에 opt_cost 컬럼 추가 (원본 미수정)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["opt_cost"] = [calculate_candidate_cost(row) for _, row in out.iterrows()]
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 최적화 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _run_milp(costs: np.ndarray, k: int) -> tuple:
    """
    scipy.optimize.milp 기반 0-1 선택 최적화.
    반환: (선택 인덱스 배열, 총 비용, 상태 메시지)
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        n = len(costs)
        k = min(k, n)
        # 목적함수: minimize Σ cost_i * x_i
        c_vec = costs.astype(float)
        # 제약: Σ x_i = k
        A = np.ones((1, n))
        constraints = LinearConstraint(A, lb=k, ub=k)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        integrality = np.ones(n)  # 모든 변수 정수
        res = milp(c_vec, constraints=constraints,
                   integrality=integrality, bounds=bounds)
        if res.success:
            selected = np.where(res.x > 0.5)[0]
            total = float(costs[selected].sum())
            return selected, total, "milp"
        return None, None, f"milp_fail:{res.message}"
    except Exception as e:
        return None, None, f"milp_error:{e}"


def _run_brute_force(costs: np.ndarray, k: int) -> tuple:
    """완전탐색 — 후보 수 ≤ BRUTE_FORCE_MAX 일 때만 허용."""
    n = len(costs)
    k = min(k, n)
    if n > BRUTE_FORCE_MAX:
        return None, None, "brute_force_skip:too_many"
    best_idx, best_cost = None, float("inf")
    for combo in itertools.combinations(range(n), k):
        c = float(costs[list(combo)].sum())
        if c < best_cost:
            best_cost = c
            best_idx  = np.array(combo)
    return best_idx, best_cost, "brute_force"


def _fallback_greedy(costs: np.ndarray, k: int) -> tuple:
    """단순 비용 낮은 순 k개 선택 (greedy approximation)."""
    n = len(costs)
    k = min(k, n)
    selected = np.argsort(costs)[:k]
    total = float(costs[selected].sum())
    return selected, total, "greedy_approx"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 메인: Optimality Gap 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calculate_optimality_gap(
    final_recommendations: pd.DataFrame,
    k: int = 5,
    max_candidates: int = MAX_CANDIDATES,
) -> dict:
    """
    Varo TOP-K 추천 vs 비용 최소화 결과 비교.
    반환 dict:
      varo_total, opt_total, cost_diff, gap_pct, gap_str,
      varo_idx, opt_idx, common_count, match_rate,
      opt_method, candidates_used, df_with_cost
    """
    empty = {
        "varo_total": None, "opt_total": None, "cost_diff": None,
        "gap_pct": None, "gap_str": "계산 불가", "varo_idx": [],
        "opt_idx": [], "common_count": 0, "match_rate": 0,
        "opt_method": "-", "candidates_used": 0,
        "df_with_cost": pd.DataFrame(),
    }

    if final_recommendations is None or final_recommendations.empty:
        empty["gap_str"] = "비교 대상 없음"
        return empty

    # 비용 컬럼 추가
    df = add_optimization_cost_column(final_recommendations)

    # 유효 후보 필터 (수량 > 0)
    qty_col = next((c for c in ["suggested_qty","move_qty","recommended_qty"]
                    if c in df.columns), None)
    if qty_col:
        df = df[pd.to_numeric(df[qty_col], errors="coerce").fillna(0) > 0]

    if df.empty:
        empty["gap_str"] = "유효 후보 없음"
        return empty

    # 후보 수 제한
    score_col = next((c for c in ["vhs2","heuristic_score","total_score"]
                      if c in df.columns), None)
    if score_col:
        df = df.nlargest(max_candidates, score_col)
    else:
        df = df.head(max_candidates)

    df = df.reset_index(drop=True)
    n  = len(df)
    k  = min(k, n)
    costs = df["opt_cost"].to_numpy(dtype=float)

    # Varo TOP-K (이미 score 기준 정렬된 상위 k개)
    varo_idx = np.arange(min(k, n))
    varo_total = float(costs[varo_idx].sum()) if len(varo_idx) > 0 else 0.0

    # 최적화 실행 (우선순위: milp → brute_force → greedy_approx)
    opt_idx, opt_total, method = _run_milp(costs, k)
    if opt_idx is None:
        opt_idx, opt_total, method = _run_brute_force(costs, k)
    if opt_idx is None:
        opt_idx, opt_total, method = _fallback_greedy(costs, k)
    if opt_idx is None or opt_total is None:
        empty["gap_str"] = "최적화 계산 실패"
        empty["df_with_cost"] = df
        return empty

    opt_idx = np.array(opt_idx, dtype=int)

    # Gap 계산
    cost_diff = varo_total - opt_total
    if opt_total > 0:
        gap_pct = round(cost_diff / opt_total * 100, 2)
        gap_str = f"{gap_pct:.1f}%"
    elif opt_total == 0 and cost_diff == 0:
        gap_pct = 0.0
        gap_str = "0.0%"
    else:
        gap_pct = None
        gap_str = "계산 불가 (최적 비용=0)"

    common = set(varo_idx.tolist()) & set(opt_idx.tolist())
    match_rate = round(len(common) / k * 100, 1) if k > 0 else 0.0

    return {
        "varo_total":       round(varo_total, 0),
        "opt_total":        round(opt_total,  0),
        "cost_diff":        round(cost_diff,  0),
        "gap_pct":          gap_pct,
        "gap_str":          gap_str,
        "varo_idx":         varo_idx.tolist(),
        "opt_idx":          opt_idx.tolist(),
        "common_count":     len(common),
        "match_rate":       match_rate,
        "opt_method":       method,
        "candidates_used":  n,
        "df_with_cost":     df,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 표시용 DataFrame
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_optimality_comparison_table(gap_result: dict) -> pd.DataFrame:
    """선택 후보 비교 표."""
    df = gap_result.get("df_with_cost", pd.DataFrame())
    if df.empty:
        return pd.DataFrame()

    varo_set = set(gap_result.get("varo_idx", []))
    opt_set  = set(gap_result.get("opt_idx",  []))

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        in_varo = i in varo_set
        in_opt  = i in opt_set
        if not (in_varo or in_opt):
            continue
        if in_varo and in_opt:
            label = "공통"
        elif in_varo:
            label = "Varo"
        else:
            label = "최적화"

        rows.append({
            "구분":      label,
            "상품명":    row.get("product_name", "-"),
            "보내는 점포":row.get("source_store", "-"),
            "받는 점포": row.get("target_store",  "-"),
            "추천 수량": row.get("suggested_qty") or row.get("move_qty") or row.get("recommended_qty", "-"),
            "추천 전략": str(row.get("final_recommendation","") or row.get("vhs2_action","") or "-")[:15],
            "비용":      f'{row.get("opt_cost", 0):,.0f}원',
            "Hybrid Score": round(_sn(row.get("vhs2") or row.get("heuristic_score"), 0), 1),
        })

    return pd.DataFrame(rows).sort_values(["구분","Hybrid Score"],
                                           ascending=[True,False],
                                           ignore_index=True)
