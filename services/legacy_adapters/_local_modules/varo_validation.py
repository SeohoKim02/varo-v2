"""
varo_validation.py
───────────────────
Varo 분석 결과 검증 리포트 모듈.

- 기존 추천 결과를 수정하지 않고 요약·점검
- 전체 분석 파이프라인 상태를 빠르게 확인
- 상태: 정상 / 확인 필요 / 데이터 부족 / 오류 가능
"""

import math
import io
import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _sn(v, d=0.0):
    try:
        f = float(v)
        return d if (math.isnan(f) or math.isinf(f)) else f
    except: return d

def _cnt(df, col, val=None):
    if df is None or df.empty or col not in df.columns: return 0
    if val is None: return int(df[col].notna().sum())
    return int((df[col] == val).sum())

def _col_any(df, *names):
    if df is None: return None
    for n in names:
        if n in df.columns: return n
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 핵심 검증 메트릭 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calculate_validation_metrics(
    final_recommendations: pd.DataFrame,
    stores:      pd.DataFrame = None,
    products:    pd.DataFrame = None,
    inventory:   pd.DataFrame = None,
) -> dict:
    fr = final_recommendations
    m  = {}

    # ── 데이터 규모 ──────────────────────────────────────
    m["n_stores"]    = len(stores)    if stores    is not None else 0
    m["n_products"]  = len(products)  if products  is not None else 0
    m["n_inventory"] = len(inventory) if inventory is not None else 0
    m["n_recs"]      = len(fr) if fr is not None and not fr.empty else 0

    if fr is None or fr.empty:
        m["status"] = "오류 가능"
        return m

    # ── 점수 요약 ────────────────────────────────────────
    score_col = _col_any(fr, "vhs2", "heuristic_score", "total_score")
    if score_col:
        sc = pd.to_numeric(fr[score_col], errors="coerce").dropna()
        m["avg_score"]  = round(float(sc.mean()), 1) if not sc.empty else 0
        m["max_score"]  = round(float(sc.max()),  1) if not sc.empty else 0
        m["min_score"]  = round(float(sc.min()),  1) if not sc.empty else 0
        m["score_nan"]  = int(pd.to_numeric(fr[score_col], errors="coerce").isna().sum())
    else:
        m["avg_score"] = m["max_score"] = m["min_score"] = m["score_nan"] = 0

    # ── 추천 등급 ─────────────────────────────────────────
    grade_col = _col_any(fr, "vhs2_grade","heuristic_grade","추천 등급","recommendation_grade")
    for g in ["최적","권장","검토","보류"]:
        m[f"grade_{g}"] = _cnt(fr, grade_col, g) if grade_col else 0

    # ── 전략별 카운트 ─────────────────────────────────────
    rec_col = _col_any(fr, "final_recommendation","추천 전략","vhs2_action")
    def _strategy_cnt(kws):
        if not rec_col: return 0
        return int(fr[rec_col].astype(str).apply(lambda x: any(k in x for k in kws)).sum())
    m["n_transfer"]  = _strategy_cnt(["이동","재배치","transfer"])
    m["n_discount"]  = _strategy_cnt(["할인","discount"])
    m["n_emergency"] = _strategy_cnt(["긴급"])
    m["n_one_plus"]  = _strategy_cnt(["1+1","one_plus"])
    m["n_dispose"]   = _strategy_cnt(["폐기","dispose"])
    m["n_hold"]      = _strategy_cnt(["보류","검토","hold"])

    # ── 신뢰도 ───────────────────────────────────────────
    for lvl in ["높음","보통","낮음"]:
        m[f"conf_{lvl}"] = _cnt(fr, "confidence_level", lvl)
    if "confidence_score" in fr.columns:
        top5 = fr.nlargest(5, score_col)["confidence_score"] if score_col else \
               pd.to_numeric(fr["confidence_score"], errors="coerce").head(5)
        m["top5_avg_conf"] = round(float(pd.to_numeric(top5, errors="coerce").mean()), 1)
    else:
        m["top5_avg_conf"] = 0

    # ── DQN ──────────────────────────────────────────────
    m["dqn_status"] = fr["dqn_status"].iloc[0] if "dqn_status" in fr.columns and not fr.empty else "데이터 없음"
    for ag in ["전방위 일치","DQN-Varo 일치","DQN 부분 일치","불일치","비교 불가"]:
        m[f"agree_{ag}"] = _cnt(fr, "agreement_status", ag)

    # ── 수요 ─────────────────────────────────────────────
    for ds in ["높음","보통","낮음","데이터 없음"]:
        m[f"demand_{ds}"] = _cnt(fr, "demand_status", ds)

    # ── 프로모션 ──────────────────────────────────────────
    for ps in ["유리","보통","비추천","데이터 부족"]:
        m[f"promo_{ps}"] = _cnt(fr, "promotion_status", ps)
    if "avoided_disposal_cost" in fr.columns:
        m["total_avoided_disposal"] = round(
            float(pd.to_numeric(fr["avoided_disposal_cost"], errors="coerce").sum()), 0)
    else:
        m["total_avoided_disposal"] = 0

    # ── 데이터 품질 ───────────────────────────────────────
    qty_col = _col_any(fr, "suggested_qty","move_qty","recommended_qty","transfer_qty")
    cost_col = _col_any(fr, "estimated_cost")
    m["qty_zero"]       = int((pd.to_numeric(fr[qty_col], errors="coerce").fillna(0) == 0).sum()) if qty_col else 0
    m["cost_missing"]   = int(pd.to_numeric(fr[cost_col], errors="coerce").isna().sum()) if cost_col else 0
    m["demand_missing"] = _cnt(fr, "demand_status", "데이터 없음")
    m["promo_missing"]  = _cnt(fr, "promotion_status", "데이터 부족")
    m["dqn_incomparable"] = _cnt(fr, "agreement_status", "비교 불가")
    # 좌표 없는 점포
    m["coord_missing"] = 0
    if stores is not None and not stores.empty:
        lat_col = _col_any(stores, "latitude","lat","위도")
        lng_col = _col_any(stores, "longitude","lng","경도")
        if lat_col and lng_col:
            bad = (pd.to_numeric(stores[lat_col], errors="coerce").isna() |
                   pd.to_numeric(stores[lng_col], errors="coerce").isna())
            m["coord_missing"] = int(bad.sum())

    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 검증 상태 분류
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def classify_validation_status(m: dict) -> str:
    if m.get("n_recs", 0) == 0:
        return "오류 가능"
    if m.get("score_nan", 0) > m.get("n_recs", 1) * 0.5:
        return "오류 가능"
    data_missing = (m.get("demand_missing", 0) + m.get("promo_missing", 0))
    if data_missing > m.get("n_recs", 1) * 0.7:
        return "데이터 부족"
    warn_items = (
        int(m.get("qty_zero", 0) > 0) +
        int(m.get("cost_missing", 0) > 0) +
        int(m.get("conf_낮음", 0) > m.get("n_recs", 1) * 0.3) +
        int(m.get("coord_missing", 0) > 0) +
        int(m.get("dqn_status", "") in ("제외","데이터 없음"))
    )
    if warn_items >= 3:
        return "확인 필요"
    return "정상"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 경고 목록 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_validation_warnings(m: dict) -> list:
    warnings = []
    if m.get("n_recs", 0) == 0:
        warnings.append("최종 추천 후보가 없습니다.")
    if m.get("qty_zero", 0) > 0:
        warnings.append(f"추천 수량 0인 후보 {m['qty_zero']}건이 있습니다.")
    if m.get("cost_missing", 0) > 0:
        warnings.append(f"비용 계산이 누락된 후보 {m['cost_missing']}건이 있습니다.")
    if m.get("coord_missing", 0) > 0:
        warnings.append(f"좌표가 없는 점포 {m['coord_missing']}개가 있습니다.")
    if m.get("demand_missing", 0) > 0:
        warnings.append(f"수요 데이터가 부족한 후보 {m['demand_missing']}건이 있습니다.")
    if m.get("promo_missing", 0) > 0:
        warnings.append(f"프로모션 계산이 불완전한 후보 {m['promo_missing']}건이 있습니다.")
    if m.get("dqn_status", "") in ("제외",):
        warnings.append("DQN 결과가 제외 상태입니다.")
    if m.get("score_nan", 0) > 0:
        warnings.append(f"Hybrid Score가 누락된 후보 {m['score_nan']}건이 있습니다.")
    return warnings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 통합 리포트 빌드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_validation_report(
    final_recommendations: pd.DataFrame,
    stores:   pd.DataFrame = None,
    products: pd.DataFrame = None,
    inventory:pd.DataFrame = None,
) -> dict:
    """전체 검증 리포트 dict 반환."""
    m        = calculate_validation_metrics(final_recommendations, stores, products, inventory)
    status   = classify_validation_status(m)
    warnings = build_validation_warnings(m)
    m["status"]        = status
    m["warnings"]      = warnings
    m["warning_count"] = len(warnings)
    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 화면 표시용 DataFrame 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_validation_summary_df(report: dict) -> pd.DataFrame:
    m = report
    rows = [
        ("전체 후보 수",     m.get("n_recs",0),          "✅" if m.get("n_recs",0)>0 else "❌", ""),
        ("평균 Hybrid Score",m.get("avg_score",0),       "✅" if m.get("avg_score",0)>50 else "⚠️",""),
        ("신뢰도 높음 후보", m.get("conf_높음",0),        "✅",""),
        ("신뢰도 낮음 후보", m.get("conf_낮음",0),        "⚠️" if m.get("conf_낮음",0)>0 else "✅",""),
        ("DQN 상태",         m.get("dqn_status","없음"),  "✅" if m.get("dqn_status")=="정상" else "⚠️",""),
        ("수요 높음 후보",   m.get("demand_높음",0),      "✅",""),
        ("프로모션 유리 후보",m.get("promo_유리",0),      "✅",""),
        ("추천 수량 0 건수", m.get("qty_zero",0),         "❌" if m.get("qty_zero",0)>0 else "✅",""),
        ("좌표 누락 점포",   m.get("coord_missing",0),    "⚠️" if m.get("coord_missing",0)>0 else "✅","지도 표시 영향"),
        ("경고 건수",        m.get("warning_count",0),    "⚠️" if m.get("warning_count",0)>0 else "✅",""),
    ]
    return pd.DataFrame(rows, columns=["항목","값","상태","메모"])


def get_data_quality_df(report: dict) -> pd.DataFrame:
    m = report
    def _st(cnt, thr=0):
        return "⚠️" if cnt > thr else "✅"
    rows = [
        ("추천 수량 0",         m.get("qty_zero",0),         _st(m.get("qty_zero",0)),        "0 → 수량 재확인"),
        ("비용 누락",           m.get("cost_missing",0),     _st(m.get("cost_missing",0)),    "비용 계산 불가"),
        ("좌표 누락 점포",      m.get("coord_missing",0),    _st(m.get("coord_missing",0)),   "지도 미표시"),
        ("수요 데이터 없음",    m.get("demand_missing",0),   _st(m.get("demand_missing",0)),  "fallback 적용"),
        ("프로모션 데이터 부족",m.get("promo_missing",0),    _st(m.get("promo_missing",0)),   "fallback 적용"),
        ("DQN 비교 불가",       m.get("dqn_incomparable",0), _st(m.get("dqn_incomparable",0)),"모델 미로딩 가능"),
        ("Score NaN 건수",      m.get("score_nan",0),        _st(m.get("score_nan",0)),       "계산 오류 확인"),
    ]
    return pd.DataFrame(rows, columns=["점검 항목","문제 건수","상태","조치 기준"])


def get_top5_validation_df(final_recommendations: pd.DataFrame) -> pd.DataFrame:
    if final_recommendations is None or final_recommendations.empty:
        return pd.DataFrame()
    score_col = _col_any(final_recommendations,
                         "vhs2","heuristic_score","total_score","confidence_score")
    if score_col:
        top5 = final_recommendations.nlargest(5, score_col).copy()
    else:
        top5 = final_recommendations.head(5).copy()
    top5.insert(0, "순위", range(1, len(top5)+1))
    show = {
        "순위":              "순위",
        "product_name":      "상품명",
        "source_store":      "보내는 점포",
        "target_store":      "받는 점포",
        "suggested_qty":     "추천 수량",
        "final_recommendation":"추천 전략",
        "vhs2":              "Hybrid Score",
        "vhs2_grade":        "추천 등급",
        "confidence_level":  "신뢰도",
        "demand_status":     "수요 수준",
        "promotion_status":  "프로모션 상태",
        "dqn_status":        "DQN 상태",
    }
    avail = [k for k in show if k in top5.columns]
    return top5[avail].rename(columns=show).reset_index(drop=True)


def get_recommendation_summary_df(final_recommendations: pd.DataFrame) -> pd.DataFrame:
    fr = final_recommendations
    if fr is None or fr.empty:
        return pd.DataFrame()
    score_col = _col_any(fr,"vhs2","heuristic_score","total_score")
    conf_col  = _col_any(fr,"confidence_score")
    rec_col   = _col_any(fr,"final_recommendation","vhs2_action")

    def _make_row(label, mask):
        sub = fr[mask]
        if sub.empty:
            return {"구분": label, "후보 수": 0, "평균 점수": "-", "평균 신뢰도": "-", "주요 전략": "-"}
        sc  = f'{pd.to_numeric(sub[score_col], errors="coerce").mean():.1f}' if score_col else "-"
        cf  = f'{pd.to_numeric(sub[conf_col],  errors="coerce").mean():.1f}' if conf_col  else "-"
        top = sub[rec_col].value_counts().index[0] if rec_col and not sub[rec_col].empty else "-"
        return {"구분": label, "후보 수": len(sub), "평균 점수": sc, "평균 신뢰도": cf, "주요 전략": str(top)[:15]}

    def _mask(kws):
        if not rec_col: return pd.Series([False]*len(fr), index=fr.index)
        return fr[rec_col].astype(str).apply(lambda x: any(k in x for k in kws))

    rows = [
        _make_row("이동 추천",    _mask(["이동","재배치"])),
        _make_row("할인 추천",    _mask(["할인","프로모션","1+1"])),
        _make_row("긴급 할인",    _mask(["긴급"])),
        _make_row("폐기 위험",    _mask(["폐기"])),
        _make_row("보류/검토",    _mask(["보류","검토"])),
        _make_row("전체",         pd.Series([True]*len(fr), index=fr.index)),
    ]
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 다운로드용 Excel (BytesIO)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_validation_excel(
    report: dict,
    final_recommendations: pd.DataFrame,
) -> bytes:
    """
    다중 시트 Excel BytesIO 반환.
    openpyxl 없으면 None 반환 (CSV fallback).
    """
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            get_validation_summary_df(report).to_excel(writer, sheet_name="검증 요약", index=False)
            get_data_quality_df(report).to_excel(writer, sheet_name="데이터 품질", index=False)
            get_top5_validation_df(final_recommendations).to_excel(writer, sheet_name="TOP5", index=False)
            get_recommendation_summary_df(final_recommendations).to_excel(writer, sheet_name="추천 요약", index=False)
            # 경고 목록
            warns = report.get("warnings", [])
            pd.DataFrame({"경고": warns}).to_excel(writer, sheet_name="경고 목록", index=False)
        return buf.getvalue()
    except Exception:
        return None


def build_validation_csv(report: dict, final_recommendations: pd.DataFrame) -> bytes:
    """단일 CSV 다운로드용."""
    try:
        parts = [
            ("=== 검증 요약 ===", get_validation_summary_df(report)),
            ("=== TOP5 ===",      get_top5_validation_df(final_recommendations)),
            ("=== 데이터 품질 ===",get_data_quality_df(report)),
            ("=== 추천 요약 ===", get_recommendation_summary_df(final_recommendations)),
        ]
        lines = []
        for title, df in parts:
            lines.append(title)
            if not df.empty:
                lines.append(df.to_csv(index=False))
            lines.append("")
        return "\n".join(lines).encode("utf-8-sig")
    except Exception:
        return b""
