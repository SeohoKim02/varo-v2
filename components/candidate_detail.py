"""Candidate judgment UI: 원본 위치 · 이동 수량 근거 · 제외 후보.

Reusable render helpers driven by the unified candidate ledger
(``services.candidate_ledger``). The basic screen stays short — a status line,
the quantity basis, and up to three reasons — while original-file locations and
full detail live in folded areas, per the operator-tool design rules.

Internal candidate ids, reason codes, file-system paths, and tracebacks are
never rendered here.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from services.candidate_lineage import source_reference_rows
from services.candidate_ledger import review_candidates_csv_bytes


def ledger_records(pipeline: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return list((pipeline or {}).get("candidate_ledger") or [])


def ledger_record(pipeline: Mapping[str, Any] | None, route_id: Any) -> dict[str, Any]:
    if route_id in (None, ""):
        return {}
    target = str(route_id)
    for record in ledger_records(pipeline):
        if str(record.get("route_id")) == target:
            return dict(record)
    return {}


def render_quantity_basis(st, record: Mapping[str, Any] | None) -> None:
    """One plain line explaining what limited the recommended quantity."""
    basis = (record or {}).get("quantity_basis") or {}
    text = basis.get("basis_text")
    if text:
        st.caption(text)


def render_source_locations(st, record: Mapping[str, Any] | None, expanded: bool = False) -> None:
    """Folded original-file locations behind the candidate (원본 위치 상세)."""
    references = (record or {}).get("source_references") or []
    if not references:
        return
    traceable = int((record or {}).get("traceable_row_count") or 0)
    label = f"원본 데이터 확인 (관련 행 {traceable}개)" if traceable else "원본 데이터 확인"
    with st.expander(label, expanded=expanded):
        rows = source_reference_rows(references)
        st.dataframe(_frame(rows), hide_index=True, width="stretch")
        st.caption("표시된 행 번호는 데이터 관리의 문제 목록과 같은 원본 위치입니다.")


def render_excluded_candidates(st, pipeline: Mapping[str, Any] | None, limit: int = 10) -> None:
    """Folded list of candidates that did not make the recommendation set.

    Shows only what a user needs to review the original data — never the full
    internal calculation. A UTF-8-BOM CSV is offered for real data fixing.
    """
    records = ledger_records(pipeline)
    excluded = [r for r in records if r.get("blocks_recommendation") or r.get("status") == "확인 필요"]
    if not excluded:
        return
    with st.expander(f"추천에서 제외된 후보 {len(excluded)}건", expanded=False):
        rows = []
        for record in excluded[:limit]:
            reference = _first_traceable(record.get("source_references") or [])
            rows.append({
                "출발 점포": record.get("source_name") or "-",
                "도착 점포": record.get("target_name") or "-",
                "상품": record.get("product_name") or "-",
                "상태": record.get("status") or "-",
                "가장 중요한 이유": record.get("short_reason") or "-",
                "원본 위치": _location_text(reference),
            })
        st.dataframe(_frame(rows), hide_index=True, width="stretch")
        if len(excluded) > limit:
            st.caption(f"전체 {len(excluded)}건 중 상위 {limit}건을 표시했습니다. 전체는 CSV로 확인하세요.")
        st.download_button(
            "제외 후보 검토 CSV",
            data=review_candidates_csv_bytes(records),
            file_name="varo_v2_후보검토.csv",
            mime="text/csv",
            width="stretch",
            key="dl_excluded_candidates_csv",
        )


def _first_traceable(references: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for ref in references:
        if ref.get("traceable"):
            return dict(ref)
    return dict(references[0]) if references else {}


def _location_text(reference: Mapping[str, Any]) -> str:
    if not reference or not reference.get("traceable"):
        return "추적 불가"
    rows = ", ".join(str(r) for r in reference.get("rows") or [])
    return f"{reference.get('sheet_name', '-')} {rows}행" if rows else str(reference.get("sheet_name", "-"))


def _frame(rows: list[dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(rows)
