"""Opt-in research controls. No model/history/database persistence."""
import json
import logging
import pandas as pd
import streamlit as st

from components.user_feedback import report_error
from services.research_information import prioritize_information, seller_scenario_example

logger = logging.getLogger(__name__)


def render_research_upgrade():
    st.subheader("추가 정보를 확인할 가치")
    st.caption("연구용 가상 비교입니다. 기존 상품 추천이나 실행 이력을 변경하지 않습니다.")
    source = st.radio("검토 자료", ["판매자 손실 가상 사례", "준비한 연구 입력"], key="research_information_source", horizontal=True)
    payload = None
    if source == "판매자 손실 가상 사례":
        try:
            payload = seller_scenario_example()
            st.info(payload["assumptions"])
        except Exception as exc:
            report_error("research seller scenarios", exc, "가상 사례를 계산하지 못했습니다. 다시 실행하거나 관리자에게 확인해 주세요.")
    else:
        uploaded = st.file_uploader("연구 입력 파일", type=["json"], key="research_information_upload")
        st.caption("같은 수량·기간·통화의 대안별 결과와 확인 비용이 필요합니다. 자료가 없으면 실제 금액으로 계산하지 않습니다.")
        if uploaded is not None:
            try:
                raw = uploaded.getvalue()
                if len(raw) > 2_000_000:
                    raise ValueError("Research input exceeds 2 MB")
                payload = json.loads(raw)
                if not isinstance(payload, dict) or payload.get("evidence_class") != "SCENARIO":
                    raise ValueError("Explicit scenario label required")
            except Exception as exc:
                report_error("research information input", exc, "연구 입력을 읽지 못했습니다. 2MB 이하의 가상 사례 파일인지 확인해 주세요.")
    if payload is not None:
        try:
            result = prioritize_information(payload["payoffs"], payload["observations"], payload["costs"], payload.get("probabilities"))
            expected = result["mode"] == "EXPECTED_INFORMATION"
            st.write("입력한 확률 기준의 정보 가치" if expected else "최대 선택 후회 감소 기준")
            st.caption("후회는 나중에 상황을 알았을 때 더 나은 선택과 벌어지는 손실 차이입니다. 주어진 가상 범위에서만 계산하며 실제 절감액이 아닙니다.")
            frame = pd.DataFrame([{"확인할 정보": r["query"], "확인 전 비용을 제외한 가치 상한": r["gross_value_bound"],
                                   "가정한 확인 비용": r["collection_cost"], "비용 차감 후 가치 상한": r["net_value_bound"]}
                                  for r in result["queries"]])
            st.dataframe(frame, hide_index=True, width="stretch")
            if result["recommended_query"]:
                st.write(f"먼저 검토할 정보: {result['recommended_query']}")
            else:
                st.write("현재 가정에서는 확인 비용을 넘는 정보 가치가 없습니다.")
            st.download_button("정보 우선순위 CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                               "Varo_research_information.csv", "text/csv", key="research_information_download")
            if source == "판매자 손실 가상 사례":
                with st.expander("가상 입력과 대안별 손실"):
                    labels = {"TRANSFER":"4개 이동·16개 정상판매", "NORMAL_SALE":"20개 정상판매", "DISCOUNT_SALE":"20개 할인판매"}
                    st.dataframe(pd.DataFrame([{"출발 수요": r["source_daily_demand"], "도착 수요": r["target_daily_demand"],
                        "가정 이동비": r["transfer_cost"], "가정 할인 반응": r["promotion_uplift"],
                        **{labels[k]:v for k,v in r["losses"].items()}} for r in payload["records"]]), hide_index=True, width="stretch")
        except Exception as exc:
            report_error("research information calculation", exc, "대안별 결과를 비교할 수 없습니다. 모든 상황에 같은 대안이 있고, 비용과 확률이 올바른지 확인해 주세요.")

    st.subheader("외부 판매 패턴 실험 결과")
    st.caption("시간 순서로 나눈 연구 보고서를 확인합니다. 실제 재고·비용 실적과 구분하며 운영 방식을 자동 변경하지 않습니다.")
    uploaded = st.file_uploader("완료된 실험 보고서", type=["json"], key="research_external_upload")
    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            if len(raw)>2_000_000: raise ValueError("Research report exceeds 2 MB")
            report=json.loads(raw)
            if report.get("schema_version")!=1 or report.get("evidence_class")!="OBSERVED_SALES_SCENARIO_OPERATIONS":
                raise ValueError("Unsupported report schema")
            names={"ALWAYS":"매일 재검토", "PERIODIC_2":"이틀마다 재검토", "THRESHOLD":"변경 이득이 비용을 넘을 때", "HISTORICAL_SELECTOR":"과거 자료로 고른 방식"}
            rows=[]
            for dataset, data in report["by_dataset"].items():
                for method, metrics in data["overall"]["mean"].items():
                    if method not in names: raise ValueError("Unsupported policy")
                    rows.append({"자료":dataset,"검토 방식":names[method],"가상 순가치 평균":float(metrics["net_value"]),
                                 "계획 변경 평균":float(metrics["changes"]),"비교 조건 수":int(data["overall"]["paired_conditions"])})
            st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
            st.warning("판매 관측 패턴에 가상 재고·가격·비용을 적용한 결과입니다. 현장 이익이나 실제 절감액으로 사용할 수 없습니다.")
        except Exception as exc:
            report_error("research external report", exc, "실험 보고서를 표시하지 못했습니다. 완료된 연구 보고서 형식인지 확인해 주세요.")
