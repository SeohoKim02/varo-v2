"""Non-destructive evidence integration, not an experiment rerun or promotion.

Every invocation requires a new output directory. Missing modules remain NOT_RUN.
Provenance-derived IDs are explicitly distinguished from recorded runtime hashes.
CSV files are machine-readable research exports of saved experiment evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from services.operating_snapshot import signature


LONG_FIELDS = ["run_id", "case_id", "dataset_id", "decision_time", "base_problem_signature",
               "candidate_set_signature", "algorithm_signature", "information_set_id", "objective_id",
               "allocation_policy", "data_class", "solver_status", "feasibility_status", "metric_name",
               "metric_value", "metric_unit", "baseline_id", "runtime_kind", "runtime_seconds", "evidence_path"]
COMPARISON_FIELDS = ["문제/데이터", "목적함수", "정보 시점", "후보집합", "할당 정책", "기준 알고리즘", "후보 알고리즘",
                     "원시 결과", "최종 선택 결과", "fallback 여부", "서비스 지표", "경제 지표", "제약 위반", "시간",
                     "표본/반복 수", "근거 파일", "해석 범위"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")


def common_row(*, run, case, dataset, decision, base, candidates, algorithm, information,
               objective, allocation, data_class, status, feasible, baseline, evidence,
               runtime=None, runtime_kind="NOT_RECORDED_PER_CASE", signature_kind="PROVENANCE_DERIVED_ID"):
    return {"run_id": run, "case_id": case, "dataset_id": dataset, "decision_time": decision,
            "base_problem_signature": base, "candidate_set_signature": candidates,
            "algorithm_signature": algorithm, "information_set_id": information, "objective_id": objective,
            "allocation_policy": allocation, "data_class": data_class, "solver_status": status,
            "feasibility_status": feasible, "baseline_id": baseline, "runtime_kind": runtime_kind,
            "runtime_seconds": runtime, "evidence_path": str(Path(evidence).resolve()),
            "signature_kind": signature_kind}


def add_metrics(rows, common, metrics):
    for name, value, unit in metrics:
        rows.append({**common, "metric_name": name, "metric_value": value, "metric_unit": unit})


def comparison(**values):
    return {name: values.get(name, "해당 없음") for name in COMPARISON_FIELDS}


def pending(rows, table, module, expected_path):
    common = common_row(run=module, case="NOT_RUN", dataset="NOT_RUN", decision="NOT_RUN", base="NOT_RUN",
        candidates="NOT_RUN", algorithm="NOT_RUN", information="NOT_RUN", objective="NOT_RUN", allocation="NOT_RUN",
        data_class="NOT_RUN", status="NOT_RUN", feasible="NOT_RUN", baseline="NOT_RUN", evidence=expected_path)
    add_metrics(rows, common, [("execution_status", "NOT_RUN", "status")])
    table.append(comparison(**{"문제/데이터": module, "후보 알고리즘": "NOT_RUN", "원시 결과": "NOT_RUN",
        "최종 선택 결과": "미실행. 승격 없음", "근거 파일": str(expected_path), "해석 범위": "해당 결과 파일이 아직 없음"}))


def build(root: Path, output: Path, *, r2_report: Path | None = None, r7_report: Path | None = None):
    root = root.resolve(); output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []; table = []; sources = []; notes = []

    def record(path, role):
        path = Path(path).resolve()
        sources.append({"path": str(path), "sha256": sha(path), "role": role})

    # Historical summary is an index only, never the 744 raw execution data.
    prior = root.parent / "upgrade_20261016"
    old_summary = prior / "deliverables" / "Varo_정량실험_비교결과.csv"
    if old_summary.exists():
        record(old_summary, "historical_25_row_summary_not_raw_experiments")
        notes.append({"prior_summary_rows": len(pd.read_csv(old_summary)), "not_equal_to": "744 executions"})
    r1 = root / "r1/run1"
    if (r1 / "report.json").exists():
        report = read(r1 / "report.json"); record(r1 / "report.json", "R1 fresh replay summary")
        for name in ("historical_744_longform.csv", "new_longform.csv"):
            path = r1 / name; record(path, "historical preserved rows" if name.startswith("historical") else "fresh R1 raw metrics")
            part = pd.read_csv(path, keep_default_na=False).to_dict("records")
            for row in part:
                linked = Path(row["evidence_path"])
                row["evidence_path"] = str((linked if linked.is_absolute() else root.parent.parent / linked).resolve())
                row["signature_kind"] = "RECORDED_RUNTIME_SIGNATURE"
                if str(row.get("algorithm", "")).startswith("PROTECTED"):
                    row["runtime_kind"] = "PORTFOLIO_GATE_ONLY_PRECOMPUTED_CHALLENGERS"
            rows.extend(part)
        final = next(x for x in report["daily_aggregate"] if x["algorithm"] == "PROTECTED")
        for result in report["daily_aggregate"]:
            name = result["algorithm"]
            time_value = report["portfolio_total_seconds"] if name == "PROTECTED" else result["seconds"]
            table.append(comparison(**{"문제/데이터": "수협 2026-07 31일 frozen PROXY", "목적함수": "계획 원수량 → 계산 운송비 → 생성기 tie",
                "정보 시점": "각 날짜 의사결정 입력. 실제 미래 판매 미사용", "후보집합": "같은 ALL_VALID. 날짜별 공통 제약",
                "할당 정책": "전량 요청" if name == "VARO" else "0 또는 정수 q=1..원 요청",
                "기준 알고리즘": "VARO 전량 휴리스틱", "후보 알고리즘": name,
                "원시 결과": f"계획량 {result['quantity']}; 계산운송비 {result['cost']}",
                "최종 선택 결과": "비교기준. 보호 포트폴리오에 포함하지 않음" if name == "MILP" else f"보호형 {final['quantity']} / {final['cost']}",
                "fallback 여부": "strict fallback 0/31. 동률 baseline 유지 21/31" if name == "PROTECTED" else "원시 결과. fallback 미적용",
                "서비스 지표": "계획 원수량. 물리 단위 미확인", "경제 지표": "계산 운송비 KRW proxy. 실제 지출/절감 아님",
                "제약 위반": result["violations"], "시간": f"{time_value:.6f}s " + ("전 후보+검증 총시간" if name == "PROTECTED" else "합계"),
                "표본/반복 수": "31개 종속 일별 문제", "근거 파일": str((r1 / "days").resolve()),
                "해석 범위": "같은 후보에서 선택기·분할 처치. 새 raw 비용개선 1일, tie-only 개선 9일. 운영 효과 미확인"}))
        # Candidate expansion is a different treatment, preserved and not relabeled.
        historical = pd.read_csv(prior / "w2_final/summary.csv")
        record(prior / "w2_final/summary.csv", "historical candidate-policy treatment summary")
        for result in historical[historical.algorithm == "MILP"].to_dict("records"):
            table.append(comparison(**{"문제/데이터": "직전 배치 수협 31일. 역사적 기록", "목적함수": "계획 원수량 → 비용 → 생성기 tie",
                "정보 시점": "동일 날짜 입력", "후보집합": result["policy"], "할당 정책": "정수 유한 수량 그리드",
                "기준 알고리즘": "LEGACY_20 / MILP", "후보 알고리즘": result["policy"] + " / MILP",
                "원시 결과": f"{result['quantity']} / {result['cost']}", "최종 선택 결과": "기존 기록 보존. 이번 원시 ALL_VALID 재현은 별도",
                "fallback 여부": "미적용", "서비스 지표": "원수량", "경제 지표": "계산 운송비 proxy", "제약 위반": result["violations"],
                "시간": result["seconds"], "표본/반복 수": "정책별 31일", "근거 파일": str(prior / "w2_final/plans.json"),
                "해석 범위": "후보집합 확대가 처치. 동일 후보 선택기 개선과 구분. 39.9%는 기존 결과 재확인"}))
        # Bound/gap details remain in raw plan logs; cumulative process memory is not solver-only memory.
        raw_summary = next(x for x in report["daily_aggregate"] if x["algorithm"] == "IMPROVED_RAW")
        milp_summary = next(x for x in report["daily_aggregate"] if x["algorithm"] == "MILP")
        notes.append({"r1_memory_scope": "Cumulative audit process peak includes loaded historical JSON/CSV, not isolated solver memory; daily_results.csv contains measured value",
                      "r1_runtime_note": {"new_raw_seconds": raw_summary["seconds"], "MILP_seconds": milp_summary["seconds"], "speed_superiority_claim": False},
                      "r1_protected_runtime": report["portfolio_total_seconds"],
                      "r1_postrun_code_changes": "trace label ENUMERATION clarified and noncanonical tolerance-only quantities rejected; recorded valid plans independently replayed"})
    else:
        pending(rows, table, "R1", r1 / "report.json")

    # R2 already emits all twenty required columns. Preserve its information IDs.
    if r2_report is None:
        matches = list((root / "r2").glob("*/economic_second_report.json"))
        r2_report = max(matches, key=lambda p: p.stat().st_mtime) if matches else root / "r2/run1/economic_second_report.json"
    if r2_report.exists():
        report = read(r2_report); record(r2_report, "R2 controlled scenario results and raw trajectories")
        lf = r2_report.with_name("economic_long_form.csv"); record(lf, "R2 saved long-form")
        part = pd.read_csv(lf, keep_default_na=False).to_dict("records")
        rows.extend({**r, "signature_kind": "RECORDED_RUNTIME_SIGNATURE"} for r in part)
        for name, scenario in report["fair_baselines"].items():
            for policy, result in scenario["results"].items():
                table.append(comparison(**{"문제/데이터": "R2 SCENARIO " + name, "목적함수": result.get("objective", "INPUT_REQUIRED"),
                    "정보 시점": "공통 t0 정보", "후보집합": "공통 제공 경로·가격수준. 정책별 허용행동은 처치", "할당 정책": policy + ": " + str(scenario["baseline_definitions"].get(policy, "")),
                    "기준 알고리즘": "A 정상판매 / B 이동 없는 할인 / C 순차 / D 독립+공동보정", "후보 알고리즘": policy,
                    "원시 결과": result.get("expected_net_value", "입력 부족"), "최종 선택 결과": "연구 비교. production 승격 없음",
                    "fallback 여부": "미적용", "서비스 지표": "경로·판매수량은 원본 결과 참조", "경제 지표": "기대 순현금가치. 가정 KRW",
                    "제약 위반": result.get("constraint_violations", "NOT_RECORDED"), "시간": result.get("runtime_seconds", "NOT_RECORDED"),
                    "표본/반복 수": "통제 시나리오 1개", "근거 파일": str(r2_report),
                    "해석 범위": "이동/할인 허용 효과와 공동최적화 효과 분리. 실제 절감액 아님"}))
        for name, scenario in report["adaptive_policy_cases"].items():
            for policy, result in scenario["results"].items():
                table.append(comparison(**{"문제/데이터": "R2 적응형 SCENARIO " + name, "목적함수": "기대 순현금가치",
                    "정보 시점": "미래 완전정보 oracle" if policy == "PERFECT_INFORMATION" else "각 시점까지 관측된 이력만",
                    "후보집합": "공통 외생 시나리오", "할당 정책": policy, "기준 알고리즘": "OPEN_LOOP", "후보 알고리즘": policy,
                    "원시 결과": f"기대 {result.get('expected_net_value')}; 최악 {result.get('worst_net_value')}",
                    "최종 선택 결과": "oracle는 배포 비교대상 아님" if policy == "PERFECT_INFORMATION" else "연구 비교만",
                    "fallback 여부": "미적용", "경제 지표": "기대/최악 순현금가치. SCENARIO",
                    "제약 위반": result.get("constraint_violations", "원본 conservation 참조"), "시간": result.get("runtime_seconds", "NOT_RECORDED"),
                    "표본/반복 수": "통제 시나리오 트리 1개", "근거 파일": str(r2_report), "해석 범위": "정보 구조가 처치. oracle와 공정한 동정보 비교라고 주장하지 않음"}))
        for risk in ("expected", "worst_case"):
            result = report["risk_tradeoff"][risk]
            table.append(comparison(**{"문제/데이터": "R2 uncertainty 동일 외생 시나리오", "목적함수": result["objective"],
                "정보 시점": "t0 공통 사전분포", "후보집합": "공통 경로·가격", "할당 정책": risk,
                "기준 알고리즘": "다른 위험 목적 정책", "후보 알고리즘": risk,
                "원시 결과": f"기대 {result['expected_net_value']}; 최악 {result['worst_net_value']}",
                "최종 선택 결과": "목적 간 trade-off. 자동 우승 선택 없음", "fallback 여부": "미적용",
                "경제 지표": "기대·최악 순현금가치 둘 다 표시", "제약 위반": result.get("constraint_violations"),
                "시간": result.get("runtime_seconds"), "표본/반복 수": "동일 통제 문제 1개", "근거 파일": str(r2_report),
                "해석 범위": "51/26 대 30/30. +21 무조건 성능 향상으로 해석 금지"}))
    else:
        pending(rows, table, "R2", r2_report)

    # Forecast per-case MAE records share the source, origin, mask, and cutoff.
    forecast = root / "r4_r5/forecast"
    if (forecast / "run_summary.json").exists():
        prereg = read(forecast / "preregistration.json"); record(forecast / "preregistration.json", "forecast train-only selection protocol")
        for sample in ("development", "expanded48"):
            report_path = forecast / (sample + "_report.json"); report = read(report_path); record(report_path, sample + " forecast summary")
            evidence = forecast / (sample + "_case_metrics.csv"); record(evidence, sample + " common-mask per-case errors")
            decisions = {(r["series"], r["origin"]): r for r in read(forecast / (sample + "_decisions.json"))}
            for r in pd.read_csv(evidence).to_dict("records"):
                d = decisions[(r["series"], r["origin"])]
                base = signature({"dataset": prereg["source"], "series": r["series"], "cutoff": d["cutoff"], "horizon": prereg["horizon"]})
                for method in prereg["methods"]:
                    common = common_row(run="R4_" + sample, case=f"{r['series']}@{r['origin']}", dataset="FreshRetailNet_same_source",
                        decision=d["cutoff"], base=base, candidates=signature(prereg["methods"]),
                        algorithm=signature({"method": method, "protocol": prereg}), information="TRAINING_PREFIX_ENDING_" + d["cutoff"],
                        objective="FORECAST_OBSERVED_SALES_MAE_EVALUATION", allocation="PREDICT_ONLY", data_class="OBSERVED_SALES",
                        status="COMPUTED", feasible="COMMON_MASK", baseline="OLD_V2", evidence=evidence)
                    add_metrics(rows, common, [("case_MAE", r[method], "normalized_sales_amount")])
            downstream = {r["method"]: r for r in report["downstream"]}
            for r in report["summary"]:
                common = common_row(run="R4_" + sample, case=r["scope"], dataset="FreshRetailNet_same_source",
                    decision="TRAINING_PREFIX_PER_ORIGIN", base=signature({"sample": sample, "scope": r["scope"], "rows": r["rows"]}),
                    candidates=signature(prereg["methods"]), algorithm=signature({"method": r["method"], "protocol": prereg}),
                    information="NO_FUTURE_LABELS_IN_SELECTION", objective="OBSERVED_SALES_ERROR_EVALUATION", allocation="PREDICT_ONLY",
                    data_class="OBSERVED_SALES", status="COMPUTED", feasible="COMMON_MASK", baseline="OLD_V2", evidence=report_path)
                add_metrics(rows, common, [("WAPE", r["wape"], "ratio"), ("MAE", r["mae"], "normalized_sales_amount"), ("bias", r["bias"], "normalized_sales_amount")])
                if r["scope"] != "ALL_OBSERVED_SALES": continue
                ds = downstream[r["method"]]
                add_metrics(rows, {**common, "data_class": "SCENARIO_OBSERVED_SALES_AS_DEMAND", "objective_id": "SIMULATED_CASH_EVALUATION_NOT_OPTIMIZATION"},
                    [("scenario_net_value", ds["net"], "normalized_scenario_value"), ("scenario_filled", ds["filled"], "normalized_sales_amount")])
                table.append(comparison(**{"문제/데이터": f"FreshRetailNet {sample}", "목적함수": "관측 판매 예측오차. 다운스트림은 별도 가정 평가",
                    "정보 시점": "각 origin 이전 학습 구간", "후보집합": "동일 공통 적격/관측 마스크", "할당 정책": "예측만. 주문=예측 SCENARIO 별도",
                    "기준 알고리즘": "OLD_V2", "후보 알고리즘": r["method"], "원시 결과": f"WAPE {r['wape']:.9f}; MAE {r['mae']:.9f}",
                    "최종 선택 결과": "신규 SECOND_CANDIDATE 승격 안 함" if not report["research_candidate_promoted"] else "연구 후보만. 운영 승인 없음",
                    "fallback 여부": "승격 판단. 실행 fallback 아님", "서비스 지표": "관측 판매 예측. 참 수요 아님",
                    "경제 지표": f"가정 net {ds['net']:.9f}. 실제 화폐/효과 아님", "제약 위반": "해당 없음",
                    "시간": "케이스별 미기록. run_summary 총시간 참조", "표본/반복 수": f"{report['series']} series, {report['valid_cases']} origins, {r['rows']} rows",
                    "근거 파일": str(report_path), "해석 범위": "같은 공개 데이터의 신규 series. 외부 사업장 일반화 증거 아님"}))
    else:
        pending(rows, table, "R4", forecast / "run_summary.json")

    dqn = root / "r4_r5/dqn"
    if (dqn / "report.json").exists():
        report = read(dqn / "report.json"); prereg = read(dqn / "preregistration.json")
        record(dqn / "report.json", "DQN baseline and mask audit")
        evidence = dqn / "policy_cases.csv"; record(evidence, "DQN exact-case raw metrics")
        for r in pd.read_csv(evidence).to_dict("records"):
            model_sha = next((v for k, v in report["existing_models_before"].items() if k.endswith(f"candidate_seed_{r['seed']}.pt")), None) if r["policy"] == "TRAINED_DQN" else None
            common = common_row(run="R5_" + r["scope"], case=f"{r['case_id']}@seed{r['seed']}", dataset="SYNTHETIC_ALLOCATION", decision="t0",
                base=r["base_problem_signature"], candidates=r["candidate_set_signature"], algorithm=signature({"policy": r["policy"], "seed": r["seed"], "model": model_sha, "protocol": prereg}),
                information=r["information_set"], objective=r["objective"], allocation="FINITE_PARTIAL_ACTIONS", data_class=r["evidence_class"],
                status="FEASIBLE_CASE_NO_OPTIMALITY_PROOF_IN_CSV", feasible="VALID" if r["violations"] == 0 else "INVALID", baseline="GREEDY_IMMEDIATE",
                evidence=evidence, runtime=r["runtime_seconds"], runtime_kind="FRESH_POLICY_WALL_SECONDS")
            add_metrics(rows, common, [("service", r["service"], "ratio"), ("cost", r["cost"], "SCENARIO_COST"), ("violations", r["violations"], "count")])
        for r in report["summary"]:
            table.append(comparison(**{"문제/데이터": "DQN " + r["scope"], "목적함수": "서비스 → 비용. VHS tie 미비교",
                "정보 시점": "현재 유한 할당 문제. 미래 label 없음", "후보집합": "동일 사례·행동마스크", "할당 정책": "부분 수량 행동",
                "기준 알고리즘": "무학습/유효무작위/Greedy/기존Varo/MILP", "후보 알고리즘": r["policy"],
                "원시 결과": f"평균 service {r['mean_service']}; cost {r['mean_cost']}", "최종 선택 결과": "학습 효과와 최적성 별개. 연구용 유지",
                "fallback 여부": "미적용", "서비스 지표": "평균 충족률", "경제 지표": "SCENARIO 계산비용",
                "제약 위반": r["violations"], "시간": r["runtime"], "표본/반복 수": r["cases"], "근거 파일": str(evidence),
                "해석 범위": "16 기존 평가문제는 구조 템플릿 겹침. 추가36도 synthetic. 실제운영 승격 없음"}))
    else:
        pending(rows, table, "R5_DQN", dqn / "report.json")

    vhs = root / "r4_r5/vhs"
    if (vhs / "report.json").exists():
        report = read(vhs / "report.json"); prereg = read(vhs / "preregistered_auxiliary_metrics.json")
        evidence = vhs / "daily_results.json"; record(evidence, "VHS/Pareto same legacy pool raw audit")
        for r in read(evidence):
            objective = "VHS_SCORE_ORDER" if not r["strategy"].startswith("QTY_COST") else "LEX_RAW_QUANTITY_COST_OPTIONAL_VHS_TIE"
            common = common_row(run="R5_VHS", case=r["date"], dataset="Suhyup_legacy20", decision=r["date"],
                base=signature({"input": report["source_hashes_before"], "date": r["date"], "scope": report["scope"]}),
                candidates=signature({"pool": "LEGACY_20", "date": r["date"], "input": report["source_hashes_before"]}),
                algorithm=signature({"strategy": r["strategy"], "protocol": prereg}), information="DATE_SNAPSHOT_PROXY",
                objective=objective, allocation="WHOLE_REQUEST_LEGACY_POOL", data_class="PROXY", status="HEURISTIC",
                feasible="NONE_REPORTED" if r["violations"] is None else ("VALID" if not r["violations"] else "INVALID"),
                baseline="QTY_COST_NO_TIE", evidence=evidence)
            add_metrics(rows, common, [("raw_quantity", r["quantity"], "raw_quantity_UNCONFIRMED"),
                ("computed_transport_cost", r["cost"], "KRW_PROXY"), ("target_HHI", r["target_hhi"], "index_not_fairness_proof"),
                ("permutation_Jaccard", min(r["jaccard_permutations"]), "ratio")])
        for r in report["summary"]:
            table.append(comparison(**{"문제/데이터": "수협 동일 LEGACY_20 31일", "목적함수": "VHS 점수순 또는 수량→비용→tie. 목적을 구분",
                "정보 시점": "각 날짜 proxy", "후보집합": "일별 동일 기존20개", "할당 정책": "기존 전량", "기준 알고리즘": "QTY_COST_NO_TIE",
                "후보 알고리즘": r["strategy"], "원시 결과": f"{r['quantity']} / {r['cost']}", "최종 선택 결과": "연구 비교. production 변경 없음",
                "fallback 여부": "미적용", "서비스 지표": "계획 원수량", "경제 지표": "계산 운송비 proxy", "제약 위반": "보고서 violation_days=" + str(r["violation_days"]),
                "시간": "미기록", "표본/반복 수": "31일, 순서교란 3회", "근거 파일": str(evidence),
                "해석 범위": "수량·비용 추가 기여 미입증. HHI는 공정성 증거 아님. 실제 폐기확률 미확인"}))
    else:
        pending(rows, table, "R5_VHS", vhs / "report.json")

    if r7_report is None:
        matches = list((root / "r7").rglob("*report*.json")) if (root / "r7").exists() else []
        r7_report = max(matches, key=lambda p: p.stat().st_mtime) if matches else root / "r7/report.json"
    if r7_report.exists():
        report = read(r7_report); record(r7_report, "R7 data acquisition and connection report; not outcome trial")
        common = common_row(run="R7", case="DATA_ACQUISITION", dataset="SOURCE_REVIEW", decision="REPORT_DATE",
            base=sha(r7_report), candidates="NOT_APPLICABLE", algorithm="NOT_APPLICABLE", information="PUBLIC_SOURCE_AND_USER_SCOPE",
            objective="DATA_LINKABILITY_AUDIT", allocation="NOT_APPLICABLE", data_class="SOURCE_AUDIT",
            status=str(report.get("status", "REPORT_AVAILABLE")), feasible="NOT_AN_ALLOCATION", baseline="NOT_APPLICABLE", evidence=r7_report)
        add_metrics(rows, common, [("report_status", report.get("status", "REPORT_AVAILABLE"), "status")])
        table.append(comparison(**{"문제/데이터": "R7 자료 확보·필드 연결", "목적함수": "연결 가능성 감사", "후보 알고리즘": "해당 없음",
            "원시 결과": str(report.get("status", "보고서 있음. 구체 연결 범위는 원본 참조")), "최종 선택 결과": "현장 성과 승인 아님",
            "근거 파일": str(r7_report), "해석 범위": "파일·공개자료 확보와 실제 사업장 운영 검증은 다름"}))
    else:
        pending(rows, table, "R7", r7_report)

    # Preserve direct routes to predictions, actions, accounting, masks and solver
    # evidence; these are not collapsed into aggregate means or new trial counts.
    support = {
        r1 / "quantity_cost_parity.csv": "every finite quantity and independent tariff coefficient",
        r1 / "daily_causes.json": "first divergent decisions and remaining resources",
        r1 / "stress_plans.json": "new stress inputs, raw plans, solver stages/bounds/gaps",
        prior / "w2_final/plans.json": "744 original per-policy per-date plans and solver evidence",
        prior / "w3/final_v01/economic_experiments.json": "ten original economic scenarios, preserved historical evidence",
        forecast / "expanded48_predictions.csv": "new-series per-date predictions and observations",
        forecast / "development_predictions.csv": "exposed development predictions and observations",
        forecast / "expanded48_decisions.json": "train-only inner choices, cutoffs and eligibility",
        forecast / "development_decisions.json": "shared-mask development choices and exclusions",
        forecast / "expanded48_downstream_rows.csv": "scenario accounting, not actual business outcomes",
        forecast / "exposure_log.json": "training-only stratified sampling and exposure record",
        dqn / "trajectories.json": "actions, rewards, valid-action sets and states",
        dqn / "mask_and_actions.json": "mask/action distribution and invalid-action counts",
        dqn / "existing_learning_curves.csv": "recorded historical returns, not retrospectively fabricated TD losses",
        dqn / "rolling_inventory_bridge.json": "sequential stock trajectories and observed-only information cutoffs",
        vhs / "feature_scores.json": "full feature missingness, normalization and score provenance",
        vhs / "pareto_exact_audit.json": "exact Pareto comparison per day",
        root / "r3/acceptance.json": "input/app acceptance evidence, not actual business effects",
    }
    seen = {r["path"] for r in sources}
    for path, role in support.items():
        if path.exists() and str(path.resolve()) not in seen:
            record(path, role); seen.add(str(path.resolve()))

    # Do not fill absent measurements/signatures with zero. Keep unavailable strings explicit.
    if any(not set(LONG_FIELDS).issubset(row) for row in rows):
        raise AssertionError("long-form fields missing")
    longframe = pd.DataFrame(rows)
    ordered = LONG_FIELDS + [c for c in longframe.columns if c not in LONG_FIELDS]
    longframe[ordered].to_csv(output / "evidence_longform.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(table, columns=COMPARISON_FIELDS).to_csv(output / "comparison_17_columns.csv", index=False, encoding="utf-8-sig")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root), "sources": sources,
                "longform_rows": len(rows), "comparison_rows": len(table), "longform_required_columns": LONG_FIELDS,
                "comparison_columns": COMPARISON_FIELDS, "source_files_modified": False, "experiment_rerun": False,
                "notes": notes + [{"signature_note": "Provenance-derived identifiers are not retroactively claimed runtime code hashes.",
                                   "runtime_note": "Unrecorded per-case time stays blank/NOT_RECORDED. Protected gate time is not full portfolio time.",
                                   "statistics_note": "Metric rows and seeds are not independent field experiments."}]}
    write_json(output / "evidence_manifest.json", manifest)
    return {"output": str(output), "longform_rows": len(rows), "comparison_rows": len(table), "sources": len(sources)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/second_upgrade_20261016"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r2-report", type=Path); parser.add_argument("--r7-report", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output, r2_report=args.r2_report, r7_report=args.r7_report), ensure_ascii=False, indent=2))
