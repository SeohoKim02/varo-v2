"""Fault-isolated orchestration of approved legacy Varo algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from services.analysis_provenance import (
    build_greedy_provenance,
    build_vhs_provenance,
    confidence_provenance,
    kpi_sources,
    sanitize_optimality_result,
)
from services.data_validator import ValidationReport, validate_workbook_data
from services.dqn_guard import dqn_exclusion_report, is_dqn_column, strip_dqn_columns
from services.legacy_adapters.data_adapter import (
    add_cluster_context,
    build_candidate_frame,
    config_time,
    config_value,
    prepare_legacy_data,
    safe_transfer_route_lookup,
)
from services.legacy_adapters.loader import (
    LegacyAlgorithmUnavailable,
    available_legacy_algorithms,
    load_legacy_module,
)
from services.recommendation_adapter import normalize_action, recommendations_from_dataframe
from services.v2_summaries import (
    V2_SUMMARY_FUNCTIONS,
    recommendation_reasons,
    sensitivity_summary,
    vhs_neutral_summary,
)
from services.vhs_score_engine import apply_auto_vhs

AUTO_VHS_INPUT_COLUMNS = (
    "savings_score", "disposal_risk_score", "demand_fit_score",
    "inventory_balance_score", "route_cost_score", "feasibility_score",
    "promotion_score", "greedy_score", "dqn_reference_score",
    "vhs_vs_greedy_match", "vhs_vs_dqn_match", "final_reason",
    "weight_profile_id", "weight_summary", "varo_final_decision",
    "varo_final_rank",
)


@dataclass
class PipelineResult:
    status: str = "empty"
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, object]] = field(default_factory=list)
    top5: List[Dict[str, object]] = field(default_factory=list)
    vhs_analysis: Dict[str, Any] = field(default_factory=dict)
    greedy_analysis: Dict[str, Any] = field(default_factory=dict)
    pareto_analysis: Dict[str, Any] = field(default_factory=dict)
    confidence_analysis: Dict[str, Any] = field(default_factory=dict)
    route_analysis: Dict[str, Any] = field(default_factory=dict)
    promotion_analysis: Dict[str, Any] = field(default_factory=dict)
    demand_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_analysis: Dict[str, Any] = field(default_factory=dict)
    validation_report: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    connected_algorithms: List[str] = field(default_factory=list)
    deferred_algorithms: List[Dict[str, str]] = field(default_factory=list)
    vhs_neutral_analysis: Dict[str, Any] = field(default_factory=dict)
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)
    reason_analysis: Dict[str, Any] = field(default_factory=dict)
    vhs_weight_analysis: Dict[str, Any] = field(default_factory=dict)
    vhs_greedy_dqn_comparison: List[Dict[str, Any]] = field(default_factory=list)
    v2_summary_functions: List[str] = field(default_factory=list)
    excluded_dqn_artifacts: Dict[str, Any] = field(default_factory=dqn_exclusion_report)
    result_basis: str = "알고리즘 미연결"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class _Runner:
    def __init__(self, result: PipelineResult) -> None:
        self.result = result
        self.technical_errors: list[dict[str, str]] = []

    def call(self, module_name: str, function_name: str, *args, **kwargs):
        label = f"{module_name}.{function_name}"
        try:
            module = load_legacy_module(module_name)
            output = getattr(module, function_name)(*args, **kwargs)
            if label not in self.result.connected_algorithms:
                self.result.connected_algorithms.append(label)
            return output
        except Exception as exc:
            self.technical_errors.append({
                "algorithm": label,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            self.result.deferred_algorithms.append({
                "algorithm": label,
                "reason": "입력 조건 또는 원본 함수 실행 조건을 확인해야 합니다.",
            })
            self.result.warnings.append(f"{label} 분석은 보류되었습니다.")
            return None

    def defer(self, label: str, reason: str) -> None:
        self.result.deferred_algorithms.append({"algorithm": label, "reason": reason})


def _dataframe_records(df: pd.DataFrame | None, limit: int = 100) -> list[dict[str, object]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    clean = df.head(limit).copy()
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict("records")


def _drop_empty_auto_vhs_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    drop_columns = [
        column for column in AUTO_VHS_INPUT_COLUMNS
        if column in df.columns and df[column].isna().all()
    ]
    return df.drop(columns=drop_columns) if drop_columns else df


def _summary_call(runner: _Runner, module_name: str, function_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    output = runner.call(module_name, function_name, frame)
    if isinstance(output, dict):
        return output
    if isinstance(output, pd.DataFrame):
        return {"rows": _dataframe_records(output), "row_count": int(len(output))}
    return {}


def _run_inventory_analysis(
    runner: _Runner, inventory: pd.DataFrame, *, collect_details: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    current = strip_dqn_columns(inventory)
    summaries: dict[str, Any] = {}
    steps = (
        ("abc_analyzer", "analyze_abc", "get_abc_summary", "abc", ("product_id", "stock_qty", "unit_price")),
        ("turnover_analyzer", "analyze_turnover", "get_turnover_summary", "turnover", ("stock_qty", "sales_30d")),
        ("disposal_risk_analyzer", "analyze_disposal_risk", "get_disposal_risk_summary", "disposal_risk", ("stock_qty", "days_to_expiry")),
        ("demand_forecast_analyzer", "analyze_demand_forecast", "get_demand_forecast_summary", "demand_forecast", ("sales_7d", "sales_30d")),
        ("safety_stock_analyzer", "analyze_safety_stock", "get_safety_stock_summary", "safety_stock", ("demand_std", "lead_time_days")),
        ("eoq_analyzer", "analyze_eoq", "get_eoq_summary", "eoq", ("avg_daily_sales", "order_cost")),
        ("store_product_matcher", "analyze_store_product_matching", None, "store_product_match", ("store_id", "product_id", "stock_qty")),
    )
    for module_name, analyze_name, summary_name, key, required_columns in steps:
        before_columns = set(current.columns)
        output = runner.call(module_name, analyze_name, current)
        if isinstance(output, pd.DataFrame) and not output.empty:
            current = strip_dqn_columns(output)
            summary = (
                _summary_call(runner, module_name, summary_name, current)
                if collect_details and summary_name
                else {"analyzed_rows": int(len(current))} if collect_details else {}
            )
            summaries[key] = {
                "status": "연결",
                "function": f"{module_name}.{analyze_name}",
                "input_columns": [column for column in required_columns if column in before_columns],
                "missing_input_columns": [column for column in required_columns if column not in before_columns],
                "output_columns": sorted(set(current.columns) - before_columns),
                "summary": summary,
            }
        else:
            summaries[key] = {
                "status": "보류",
                "function": f"{module_name}.{analyze_name}",
                "input_columns": [column for column in required_columns if column in before_columns],
                "missing_input_columns": [column for column in required_columns if column not in before_columns],
                "output_columns": [],
                "summary": {},
            }
    return current, summaries


def _run_clustering(
    runner: _Runner, stores: pd.DataFrame, inventory: pd.DataFrame, *, collect_details: bool = True,
) -> tuple[dict[str, object], dict[str, Any]]:
    output = runner.call("store_clustering", "analyze_store_clustering", stores, inventory)
    if not isinstance(output, tuple) or len(output) < 3:
        return {}, {}
    stores_clustered, cluster_summary, cluster_map = output[:3]
    details = {
        "stores": _dataframe_records(stores_clustered),
        "summary": _dataframe_records(cluster_summary),
    } if collect_details else {}
    return dict(cluster_map or {}), details


def _enrich_route_comparison(candidates: pd.DataFrame, transfer: pd.DataFrame | None) -> pd.DataFrame:
    if candidates is None or candidates.empty or transfer is None or transfer.empty:
        return candidates
    result = candidates.copy()
    keyed = transfer.copy()
    for column in ("product_id", "source_store_id", "target_store_id"):
        if column in keyed.columns:
            keyed[column] = keyed[column].astype(str)
    for index, row in result.iterrows():
        match = keyed[
            (keyed["product_id"] == str(row.get("product_id")))
            & (keyed["source_store_id"] == str(row.get("source_id")))
            & (keyed["target_store_id"] == str(row.get("target_id")))
        ]
        if match.empty:
            continue
        path = match.iloc[0]
        values = {
            "direct_distance_km": path.get("direct_distance_km"),
            "direct_time_min": path.get("direct_time_min"),
            "direct_cost": path.get("direct_cost"),
            "via_dc_distance_km": path.get("via_distance_km"),
            "via_dc_time_min": path.get("via_time_min"),
            "via_dc_cost": path.get("via_cost"),
            "path_reason": path.get("transfer_reason"),
        }
        for column, value in values.items():
            result.at[index, column] = value
    return result


def _enrich_route_constraints(
    candidates: pd.DataFrame,
    cutline: pd.DataFrame | None,
    time_windows: pd.DataFrame | None,
) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return candidates
    result = candidates.copy()
    cutline = cutline if isinstance(cutline, pd.DataFrame) else pd.DataFrame()
    time_windows = time_windows if isinstance(time_windows, pd.DataFrame) else pd.DataFrame()
    for index, row in result.iterrows():
        product_id = str(row.get("product_id"))
        target_id = str(row.get("target_id"))
        cutline_match = cutline[
            (cutline.get("product_id", pd.Series(dtype=object)).astype(str) == product_id)
            & (cutline.get("store_id", pd.Series(dtype=object)).astype(str) == target_id)
        ] if not cutline.empty else pd.DataFrame()
        if not cutline_match.empty:
            item = cutline_match.iloc[0]
            result.at[index, "cutline_passed"] = item.get("cutline_status")
            result.at[index, "cutline_reason"] = item.get("cutline_reason")
            result.at[index, "distance_cutline_km"] = item.get("distance_cutline_km")

        time_match = time_windows[
            (time_windows.get("product_id", pd.Series(dtype=object)).astype(str) == product_id)
            & (time_windows.get("store_id", pd.Series(dtype=object)).astype(str) == target_id)
        ] if not time_windows.empty else pd.DataFrame()
        if not time_match.empty:
            item = time_match.iloc[0]
            result.at[index, "time_window_status"] = item.get("final_status") or item.get("time_status")
            result.at[index, "time_window_reason"] = item.get("time_reason")
            result.at[index, "arrival_time"] = item.get("arrival_time")
            start, end = item.get("available_start"), item.get("available_end")
            if start and end:
                result.at[index, "available_window"] = f"{start}~{end}"
    return result


def _enrich_promotion(candidates: pd.DataFrame, promotion: pd.DataFrame | None) -> pd.DataFrame:
    if candidates is None or candidates.empty or promotion is None or promotion.empty:
        return candidates
    result = candidates.copy()
    for index, row in result.iterrows():
        match = promotion[
            (promotion["product_name"].astype(str) == str(row.get("product_name")))
            & (promotion["source_store"].astype(str) == str(row.get("source_name")))
            & (promotion["target_store"].astype(str) == str(row.get("target_name")))
        ]
        if match.empty:
            continue
        item = match.iloc[0]
        result.at[index, "promotion_recommended"] = item.get("final_decision")
        transfer_cost = pd.to_numeric(item.get("transfer_cost"), errors="coerce")
        promotion_cost = pd.to_numeric(item.get("promotion_net_cost"), errors="coerce")
        result.at[index, "promotion_transfer_cost"] = transfer_cost
        result.at[index, "promotion_net_cost"] = promotion_cost
        result.at[index, "promotion_effect"] = (
            transfer_cost - promotion_cost
            if pd.notna(transfer_cost) and pd.notna(promotion_cost) else None
        )
        result.at[index, "promotion_reason"] = item.get("decision_reason")
    return result


def _route_type_for_row(row: pd.Series) -> str:
    return str(row.get("route_type") or "").upper()


def _finalize_candidate_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    result = strip_dqn_columns(candidates)
    if result.empty:
        return result
    for safe_column in ("dqn_reference_score", "vhs_vs_dqn_match"):
        if safe_column in candidates.columns:
            result[safe_column] = candidates[safe_column]
    result["recalculated_vhs_score"] = pd.to_numeric(
        result.get("vhs", result.get("vhs_score")), errors="coerce"
    )
    result["vhs_score"] = result["recalculated_vhs_score"]
    result["vhs_score_source"] = "재계산 VHS · varo_hybrid_score.calculate_varo_hybrid_score"
    if "auto_vhs_score" in result.columns:
        result["recalculated_vhs_score"] = pd.to_numeric(result["auto_vhs_score"], errors="coerce")
        result["vhs_score"] = result["recalculated_vhs_score"]
        result["vhs_score_source"] = "VHS 자동 가중치 최적화"
    result["grade"] = result.get("vhs_grade", result.get("recommendation_grade"))
    result["recommendation_grade"] = result["grade"]
    if "auto_vhs_score" in result.columns:
        result["recommendation_grade"] = result["vhs_score"].apply(
            lambda value: "최적" if float(value or 0) >= 80 else "권장" if float(value or 0) >= 65 else "검토" if float(value or 0) >= 50 else "보류"
        )
        result["grade"] = result["recommendation_grade"]
    result["varo_action"] = result.get("vhs_action", result.get("final_recommendation", "보류")).apply(normalize_action)
    result["greedy_action"] = result.get("final_recommendation", pd.Series("비교 불가", index=result.index)).apply(normalize_action)
    result["greedy_selected"] = result.get(
        "is_greedy_selected", pd.Series(False, index=result.index)
    ).fillna(False).astype(bool)
    result["greedy_rank"] = pd.to_numeric(result.get("greedy_rank"), errors="coerce")
    result["heuristic_score"] = pd.to_numeric(result.get("heuristic_score"), errors="coerce")
    result["strategy_match"] = result["greedy_action"] == result["varo_action"]
    result["dqn_action"] = "미연결"
    result["dqn_status"] = "학습 필요"
    result["dqn_confidence"] = None
    result["dqn_correction"] = 0.0
    result["confidence"] = pd.to_numeric(result.get("confidence_score"), errors="coerce")
    result["confidence_source"] = "vhs_confidence.add_confidence_columns · DQN 제외"
    result["rank"] = pd.to_numeric(
        result.get("varo_final_rank", result.get("vhs_rank", result.get("greedy_rank"))),
        errors="coerce",
    )
    situations = result.get("vhs_dominant_situation", pd.Series("표준", index=result.index)).fillna("표준")
    result["reason"] = [
        f"{situation} 상황과 재고·수요·비용 지표를 반영해 {action} 전략을 권장합니다."
        for situation, action in zip(situations, result["varo_action"])
    ]
    return result


def _vhs_analysis(
    candidates: pd.DataFrame,
    runner: _Runner,
    input_columns: list[str],
) -> dict[str, Any]:
    summary = _summary_call(runner, "varo_hybrid_score", "get_vhs_summary", candidates)
    weight_columns = [column for column in candidates.columns if column.startswith("vhs_w_")]
    contribution_columns = [column for column in candidates.columns if column.startswith("vhs_contrib_")]
    weights = {
        column.removeprefix("vhs_w_"): round(float(pd.to_numeric(candidates[column], errors="coerce").mean()), 4)
        for column in weight_columns
        if pd.to_numeric(candidates[column], errors="coerce").notna().any()
    }
    contributions = {
        column.removeprefix("vhs_contrib_"): round(float(pd.to_numeric(candidates[column], errors="coerce").mean()), 2)
        for column in contribution_columns
        if pd.to_numeric(candidates[column], errors="coerce").notna().any()
    }
    provenance = build_vhs_provenance(candidates, input_columns, summary)
    provenance.update({
        "weights": weights,
        "contributions": contributions,
        "score_rows": _dataframe_records(candidates[[
            column for column in [
                "route_id", "product_name", "uploaded_vhs_score", "vhs",
                "vhs_grade", "vhs_action", "vhs_dominant_situation",
            ] if column in candidates.columns
        ]]),
        "connected_function": "varo_hybrid_score.calculate_varo_hybrid_score",
    })
    return provenance


def _run_routes(
    runner: _Runner, data: dict[str, pd.DataFrame], *, collect_details: bool = True,
) -> dict[str, Any]:
    stores, products, inventory, routes, config = (
        data["stores"], data["products"], data["inventory"], data["routes"], data["config"]
    )
    transfer_module = None
    try:
        transfer_module = load_legacy_module("transfer_path_analyzer")
        transfer_module._build_route_lookup = safe_transfer_route_lookup
        transfer = transfer_module.analyze_direct_vs_dc_transfer(
            stores, products, inventory, routes, max_result_rows=500
        )
        runner.result.connected_algorithms.append("transfer_path_analyzer.analyze_direct_vs_dc_transfer")
    except Exception as exc:
        runner.technical_errors.append({
            "algorithm": "transfer_path_analyzer.analyze_direct_vs_dc_transfer",
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        runner.defer(
            "transfer_path_analyzer.analyze_direct_vs_dc_transfer",
            "경로 입력 또는 원본 함수 실행 조건을 확인해야 합니다.",
        )
        runner.result.warnings.append("직접/DC 경로 비교는 보류되었습니다.")
        transfer = pd.DataFrame()

    departure = config_time(config)
    dc_routes, best_dc = pd.DataFrame(), pd.DataFrame()
    network = pd.DataFrame()
    if collect_details:
        route_output = runner.call("route_analyzer", "analyze_dc_retailer_routes", stores, routes)
        dc_routes, best_dc = (
            route_output
            if isinstance(route_output, tuple) and len(route_output) >= 2
            else (pd.DataFrame(), pd.DataFrame())
        )
        network = runner.call(
            "network_path_analyzer", "analyze_multi_store_network_paths",
            stores, products, routes, transfer, departure_time=departure,
        )
        if not isinstance(network, pd.DataFrame):
            network = pd.DataFrame()

    cutline_output = runner.call("cutline_analyzer", "analyze_product_distance_cutline", stores, products, inventory, routes)
    if isinstance(cutline_output, tuple):
        cutline_all = cutline_output[0] if len(cutline_output) else pd.DataFrame()
        cutline_best = cutline_output[1] if len(cutline_output) > 1 else pd.DataFrame()
        cutline_failed = cutline_output[2] if len(cutline_output) > 2 else pd.DataFrame()
    else:
        cutline_all, cutline_best, cutline_failed = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    time_output = runner.call("time_window_analyzer", "analyze_trade_time_windows", cutline_all, stores, departure)
    if isinstance(time_output, tuple):
        time_windows, time_message = time_output[0], str(time_output[1])
    else:
        time_windows, time_message = pd.DataFrame(), ""

    flow, network_nodes, network_summary = pd.DataFrame(), pd.DataFrame(), {}
    if collect_details:
        min_cost_output = runner.call("min_cost_network", "analyze_min_cost_network", inventory, stores, routes)
        if isinstance(min_cost_output, tuple) and len(min_cost_output) >= 3:
            flow, network_nodes, network_summary = min_cost_output[:3]

    return {
        "transfer_frame": transfer,
        "cutline_frame": cutline_all,
        "time_window_frame": time_windows,
        "status": "연결" if not transfer.empty else "일부 연결",
        "calculation_functions": [
            "transfer_path_analyzer.analyze_direct_vs_dc_transfer",
            "route_analyzer.analyze_dc_retailer_routes",
            "cutline_analyzer.analyze_product_distance_cutline",
            "time_window_analyzer.analyze_trade_time_windows",
        ],
        "comparison_count": int(len(transfer)),
        "cutline_comparable_count": int(len(cutline_all)),
        "time_window_comparable_count": int(len(time_windows)),
        "direct_vs_dc": _dataframe_records(transfer) if collect_details else [],
        "dc_routes": _dataframe_records(dc_routes) if collect_details else [],
        "best_dc_routes": _dataframe_records(best_dc) if collect_details else [],
        "network_paths": _dataframe_records(network) if collect_details else [],
        "cutline_best": _dataframe_records(cutline_best) if collect_details else [],
        "cutline_failed_count": int(len(cutline_failed)) if isinstance(cutline_failed, pd.DataFrame) else 0,
        "time_windows": _dataframe_records(time_windows) if collect_details else [],
        "time_window_message": time_message,
        "min_cost_flow": _dataframe_records(flow) if collect_details else [],
        "min_cost_nodes": _dataframe_records(network_nodes) if collect_details else [],
        "min_cost_summary": network_summary if collect_details and isinstance(network_summary, dict) else {},
    }


def _run_promotion(runner: _Runner, data: dict[str, pd.DataFrame], transfer: pd.DataFrame) -> pd.DataFrame:
    config = data["config"]
    output = runner.call(
        "promotion_analyzer", "analyze_promotion_vs_transfer",
        data["stores"], data["inventory"], transfer,
        promotion_type=str(config_value(config, "promotion_type", "할인 프로모션")),
        promotion_discount_rate=float(config_value(config, "promotion_discount_rate", 20)),
        promotion_sales_increase_rate=float(config_value(config, "promotion_sales_increase_rate", 80)),
        promotion_fixed_cost=float(config_value(config, "promotion_fixed_cost", 0)),
    )
    return output if isinstance(output, pd.DataFrame) else pd.DataFrame()


def run_analysis_pipeline(
    uploaded_data: Dict[str, Any], *, detail_level: str = "full",
    validation_report: ValidationReport | None = None,
) -> PipelineResult:
    """Run the recommendation pipeline.

    ``core`` keeps every calculation that can affect recommendation fields or
    ordering, while postponing report-only route summaries, optimality search,
    legacy validation, sensitivity, and reason tables.  ``full`` preserves the
    complete analysis contract used by the validation page and tests.
    """
    collect_details = detail_level == "full"
    result = PipelineResult(excluded_dqn_artifacts=dqn_exclusion_report())
    if not uploaded_data:
        return result

    validation = validation_report or validate_workbook_data(uploaded_data)
    result.validation_report = {"data_validation": validation.to_dict()}
    if validation.has_errors:
        result.status = "validation_error"
        result.diagnostics = {"status": "validation_error", "validation": validation.to_dict()}
        return result

    try:
        base_recommendations = recommendations_from_dataframe(uploaded_data["recommendations"])
    except ValueError as exc:
        result.status = "adapter_error"
        result.warnings.append(str(exc))
        return result

    source_dqn_columns = [
        str(column)
        for column in uploaded_data["recommendations"].columns
        if is_dqn_column(column)
    ]

    legacy_data = prepare_legacy_data(uploaded_data)
    runner = _Runner(result)
    analyzed_inventory, inventory_summaries = _run_inventory_analysis(
        runner, legacy_data["inventory"], collect_details=collect_details,
    )
    legacy_data["inventory"] = analyzed_inventory

    cluster_map, cluster_analysis = _run_clustering(
        runner, legacy_data["stores"], analyzed_inventory, collect_details=collect_details,
    )
    candidates = build_candidate_frame(base_recommendations, analyzed_inventory)
    candidates = _drop_empty_auto_vhs_columns(candidates)
    candidates = add_cluster_context(candidates, cluster_map)

    greedy_input = strip_dqn_columns(candidates)
    greedy = runner.call("heuristic_optimizer", "add_heuristic_scores", greedy_input)
    if isinstance(greedy, pd.DataFrame) and not greedy.empty:
        candidates = strip_dqn_columns(greedy)

    vhs_input = strip_dqn_columns(candidates)
    vhs_input_columns = list(vhs_input.columns)
    vhs = runner.call("varo_hybrid_score", "calculate_varo_hybrid_score", vhs_input)
    if isinstance(vhs, pd.DataFrame) and not vhs.empty:
        candidates = strip_dqn_columns(vhs)
    if "vhs_dqn_correction" in candidates.columns:
        candidates["vhs_dqn_correction"] = 0.0

    confidence_removed_columns = sorted(set(source_dqn_columns + [
        str(column) for column in candidates.columns if is_dqn_column(column)
    ]))
    confidence_input = strip_dqn_columns(candidates)
    if "vhs" in confidence_input.columns:
        confidence_input["vhs2"] = confidence_input["vhs"]
    confidence = runner.call("vhs_confidence", "add_confidence_columns", confidence_input, dqn_status=None)
    if isinstance(confidence, pd.DataFrame) and not confidence.empty:
        candidates = strip_dqn_columns(confidence)

    runner.defer(
        "varo_sensitivity.run_hybrid_score_sensitivity_analysis",
        "원본 함수는 VHS V2 이력 보정 그룹 컬럼을 요구하므로 DQN/이력 제외 원칙에 따라 보류",
    )
    runner.defer(
        "vhs_reason.get_reason_sentences",
        "원본 함수는 VHS V2 이력 보정 그룹을 전제로 하므로 현재 VHS 설명은 연결된 VHS 상황·기여도로 생성",
    )

    route_analysis = _run_routes(runner, legacy_data, collect_details=collect_details)
    transfer_frame = route_analysis.pop("transfer_frame", pd.DataFrame())
    candidates = _enrich_route_comparison(candidates, transfer_frame)
    candidates = _enrich_route_constraints(
        candidates,
        route_analysis.pop("cutline_frame", pd.DataFrame()),
        route_analysis.pop("time_window_frame", pd.DataFrame()),
    )

    transfer_for_promotion = (
        transfer_frame.copy() if isinstance(transfer_frame, pd.DataFrame) else pd.DataFrame()
    )
    promotion = _run_promotion(runner, legacy_data, transfer_for_promotion)
    candidates = _enrich_promotion(candidates, promotion)
    auto_vhs = apply_auto_vhs(candidates)
    if not auto_vhs.frame.empty:
        candidates = auto_vhs.frame
        if "services.vhs_score_engine.apply_auto_vhs" not in result.connected_algorithms:
            result.connected_algorithms.append("services.vhs_score_engine.apply_auto_vhs")
    else:
        auto_vhs = apply_auto_vhs(pd.DataFrame())

    optimality_input = strip_dqn_columns(candidates)
    optimality: dict[str, Any] = {
        "status": "지연 실행",
        "message": "분석 및 검증 페이지에서 계산합니다.",
        "comparable_candidate_count": 0,
    }
    legacy_validation: dict[str, Any] = {}
    if collect_details:
        has_cost_input = any(
            column in optimality_input.columns
            for column in ("estimated_cost", "transport_cost")
        )
        if has_cost_input:
            optimality_raw = runner.call(
                "varo_optimality_gap", "calculate_optimality_gap", optimality_input, k=5
            )
        else:
            optimality_raw = {}
            runner.defer(
                "varo_optimality_gap.calculate_optimality_gap",
                "이동비용 입력 컬럼이 없어 비교를 보류했습니다.",
            )
        optimality = sanitize_optimality_result(
            optimality_raw if isinstance(optimality_raw, dict) else {},
            len(optimality_input),
        )
        validation_output = runner.call(
            "varo_validation", "build_validation_report", strip_dqn_columns(candidates),
            legacy_data["stores"], legacy_data["products"], analyzed_inventory,
        )
        legacy_validation = validation_output if isinstance(validation_output, dict) else {}

    candidates = _finalize_candidate_columns(candidates)
    try:
        standard_recommendations = recommendations_from_dataframe(candidates)
    except ValueError as exc:
        result.warnings.append(f"분석 결과 표준화에 실패해 업로드 추천 구조를 유지했습니다: {exc}")
        standard_recommendations = base_recommendations

    result.recommendations = standard_recommendations
    result.top5 = top_recommendations(standard_recommendations, limit=5)
    result.vhs_analysis = (
        _vhs_analysis(candidates, runner, vhs_input_columns)
        if collect_details and not candidates.empty and "vhs" in candidates.columns else {}
    )
    if auto_vhs.analysis:
        result.vhs_analysis.update(auto_vhs.analysis)
        result.vhs_weight_analysis = auto_vhs.analysis
    result.vhs_greedy_dqn_comparison = list(auto_vhs.comparison_rows)
    result.greedy_analysis = build_greedy_provenance(candidates) if collect_details else {}
    if collect_details:
        result.greedy_analysis.update({
            "rows": _dataframe_records(candidates[[column for column in [
                "route_id", "product_name", "greedy_rank", "heuristic_score",
                "cost_score", "quantity_score", "strategy_score", "reason_bonus",
                "greedy_action", "greedy_strategy", "varo_action", "strategy_match",
                "vhs_rank", "vhs_score", "varo_final_decision", "vhs_vs_greedy_match",
                "greedy_reason",
            ] if column in candidates.columns]]),
            "dqn_status": "미연결",
        })
        result.greedy_analysis["comparison_rows"] = result.vhs_greedy_dqn_comparison
    pareto_rows = [
        {
            "route_id": item.get("route_id"),
            "product_name": item.get("product_name"),
            "pareto_rank": item.get("pareto_rank"),
            "pareto_status": item.get("pareto_status"),
            "pareto_reason": item.get("pareto_reason"),
        }
        for item in standard_recommendations
    ]
    result.pareto_analysis = {
        "status": "보조 검증",
        "comparison_count": len(pareto_rows),
        "non_dominated_count": sum(1 for item in pareto_rows if item.get("pareto_rank") == 1),
        "criteria": ["절감액", "폐기 위험", "수요 적합도", "경로 비용", "실행 가능성"],
        "rows": pareto_rows if collect_details else [],
    }
    if "services.vhs_score_engine.pareto_ranks" not in result.connected_algorithms:
        result.connected_algorithms.append("services.vhs_score_engine.pareto_ranks")
    result.confidence_analysis = (
        confidence_provenance(candidates, confidence_removed_columns)
        if collect_details else {}
    )
    result.route_analysis = route_analysis
    result.promotion_analysis = {
        "status": "연결" if not promotion.empty else "프로모션 비교 보류",
        "calculation_function": "promotion_analyzer.analyze_promotion_vs_transfer",
        "rows": _dataframe_records(promotion) if collect_details else [],
        "recommendation_count": int(len(promotion)),
        "input_columns": [
            "suggested_qty", "estimated_cost", "unit_cost", "daily_holding_cost",
        ],
    }
    result.demand_analysis = {
        "demand_forecast": inventory_summaries.get("demand_forecast", {}),
        "safety_stock": inventory_summaries.get("safety_stock", {}),
        "eoq": inventory_summaries.get("eoq", {}),
        "store_product_match": inventory_summaries.get("store_product_match", {}),
        "store_clustering": cluster_analysis,
    }
    result.risk_analysis = {
        "abc": inventory_summaries.get("abc", {}),
        "turnover": inventory_summaries.get("turnover", {}),
        "disposal_risk": inventory_summaries.get("disposal_risk", {}),
    }
    result.validation_report.update({
        "legacy_validation": legacy_validation,
        "optimality_gap": optimality,
        "calculation_sources": {
            "vhs": "services.vhs_score_engine.apply_auto_vhs",
            "legacy_vhs_reference": "varo_hybrid_score.calculate_varo_hybrid_score",
            "greedy": "heuristic_optimizer.add_heuristic_scores",
            "pareto": "services.vhs_score_engine.pareto_ranks",
            "optimality_gap": "varo_optimality_gap.calculate_optimality_gap",
            "confidence": "vhs_confidence.add_confidence_columns · DQN 제외",
            "routes": "transfer_path_analyzer.analyze_direct_vs_dc_transfer",
        },
        "connected_algorithms": list(result.connected_algorithms),
        "deferred_algorithms": list(result.deferred_algorithms),
        "warnings": list(result.warnings),
        "dqn_exclusion": result.excluded_dqn_artifacts,
        "vhs_component_quality": list(auto_vhs.analysis.get("weight_rows") or []),
        "pareto_validation": dict(result.pareto_analysis),
    })
    if collect_details:
        result.vhs_neutral_analysis = vhs_neutral_summary({"vhs_analysis": result.vhs_analysis})
        result.sensitivity_analysis = {
            "calculation_basis": "V2 내부 추천 결과 기준 (원본 varo_sensitivity는 보류)",
            "rows": sensitivity_summary(standard_recommendations),
        }
        result.reason_analysis = {
            "calculation_basis": "V2 내부 결과 기준 rule-based (원본 vhs_reason은 보류)",
            "reasons": recommendation_reasons(standard_recommendations),
        }
    result.v2_summary_functions = list(V2_SUMMARY_FUNCTIONS)

    result.summary = calculate_overview_kpis(standard_recommendations, validation)
    result.summary.update({
        "recommendation_count": len(standard_recommendations),
        "connected_algorithm_count": len(result.connected_algorithms),
        "deferred_algorithm_count": len(result.deferred_algorithms),
        "sources": kpi_sources(),
        "calculation_basis": "일부 알고리즘 연결 결과 기준",
    })
    # varo_sensitivity and vhs_reason are deferred by design (history/DQN-excluded
    # path), so they must not downgrade the status. Only real execution failures
    # (technical errors) make the result partial.
    core_failed = bool(runner.technical_errors) or not standard_recommendations
    result.status = "success" if not core_failed else "partial"
    if result.status == "success":
        result.result_basis = "실제 V2 내부 알고리즘 재계산 결과 기준"
    elif result.connected_algorithms:
        result.result_basis = "일부 알고리즘 연결 결과 기준"
    else:
        result.result_basis = "업로드 사전 계산 추천 결과 기준"
    result.summary["calculation_basis"] = result.result_basis
    result.validation_report["final_status"] = result.status
    result.validation_report["result_basis"] = result.result_basis
    result.validation_report["recommendation_count"] = len(standard_recommendations)
    result.validation_report["comparable_candidate_count"] = optimality.get(
        "comparable_candidate_count", 0
    )
    result.diagnostics = {
        "status": result.status,
        "legacy_root_available": any(available_legacy_algorithms().values()),
        "result_basis": result.result_basis,
        "dqn_artifacts_read": False,
        "detail_level": "full" if collect_details else "core",
        "deferred_until_validation": [] if collect_details else [
            "route_report_details", "optimality_gap", "legacy_validation",
            "sensitivity", "recommendation_reasons",
        ],
        "algorithm_errors": list(runner.technical_errors),
    }
    return result


def ensure_recommendations(data: Dict[str, Any]) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Guarantee a recommendations frame, generating V2 candidates if absent.

    Returns (data, recommendation_source, candidate_info) where source is one of
    "uploaded", "generated", or "none".
    """
    from services.candidate_generator import generate_candidates

    recommendations = data.get("recommendations")
    has_recs = isinstance(recommendations, pd.DataFrame) and not recommendations.empty
    if has_recs:
        return data, "uploaded", {}
    generated, info = generate_candidates(data)
    if isinstance(generated, pd.DataFrame) and not generated.empty:
        enriched = dict(data)
        enriched["recommendations"] = generated
        return enriched, "generated", info
    return data, "none", info


def build_v2_state(
    data: Dict[str, pd.DataFrame], *, detail_level: str = "core",
) -> Dict[str, object]:
    data, rec_source, candidate_info = ensure_recommendations(data)
    validation = validate_workbook_data(data)
    if validation.has_errors:
        return {
            "validation": validation,
            "recommendations": [],
            "pipeline_result": PipelineResult(status="validation_error").to_dict(),
            "recommendation_source": rec_source,
            "candidate_info": candidate_info,
        }
    result = run_analysis_pipeline(
        data, detail_level=detail_level, validation_report=validation,
    )
    return {
        "validation": validation,
        "recommendations": result.recommendations,
        "pipeline_result": result.to_dict(),
        "recommendation_source": rec_source,
        "candidate_info": candidate_info,
    }


def sort_recommendations(recommendations: List[Mapping[str, object]]) -> List[Dict[str, object]]:
    def rank_key(item: Mapping[str, object]) -> tuple[float, float, float]:
        raw_rank = (
            item.get("varo_final_rank")
            or item.get("vhs_rank")
            or item.get("rank")
            or item.get("recommendation_rank")
        )
        if raw_rank is not None:
            try:
                return (float(raw_rank), 0.0, 0.0)
            except (TypeError, ValueError):
                pass
        vhs = float(item.get("vhs_score") or 0)
        saving = float(item.get("expected_saving") or 0)
        return (999999.0, -vhs, -saving)
    return [dict(item) for item in sorted(recommendations, key=rank_key)]


def top_recommendations(recommendations: List[Mapping[str, object]], limit: int = 5) -> List[Dict[str, object]]:
    return sort_recommendations(recommendations)[:limit]


def calculate_overview_kpis(recommendations: List[Mapping[str, object]], validation: Optional[ValidationReport | Dict[str, object]] = None) -> Dict[str, object]:
    total_qty = sum(float(item.get("recommended_qty") or 0) for item in recommendations)
    route_count = len(recommendations)
    total_saving = sum(float(item.get("expected_saving") or 0) for item in recommendations)
    vhs_values = [float(item.get("vhs_score")) for item in recommendations if item.get("vhs_score") is not None]
    average_vhs = sum(vhs_values) / len(vhs_values) if vhs_values else None
    if isinstance(validation, ValidationReport):
        quality = validation.status
    elif isinstance(validation, dict):
        quality = str(validation.get("status") or "검증 전")
    else:
        quality = "검증 전"
    return {
        "total_recommended_qty": total_qty,
        "active_route_count": route_count,
        "total_expected_saving": total_saving,
        "average_vhs_score": average_vhs,
        "data_quality": quality,
    }


def summarize_loaded_data(data: Dict[str, pd.DataFrame], validation: ValidationReport | None = None) -> Dict[str, int]:
    if validation and validation.summary:
        return {
            "점포 수": validation.summary.get("store_count", 0), "DC 수": validation.summary.get("dc_count", 0),
            "상품 수": validation.summary.get("product_count", 0), "재고 행 수": validation.summary.get("inventory_count", 0),
            "경로 수": validation.summary.get("route_count", 0), "추천 결과 수": validation.summary.get("recommendation_count", 0),
            "DIRECT 추천 수": validation.summary.get("direct_count", 0), "VIA_DC 추천 수": validation.summary.get("via_dc_count", 0),
        }
    stores, products = data.get("stores", pd.DataFrame()), data.get("products", pd.DataFrame())
    inventory, routes, recs = data.get("inventory", pd.DataFrame()), data.get("routes", pd.DataFrame()), data.get("recommendations", pd.DataFrame())
    node_type = stores.get("node_type", pd.Series(dtype=str)).astype(str).str.upper()
    route_type = recs.get("route_type", pd.Series(dtype=str)).astype(str).str.upper()
    return {
        "점포 수": int((node_type == "STORE").sum()), "DC 수": int((node_type == "DC").sum()),
        "상품 수": int(len(products)), "재고 행 수": int(len(inventory)), "경로 수": int(len(routes)),
        "추천 결과 수": int(len(recs)), "DIRECT 추천 수": int((route_type == "DIRECT").sum()),
        "VIA_DC 추천 수": int((route_type == "VIA_DC").sum()),
    }


def find_recommendation(recommendations: List[Mapping[str, object]], route_id: Optional[str]) -> Optional[Dict[str, object]]:
    if route_id:
        for item in recommendations:
            if item.get("route_id") == route_id:
                return dict(item)
    sorted_items = sort_recommendations(recommendations)
    return sorted_items[0] if sorted_items else None
