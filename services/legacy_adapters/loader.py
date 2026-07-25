"""Allowlisted, read-only loading of legacy Varo algorithm modules."""
from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from services.dqn_guard import BLOCKED_DQN_MODULES

_CPU_COUNT = os.cpu_count() or 2
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, _CPU_COUNT - 1)))

LEGACY_ALGORITHM_ALLOWLIST = {
    "abc_analyzer": ("analyze_abc", "get_abc_summary"),
    "turnover_analyzer": ("analyze_turnover", "get_turnover_summary"),
    "disposal_risk_analyzer": ("analyze_disposal_risk", "get_disposal_risk_summary"),
    "demand_forecast_analyzer": ("analyze_demand_forecast", "get_demand_forecast_summary"),
    "safety_stock_analyzer": ("analyze_safety_stock", "get_safety_stock_summary"),
    "eoq_analyzer": ("analyze_eoq", "get_eoq_summary"),
    "store_product_matcher": ("analyze_store_product_matching", "get_matching_summary"),
    "store_clustering": ("analyze_store_clustering", "add_cluster_to_recommendations"),
    "heuristic_optimizer": ("add_heuristic_scores", "select_greedy_best_candidate"),
    "varo_hybrid_score": ("calculate_varo_hybrid_score", "get_vhs_summary"),
    "vhs_confidence": ("add_confidence_columns", "get_confidence_summary"),
    "varo_sensitivity": ("run_hybrid_score_sensitivity_analysis", "build_sensitivity_stability_report"),
    "varo_optimality_gap": ("calculate_optimality_gap", "build_optimality_comparison_table"),
    "varo_validation": ("build_validation_report", "calculate_validation_metrics"),
    "transfer_path_analyzer": ("analyze_direct_vs_dc_transfer",),
    "network_path_analyzer": ("analyze_multi_store_network_paths",),
    "route_analyzer": ("analyze_dc_retailer_routes",),
    "cutline_analyzer": ("analyze_product_distance_cutline",),
    "time_window_analyzer": ("analyze_trade_time_windows",),
    "min_cost_network": ("analyze_min_cost_network", "add_network_score_to_recommendations"),
    "promotion_analyzer": ("analyze_promotion_vs_transfer",),
}


class LegacyAlgorithmUnavailable(RuntimeError):
    """Raised when an approved legacy module cannot be safely loaded."""


# Self-contained default: an in-project directory. Approved non-DQN algorithm
# modules may optionally be dropped here to enable full recomputation. When it
# is empty (the default), every algorithm is reported unavailable and the
# pipeline degrades gracefully to the uploaded pre-computed values.
LOCAL_ALGORITHM_DIR = Path(__file__).resolve().parent / "_local_modules"

# Hard guard: the loader must never resolve to the original/backup source tree,
# even if VARO_LEGACY_PATH is misconfigured to point at it.
_FORBIDDEN_ROOT_TOKENS = ("bad_inventory_simulator",)


def _is_forbidden_root(path: Path) -> bool:
    text = str(path).replace("\\", "/").lower()
    return any(token in text for token in _FORBIDDEN_ROOT_TOKENS)


def get_legacy_root() -> Path:
    """Resolve the approved-algorithm source root.

    Defaults to a varo_v2-internal directory so the app stays fully
    self-contained and never reaches outside the project. A ``VARO_LEGACY_PATH``
    override is honored only when it does not point at the forbidden
    original/backup tree.
    """
    configured = os.environ.get("VARO_LEGACY_PATH")
    if configured:
        candidate = Path(configured).expanduser()
        if not _is_forbidden_root(candidate):
            return candidate
    return LOCAL_ALGORITHM_DIR


def legacy_module_path(module_name: str) -> Path:
    if module_name in BLOCKED_DQN_MODULES:
        raise LegacyAlgorithmUnavailable(f"DQN 모듈은 이번 단계에서 차단됩니다: {module_name}")
    if module_name not in LEGACY_ALGORITHM_ALLOWLIST:
        raise LegacyAlgorithmUnavailable(f"허용되지 않은 원본 모듈입니다: {module_name}")
    return get_legacy_root() / f"{module_name}.py"


@lru_cache(maxsize=None)
def load_legacy_module(module_name: str) -> ModuleType:
    """Load only one allowlisted Python source file; never scan artifact files."""
    path = legacy_module_path(module_name)
    if not path.is_file():
        raise LegacyAlgorithmUnavailable(f"원본 알고리즘 파일이 없습니다: {path.name}")
    spec = importlib.util.spec_from_file_location(f"_varo_legacy_{module_name}", path)
    if spec is None or spec.loader is None:
        raise LegacyAlgorithmUnavailable(f"원본 모듈을 불러올 수 없습니다: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in LEGACY_ALGORITHM_ALLOWLIST[module_name] if not hasattr(module, name)]
    if missing:
        raise LegacyAlgorithmUnavailable(f"{path.name}에 함수가 없습니다: {', '.join(missing)}")
    return module


def available_legacy_algorithms() -> dict[str, bool]:
    root = get_legacy_root()
    return {name: (root / f"{name}.py").is_file() for name in LEGACY_ALGORITHM_ALLOWLIST}
