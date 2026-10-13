"""Sales semantics, inventory turnover and inventory risk metric integrity (P1 / T4).

A validation contract that runs next to the legacy inventory analyzers (ABC, turnover, disposal risk, safety stock,
store-product matching, the V1 forecast inputs). It never changes their formulas, thresholds or outputs, and it never
changes a recommendation, rank, action or KPI.

What it separates:
* the meaning of a quantity: observed retail sales, observed outbound flow, observed transfer, observed order,
  forecast demand, proxy demand, a user estimate, or unknown (``SIGNAL_TYPES``); the evidence for that meaning
  (``SIGNAL_EVIDENCE``). A meaning is never assigned without evidence.
* the period of a quantity: one day, a rolling 7 / 30-day sum, a per-day mean, a forecast horizon total, a calendar
  month, an inventory snapshot. A 30-day sum is never used as a daily value, and the observed days are counted.
* a day without a record (unknown), a NULL value (missing), an observed zero, a product that is not carried or not on
  sale, and a day outside the period (``DAY_STATES``). None of them is turned into another, and none becomes 0.
* negatives: a documented return / cancellation / settlement adjustment is netted with its label; an unexplained
  negative is an invalid value. A negative is never abs()'d or clipped.
* turnover: physical unit turnover (sales units / average units held over the same period) and financial turnover
  (cost of goods sold / average inventory value). Retail sales with observed inventory is the only real-sales basis;
  outbound flow and forecasts are proxies and stay labelled as such (``TURNOVER_BASES``).
* days of supply (what the legacy ``turnover_days`` / ``turnover_grade`` measure) versus the turnover ratio.

The legacy turnover grade thresholds are read from the legacy module (``turnover_analyzer._assign_grade``) and the ABC
cut-offs from ``abc_analyzer``; nothing here sets a new threshold.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from services.column_aliases import INVENTORY_ALIASES, _canonical
from services.legacy_adapters.loader import LegacyAlgorithmUnavailable, load_legacy_module

CONTRACT_VERSION = "1.0.0"

# ------------------------------------------------------------------------------------------------ signal types (§3)

OBSERVED_RETAIL_SALES = "OBSERVED_RETAIL_SALES"
OBSERVED_OUTBOUND_FLOW = "OBSERVED_OUTBOUND_FLOW"
OBSERVED_TRANSFER = "OBSERVED_TRANSFER"
OBSERVED_ORDER = "OBSERVED_ORDER"
FORECAST_DEMAND = "FORECAST_DEMAND"
PROXY_DEMAND = "PROXY_DEMAND"
USER_ESTIMATED_DEMAND = "USER_ESTIMATED_DEMAND"
UNKNOWN_SIGNAL = "UNKNOWN"
SIGNAL_TYPES = {
    OBSERVED_RETAIL_SALES: "units sold to end customers, recorded by the point of sale",
    OBSERVED_OUTBOUND_FLOW: "units shipped out of a node (logistics outbound); not retail sales",
    OBSERVED_TRANSFER: "units moved between the seller's own nodes",
    OBSERVED_ORDER: "units requested / ordered (purchase or replenishment request); not a receipt and not demand",
    FORECAST_DEMAND: "model forecast of future demand",
    PROXY_DEMAND: "an observed quantity of another kind (outbound, regional flow, order) used in place of demand",
    USER_ESTIMATED_DEMAND: "a demand value entered by a person without an observation behind it",
    UNKNOWN_SIGNAL: "no evidence for what the quantity measures",
}
# Evidence for a signal type, strongest first.
SOURCE_DOCUMENTED = "SOURCE_DOCUMENTED"        # official dataset documentation / canonical mapping notes
DECLARED_SEMANTICS = "DECLARED_SEMANTICS"      # an explicit <column>_semantics / <column>_provenance in the upload
COLUMN_HEADER_CLAIM = "COLUMN_HEADER_CLAIM"    # only the column name says what the quantity is (unverified claim)
NO_EVIDENCE = "NO_EVIDENCE"
SIGNAL_EVIDENCE = (SOURCE_DOCUMENTED, DECLARED_SEMANTICS, COLUMN_HEADER_CLAIM, NO_EVIDENCE)
# Signal type -> the role it can play as demand. Only observed retail sales is demand evidence (censoring aside).
DEMAND_ROLE = {OBSERVED_RETAIL_SALES: "OBSERVED_SALES", OBSERVED_OUTBOUND_FLOW: "PROXY", OBSERVED_TRANSFER: "PROXY",
               OBSERVED_ORDER: "NOT_DEMAND", FORECAST_DEMAND: "FORECAST", PROXY_DEMAND: "PROXY",
               USER_ESTIMATED_DEMAND: "ESTIMATE", UNKNOWN_SIGNAL: "UNKNOWN"}

# ------------------------------------------------------------------------------------------------ periods (§5)

DAILY_QUANTITY = "DAILY_QUANTITY"          # the quantity of one calendar day
WINDOW_SUM = "WINDOW_SUM"                  # sum over period_days consecutive days
DAILY_MEAN = "DAILY_MEAN"                  # mean per day over a window (the window may be undeclared)
FORECAST_SUM = "FORECAST_SUM"              # forecast total over a horizon
CALENDAR_MONTH_SUM = "CALENDAR_MONTH_SUM"  # sum over one calendar month
SNAPSHOT = "SNAPSHOT"                      # an inventory level at one date
PERIOD_UNDECLARED = "PERIOD_UNDECLARED"    # a quantity whose period nothing states
AGGREGATIONS = (DAILY_QUANTITY, WINDOW_SUM, DAILY_MEAN, FORECAST_SUM, CALENDAR_MONTH_SUM, SNAPSHOT, PERIOD_UNDECLARED)
PER_DAY_AGGREGATIONS = (DAILY_QUANTITY, DAILY_MEAN)
# The legacy turnover / disposal / ABC formulas are written on a 30-day sales window (sales_30d / 30); a rate measured
# on a shorter or undeclared window is reported with lower confidence. This is the legacy design, not a new threshold.
LEGACY_RATE_WINDOW_DAYS = 30

# ------------------------------------------------------------------------------------------------ day states (§7)

OBSERVED_ZERO = "OBSERVED_ZERO"        # A: the source records a zero for the day
NO_RECORD = "NO_RECORD"                # B: the source has no row for the day (unknown: zero, closed or not recorded)
NOT_SELLABLE = "NOT_SELLABLE"          # C: the product is not on sale that day (declared)
NOT_CARRIED = "NOT_CARRIED"            # D: the location does not carry the product (declared)
SYSTEM_MISSING = "SYSTEM_MISSING"      # E: a row exists but the quantity is NULL / unparseable
OUT_OF_PERIOD = "OUT_OF_PERIOD"        # F: the day lies outside the observation period
OBSERVED_VALUE = "OBSERVED_VALUE"      # a finite non-zero observation
DAY_STATES = {OBSERVED_VALUE: "finite non-zero observation", OBSERVED_ZERO: "A: observed zero",
              NO_RECORD: "B: no record (never read as zero)", NOT_SELLABLE: "C: product not on sale (declared)",
              NOT_CARRIED: "D: location does not carry the product (declared)",
              SYSTEM_MISSING: "E: record with a NULL / unparseable quantity", OUT_OF_PERIOD: "F: outside the period"}
DECLARED_DAY_STATES = (NOT_SELLABLE, NOT_CARRIED)

# zero semantics of a source (how a zero and an absent day are to be read)
ZERO_RECORDED = "ZERO_RECORDED"                              # zero days are rows with 0
ZERO_ROWS_ABSENT = "ZERO_ROWS_ABSENT"                        # the source omits zero days: an absent day is unknown
ZERO_RECORDED_POSSIBLY_CENSORED = "ZERO_RECORDED_POSSIBLY_CENSORED"    # zero sales may be a stock-out (no inventory)
ZERO_RECORDED_AVAILABILITY_KNOWN = "ZERO_RECORDED_AVAILABILITY_KNOWN"  # zero sales with stock-out hours recorded
ZERO_SEMANTICS_UNDECLARED = "ZERO_SEMANTICS_UNDECLARED"
ZERO_SEMANTICS = (ZERO_RECORDED, ZERO_ROWS_ABSENT, ZERO_RECORDED_POSSIBLY_CENSORED, ZERO_RECORDED_AVAILABILITY_KNOWN,
                  ZERO_SEMANTICS_UNDECLARED)

# ------------------------------------------------------------------------------------------------ negatives (§8)

RETURN_DOCUMENTED = "RETURN_DOCUMENTED"
CANCELLATION = "CANCELLATION"
SETTLEMENT_ADJUSTMENT = "SETTLEMENT_ADJUSTMENT"
SOURCE_ERROR = "SOURCE_ERROR"
NEGATIVE_UNKNOWN = "NEGATIVE_UNKNOWN"
NO_NEGATIVES = "NO_NEGATIVES"
NEGATIVE_SEMANTICS = (RETURN_DOCUMENTED, CANCELLATION, SETTLEMENT_ADJUSTMENT, SOURCE_ERROR, NEGATIVE_UNKNOWN, NO_NEGATIVES)
NETTABLE_NEGATIVES = (RETURN_DOCUMENTED, CANCELLATION, SETTLEMENT_ADJUSTMENT)

# ------------------------------------------------------------------------------------------------ quality gate (§25)

VALID = "VALID"
VALID_WITH_LIMITATION = "VALID_WITH_LIMITATION"
PROXY_ONLY = "PROXY_ONLY"
MISSING_DATA = "MISSING_DATA"
UNIT_MISMATCH = "UNIT_MISMATCH"
PERIOD_MISMATCH = "PERIOD_MISMATCH"
INVALID_VALUE = "INVALID_VALUE"
NOT_COMPARABLE = "NOT_COMPARABLE"
GATE_SEVERITY = (INVALID_VALUE, UNIT_MISMATCH, PERIOD_MISMATCH, NOT_COMPARABLE, MISSING_DATA, PROXY_ONLY,
                 VALID_WITH_LIMITATION, VALID)
GATE_STATUSES = {
    VALID: "observed, declared period and unit, complete",
    VALID_WITH_LIMITATION: "usable with a stated limitation (undeclared window, unknown unit of one source, partial period)",
    PROXY_ONLY: "usable only as a proxy analysis (outbound, forecast or estimate in place of sales)",
    MISSING_DATA: "the value is absent; it is never replaced by 0",
    UNIT_MISMATCH: "units differ or cannot be verified to be the same",
    PERIOD_MISMATCH: "the value's period differs from the field's declared period, or two periods are mixed",
    INVALID_VALUE: "duplicate day, unexplained negative, non-finite or contradictory value",
    NOT_COMPARABLE: "different signal types or datasets combined",
}
# Existing canonical quality codes (services.canonical_schema.QUALITY_CODES) -> gate; reused, not redefined.
CANONICAL_CODE_GATE = {"MISSING_KEY": MISSING_DATA, "MISSING_VALUE": MISSING_DATA, "SOURCE_ANOMALY": INVALID_VALUE,
                       "NEGATIVE_QUANTITY": INVALID_VALUE, "DUPLICATE_OBSERVATION": INVALID_VALUE,
                       "UNKNOWN_UNIT": VALID_WITH_LIMITATION, "UNKNOWN_LOCATION": MISSING_DATA,
                       "AMBIGUOUS_PRODUCT": VALID_WITH_LIMITATION, "BENCHMARK_PROXY": PROXY_ONLY,
                       "OUT_OF_SCOPE": NOT_COMPARABLE, "VALID": VALID}


def combine_gate(*statuses: str | None) -> str:
    """The most severe of the given gate statuses (VALID when none is given)."""
    present = [s for s in statuses if s]
    return min(present, key=GATE_SEVERITY.index) if present else VALID


def canonical_flags_gate(validation_flags: str | None, quality_flags: str | None = None) -> str:
    """Gate of a canonical row from its existing flags; a documented return alone is a limitation, not invalid."""
    flags = [f for f in str(validation_flags or "").split("|") if f]
    codes = [c for c in str(quality_flags or "").split("|") if c]
    gates = [CANONICAL_CODE_GATE.get(code, INVALID_VALUE) for code in codes]
    documented = any(f.startswith("documented_return:") for f in flags)
    if documented and "NEGATIVE_QUANTITY" in codes:
        gates = [g for code, g in zip(codes, gates) if code != "NEGATIVE_QUANTITY"] + [VALID_WITH_LIMITATION]
    return combine_gate(*gates)


# ------------------------------------------------------------------------------------------------ turnover (§14-18)

TURNOVER_AVAILABLE = "TURNOVER_AVAILABLE"
ZERO_TURNOVER = "ZERO_TURNOVER"                    # observed zero sales and a positive average inventory
TURNOVER_UNAVAILABLE = "TURNOVER_UNAVAILABLE"      # sales missing
DIVISION_BY_ZERO = "DIVISION_BY_ZERO"              # average inventory is 0
AVG_INVENTORY_UNAVAILABLE = "AVG_INVENTORY_UNAVAILABLE"
PARTIAL_PERIOD = "PARTIAL_PERIOD"
NET_SALES_NEGATIVE = "NET_SALES_NEGATIVE"
NOT_APPLICABLE = "NOT_APPLICABLE"                  # the location does not carry the product
DAYS_OF_SUPPLY_ONLY = "DAYS_OF_SUPPLY_ONLY"        # a per-day rate without a period total: no ratio, days of supply only
TURNOVER_STATUSES = (TURNOVER_AVAILABLE, ZERO_TURNOVER, TURNOVER_UNAVAILABLE, DIVISION_BY_ZERO, AVG_INVENTORY_UNAVAILABLE,
                     PARTIAL_PERIOD, PERIOD_MISMATCH, UNIT_MISMATCH, INVALID_VALUE, NET_SALES_NEGATIVE, NOT_COMPARABLE,
                     NOT_APPLICABLE, DAYS_OF_SUPPLY_ONLY)
REAL_SALES_TURNOVER = "REAL_SALES_TURNOVER"              # path A: observed retail sales + observed inventory
FLOW_BASED_TURNOVER_PROXY = "FLOW_BASED_TURNOVER_PROXY"  # path B: logistics outbound + observed inventory
FORECAST_DEMAND_PROXY = "FORECAST_DEMAND_PROXY"          # path C: forecast demand (+ estimated inventory)
USER_ESTIMATE_PROXY = "USER_ESTIMATE_PROXY"
NO_BASIS = "NO_BASIS"
TURNOVER_BASES = {REAL_SALES_TURNOVER: "A: observed retail sales / observed average inventory (real-sales candidate)",
                  FLOW_BASED_TURNOVER_PROXY: "B: logistics outbound / observed average inventory (proxy; not sales)",
                  FORECAST_DEMAND_PROXY: "C: forecast demand / inventory (proxy)",
                  USER_ESTIMATE_PROXY: "user-estimated demand / inventory (proxy)",
                  NO_BASIS: "no usable numerator"}
SIGNAL_TURNOVER_BASIS = {OBSERVED_RETAIL_SALES: REAL_SALES_TURNOVER, OBSERVED_OUTBOUND_FLOW: FLOW_BASED_TURNOVER_PROXY,
                         OBSERVED_TRANSFER: FLOW_BASED_TURNOVER_PROXY, FORECAST_DEMAND: FORECAST_DEMAND_PROXY,
                         USER_ESTIMATED_DEMAND: USER_ESTIMATE_PROXY}
FINANCIAL_TURNOVER_AVAILABLE = "FINANCIAL_TURNOVER_AVAILABLE"
FINANCIAL_TURNOVER_UNAVAILABLE = "FINANCIAL_TURNOVER_UNAVAILABLE"

# days of supply (legacy turnover_days)
DOS_AVAILABLE = "DAYS_OF_SUPPLY_AVAILABLE"
DOS_NO_MOVEMENT = "NO_MOVEMENT_OBSERVED"       # rate 0, stock > 0: infinite days of supply
DOS_ZERO_STOCK = "ZERO_STOCK"                   # stock 0, rate > 0: stocked out, not fast-moving
DOS_NO_STOCK_NO_MOVEMENT = "NO_STOCK_NO_MOVEMENT"
DOS_RATE_UNAVAILABLE = "RATE_UNAVAILABLE"
DOS_STOCK_UNAVAILABLE = "STOCK_UNAVAILABLE"
DOS_NEGATIVE_RATE = "NEGATIVE_NET_RATE"
CLASSIFIABLE_DOS = (DOS_AVAILABLE, DOS_NO_MOVEMENT)

UNCLASSIFIED = "UNCLASSIFIED"
CONFIDENCE_LEVELS = ("HIGH", "MEDIUM", "LOW", "NONE")
SCOPE_REAL = "REAL_SALES"
SCOPE_CLAIMED = "SALES_CLAIMED_BY_HEADER"
SCOPE_PROXY = "PROXY_ONLY"
SCOPE_NONE = "NONE"
LEGACY_GRADE_SOURCE = ("turnover_analyzer._assign_grade on days of supply (<=30 FAST, <=60 NORMAL, <=90 SLOW, else DEAD); "
                       "legacy heuristic thresholds, not operationally validated")

# ------------------------------------------------------------------------------------------------ reason codes

REASONS = {
    # signal / window
    "SIGNAL_TYPE_UNKNOWN": "no evidence for what the quantity measures",
    "SIGNAL_TYPE_MISMATCH": "records of different signal types cannot be combined",
    "DATASET_MISMATCH": "records of different datasets cannot be combined",
    "UNIT_MISMATCH": "quantity units differ",
    "UNIT_UNVERIFIABLE": "one unit is declared and the other is unknown",
    "UNIT_UNKNOWN_SAME_DATASET": "both units unknown; same dataset and source column family (not verified equal)",
    "UNIT_UNKNOWN": "the quantity unit is unknown",
    "PROXY_SIGNAL": "the quantity is not retail sales (outbound / forecast / estimate used as a proxy)",
    "ORDER_IS_NOT_OUTFLOW": "an order / request quantity is not an outflow from inventory",
    "SEMANTICS_FROM_COLUMN_NAME_ONLY": "only the column header claims the quantity is sales",
    "DUPLICATE_DATE": "the same day occurs more than once; the value is neither summed nor picked",
    "FUTURE_DATA_EXCLUDED": "records dated after as_of were excluded",
    "WINDOW_AFTER_AS_OF": "the window ends after as_of",
    "OUT_OF_PERIOD_IGNORED": "records outside the window were ignored",
    "PARTIAL_OBSERVATION": "fewer observed days than window days; absent days are not zero",
    "NO_OBSERVATION": "no observed day in the window",
    "SYSTEM_MISSING_DAYS": "records with a NULL quantity (missing, not zero)",
    "NO_RECORD_DAYS": "window days without a record (unknown, not zero)",
    "NOT_CARRIED_NOT_ZERO": "the location does not carry the product; not a zero-demand location",
    "NOT_SELLABLE_EXCLUDED": "days declared not on sale are excluded, not zero",
    "RETURNS_NETTED_DOCUMENTED": "negative quantities are documented returns / cancellations / adjustments and are netted",
    "NEGATIVE_UNRESOLVED": "negative quantity without documented semantics; never abs()'d or clipped",
    "ZERO_POSSIBLY_CENSORED": "zero sales may be a stock-out (availability not observed)",
    "ZERO_DURING_STOCKOUT": "zero sales on a day with recorded stock-out hours",
    "RATE_OVER_OBSERVED_DAYS": "the daily rate divides by observed days (absent days not zero-filled)",
    "AVERAGE_WINDOW_UNDECLARED": "a per-day mean whose averaging window is not declared",
    "WINDOW_COMPLETENESS_UNDECLARED": "an aggregate field whose observed days are not declared",
    "PERIOD_UNDECLARED": "the field states no period",
    "SINGLE_DAY_SAMPLE": "a one-day quantity used as a rate",
    "RATE_WINDOW_SHORTER_THAN_LEGACY_30D": "rate window shorter than the legacy 30-day sales window",
    "ALIAS_COPY_ACROSS_PERIODS": "the field is an exact copy of a field with a different period (loader alias)",
    "PROJECTED_FROM_SINGLE_DAY": "a 30-day value projected as 30 x one day's quantity",
    # inventory
    "NO_SNAPSHOT": "no inventory snapshot in the period",
    "START_SNAPSHOT_ONLY": "only the opening snapshot exists",
    "END_SNAPSHOT_ONLY": "only the closing snapshot exists",
    "SINGLE_MID_SNAPSHOT": "one snapshot inside the period",
    "PERIOD_ENDPOINT_MISSING": "the opening or the closing snapshot is missing",
    "SPARSE_SNAPSHOTS": "fewer snapshots than period days; no continuous inventory is generated",
    "DECLARED_AVERAGE_WINDOW_UNDECLARED": "an uploaded average inventory with no stated averaging window",
    "SNAPSHOT_NULL": "a snapshot with a NULL quantity (missing, not zero)",
    "NEGATIVE_INVENTORY": "negative inventory quantity",
    "STOCK_MISSING": "current stock missing (never read as 0)",
    "SINGLE_SNAPSHOT_AS_AVERAGE": "a single current stock is not an average inventory",
    # turnover
    "PERIOD_NOT_ALIGNED": "sales window and inventory period differ",
    "RATIO_NEEDS_PERIOD_TOTAL": "a per-day sales rate has no period total: the turnover ratio is not computed",
    "SALES_MISSING": "no sales quantity; turnover unavailable",
    "AVERAGE_INVENTORY_ZERO": "average inventory is 0 (division by zero)",
    "ZERO_SALES_OBSERVED": "observed zero sales with positive inventory: zero turnover",
    "NET_SALES_NEGATIVE": "net sales negative over the period",
    "ZERO_STOCK_NOT_CLASSIFIED": "stock 0: days of supply 0 is a stock-out, not a fast mover",
    "NO_STOCK_NO_MOVEMENT": "stock 0 and no movement",
    "COGS_MISSING": "no cost of goods sold: financial turnover not computed",
    "INVENTORY_VALUE_MISSING": "no inventory valuation: financial turnover not computed",
    "CURRENCY_MISMATCH": "COGS and inventory value currencies differ",
    "LEGACY_THRESHOLD_HEURISTIC": "classification uses the legacy 30/60/90-day thresholds (not operationally validated)",
    # legacy behaviour (diagnostics of the existing analyzers; they stay unchanged)
    "LEGACY_SALES_COLUMN_ABSENT_AS_ZERO": "legacy: no sales column, sales read as 0",
    "LEGACY_SALES_NULL_AS_ZERO": "legacy: NULL sales read as 0 (turnover_days 999 -> DEAD)",
    "LEGACY_NEGATIVE_SALES_NEGATIVE_DAYS": "legacy: negative sales give negative days of supply (graded FAST)",
    "LEGACY_STOCK_NULL_AS_ZERO": "legacy: NULL stock read as 0",
    "LEGACY_ZERO_STOCK_GRADED_FAST": "legacy: stock 0 with sales gives 0 days (graded FAST)",
    "LEGACY_ZERO_STOCK_ZERO_SALES_DEAD": "legacy: stock 0 and sales 0 give 999 days (graded DEAD)",
    "LEGACY_ZERO_SALES_DEAD": "legacy: zero sales give 999 days (graded DEAD)",
    "LEGACY_AVG_INVENTORY_ZERO_RATE_ZERO": "legacy: average inventory 0 gives an annual rate of 0",
    "LEGACY_SINGLE_SNAPSHOT_AS_AVERAGE": "legacy: current stock used as average inventory",
    "LEGACY_DAILY_VALUE_IN_30D_FIELD": "legacy: sales_30d holds a daily value (loader alias), rate understated 30x",
    "LEGACY_30D_FIELD_IN_DAILY_FIELD": "legacy: a daily field holds a 30-day / 7-day total (loader alias)",
    "LEGACY_30D_PROJECTED_FROM_ONE_DAY": "legacy: sales_30d = 30 x one day's outbound proxy",
    "LEGACY_PROXY_AS_SALES": "legacy: an outbound / proxy quantity is read as sales",
}


def _reasons(*codes: str) -> list[str]:
    for code in codes:
        if code not in REASONS:
            raise KeyError(f"undocumented reason code {code}")
    return list(dict.fromkeys(codes))


# ------------------------------------------------------------------------------------------------ helpers


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "<na>", "nat") else text


def _day(value: Any) -> date | None:
    text = _text(value)
    if not text:
        return None
    try:
        return pd.Timestamp(text).date()
    except (TypeError, ValueError):
        return None


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if math.isinf(value):
        return value
    return round(float(value), digits)


# ------------------------------------------------------------------------------------------------ sales signal (§4)


@dataclass(frozen=True)
class SalesSignal:
    """The minimum contract of one sales / demand quantity.

    company_id is not adopted: Varo data carries no company key, and ``dataset_id`` + location already scope a value.
    signal_period is expressed by ``aggregation`` + ``period_days``. Two quantities are compared only through
    :func:`signals_comparable`.
    """
    dataset_id: str | None
    location_id: str | None
    product_id: str | None
    quantity: float | None
    signal_type: str = UNKNOWN_SIGNAL
    aggregation: str = PERIOD_UNDECLARED
    period_days: int | None = None
    observation_start: str | None = None
    observation_end: str | None = None
    as_of_date: str | None = None
    quantity_unit: str | None = None
    observed_days: int | None = None
    is_complete_observation: bool | None = None
    zero_semantics: str = ZERO_SEMANTICS_UNDECLARED
    negative_semantics: str = NO_NEGATIVES
    semantics_evidence: str = NO_EVIDENCE
    provenance: str = ""
    source_lineage: str = ""
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.signal_type not in SIGNAL_TYPES:
            raise ValueError(f"unknown signal_type {self.signal_type}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation {self.aggregation}")
        if self.semantics_evidence not in SIGNAL_EVIDENCE:
            raise ValueError(f"unknown semantics_evidence {self.semantics_evidence}")
        if self.negative_semantics not in NEGATIVE_SEMANTICS:
            raise ValueError(f"unknown negative_semantics {self.negative_semantics}")
        if self.zero_semantics not in ZERO_SEMANTICS:
            raise ValueError(f"unknown zero_semantics {self.zero_semantics}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quality_flags"] = list(self.quality_flags)
        return data


def daily_rate(signal: SalesSignal) -> dict[str, Any]:
    """A per-day rate of a signal, only where its period supports one; the period is never guessed."""
    reasons: list[str] = []
    gate = VALID
    quantity = _finite(signal.quantity)
    if quantity is None:
        return {"value": None, "basis": None, "gate": MISSING_DATA, "reasons": _reasons("SALES_MISSING"),
                "window_days": signal.period_days}
    if signal.aggregation == PERIOD_UNDECLARED:
        return {"value": None, "basis": None, "gate": PERIOD_MISMATCH, "reasons": _reasons("PERIOD_UNDECLARED"),
                "window_days": None}
    if signal.aggregation == SNAPSHOT:
        return {"value": None, "basis": None, "gate": NOT_COMPARABLE, "reasons": _reasons("SIGNAL_TYPE_MISMATCH"),
                "window_days": None}
    window = signal.period_days
    if signal.aggregation == DAILY_QUANTITY:
        value, basis, window = quantity, "ONE_DAY", 1
        reasons.append("SINGLE_DAY_SAMPLE")
    elif signal.aggregation == DAILY_MEAN:
        value, basis = quantity, "DECLARED_DAILY_MEAN"
        if window is None:
            reasons.append("AVERAGE_WINDOW_UNDECLARED")
            gate = combine_gate(gate, VALID_WITH_LIMITATION)
    elif signal.aggregation in (WINDOW_SUM, CALENDAR_MONTH_SUM, FORECAST_SUM):
        if not window:
            return {"value": None, "basis": None, "gate": PERIOD_MISMATCH, "reasons": _reasons("PERIOD_UNDECLARED"),
                    "window_days": None}
        observed = signal.observed_days
        if signal.is_complete_observation is False and observed:
            value, basis = quantity / observed, "OBSERVED_DAYS"
            reasons += ["PARTIAL_OBSERVATION", "RATE_OVER_OBSERVED_DAYS"]
            gate = combine_gate(gate, VALID_WITH_LIMITATION)
        elif signal.is_complete_observation is False:
            return {"value": None, "basis": None, "gate": MISSING_DATA, "reasons": _reasons("NO_OBSERVATION"),
                    "window_days": window}
        else:
            value, basis = quantity / window, "WINDOW_DAYS"
            if signal.is_complete_observation is None:
                reasons.append("WINDOW_COMPLETENESS_UNDECLARED")
                gate = combine_gate(gate, VALID_WITH_LIMITATION)
    else:  # pragma: no cover - guarded by SalesSignal
        raise ValueError(signal.aggregation)
    if window is not None and window < LEGACY_RATE_WINDOW_DAYS:
        reasons.append("RATE_WINDOW_SHORTER_THAN_LEGACY_30D")
    if signal.signal_type != OBSERVED_RETAIL_SALES:
        gate = combine_gate(gate, PROXY_ONLY if signal.signal_type in SIGNAL_TURNOVER_BASIS or
                            signal.signal_type == PROXY_DEMAND else NOT_COMPARABLE)
        reasons.append("PROXY_SIGNAL" if signal.signal_type != UNKNOWN_SIGNAL else "SIGNAL_TYPE_UNKNOWN")
    if signal.quantity_unit is None:
        reasons.append("UNIT_UNKNOWN")
        gate = combine_gate(gate, VALID_WITH_LIMITATION)
    if signal.semantics_evidence == COLUMN_HEADER_CLAIM:
        reasons.append("SEMANTICS_FROM_COLUMN_NAME_ONLY")
        gate = combine_gate(gate, VALID_WITH_LIMITATION)
    return {"value": _round(value), "basis": basis, "gate": gate, "reasons": _reasons(*reasons), "window_days": window}


def signals_comparable(left: SalesSignal, right: SalesSignal) -> dict[str, Any]:
    """Whether two quantities may be compared directly (same dataset, type, unit, aggregation and period)."""
    reasons: list[str] = []
    if (left.dataset_id or None) != (right.dataset_id or None):
        reasons.append("DATASET_MISMATCH")
    if left.signal_type != right.signal_type:
        reasons.append("SIGNAL_TYPE_MISMATCH")
    if left.quantity_unit != right.quantity_unit:
        reasons.append("UNIT_MISMATCH")
    elif left.quantity_unit is None:
        reasons.append("UNIT_UNKNOWN_SAME_DATASET" if not reasons else "UNIT_UNKNOWN")
    if (left.aggregation, left.period_days) != (right.aggregation, right.period_days):
        reasons.append("PERIOD_NOT_ALIGNED")
    blocking = [r for r in reasons if r not in ("UNIT_UNKNOWN_SAME_DATASET",)]
    gate = VALID_WITH_LIMITATION if reasons and not blocking else VALID
    if "DATASET_MISMATCH" in reasons or "SIGNAL_TYPE_MISMATCH" in reasons:
        gate = NOT_COMPARABLE
    elif "UNIT_MISMATCH" in reasons:
        gate = UNIT_MISMATCH
    elif "PERIOD_NOT_ALIGNED" in reasons:
        gate = PERIOD_MISMATCH
    return {"comparable": gate in (VALID, VALID_WITH_LIMITATION), "gate": gate, "reasons": _reasons(*reasons)}


# ------------------------------------------------------------------------------------------------ daily window (§5-8)


def aggregate_daily_window(records: Iterable[Mapping[str, Any]], *, window_end: Any, window_days: int,
                           as_of: Any = None, dataset_id: str | None = None, location_id: str | None = None,
                           product_id: str | None = None, signal_type: str | None = None,
                           quantity_unit: str | None = None, zero_semantics: str = ZERO_SEMANTICS_UNDECLARED,
                           negative_semantics: str | None = None, semantics_evidence: str = NO_EVIDENCE,
                           provenance: str = "", source_lineage: str = "") -> dict[str, Any]:
    """Sum one location-product daily signal over the ``window_days`` days ending at ``window_end``.

    Records are mappings with ``date`` and ``quantity`` and optionally ``signal_type``, ``dataset_id``,
    ``quantity_unit`` and ``day_state`` (NOT_CARRIED / NOT_SELLABLE). A NULL quantity is SYSTEM_MISSING, an absent day
    is NO_RECORD: neither is 0. Mixed signal types, datasets or units are not summed; a repeated day is not summed.
    """
    end = _day(window_end)
    cutoff = _day(as_of)
    if end is None or window_days <= 0:
        raise ValueError("window_end and a positive window_days are required")
    start = end - timedelta(days=window_days - 1)
    reasons: list[str] = []
    dated = [(dict(r), _day(r.get("date"))) for r in records]
    future = sum(1 for _, d in dated if cutoff is not None and d is not None and d > cutoff)
    if future:
        reasons.append("FUTURE_DATA_EXCLUDED")
    current = [(r, d) for r, d in dated if not (cutoff is not None and d is not None and d > cutoff)]
    in_window = [r for r, d in current if d is not None and start <= d <= end]
    out_of_period = len(current) - len(in_window)
    if out_of_period:
        reasons.append("OUT_OF_PERIOD_IGNORED")
    types = {r.get("signal_type") or signal_type or UNKNOWN_SIGNAL for r in in_window} or {signal_type or UNKNOWN_SIGNAL}
    datasets = {r.get("dataset_id") or dataset_id for r in in_window} or {dataset_id}
    units = {r.get("quantity_unit", quantity_unit) for r in in_window} or {quantity_unit}
    resolved_type = next(iter(types)) if len(types) == 1 else None
    base = {"dataset_id": next(iter(datasets)) if len(datasets) == 1 else None, "location_id": location_id,
            "product_id": product_id, "signal_type": resolved_type, "aggregation": WINDOW_SUM,
            "period_days": window_days, "window_start": _iso(start), "window_end": _iso(end), "as_of": _iso(cutoff),
            "quantity_unit": next(iter(units)) if len(units) == 1 else None, "zero_semantics": zero_semantics,
            "semantics_evidence": semantics_evidence, "provenance": provenance, "source_lineage": source_lineage,
            "future_records_excluded": future, "out_of_period_records": out_of_period}
    empty = {"quantity": None, "gross_positive": None, "negative_total": None, "observed_days": 0, "zero_days": 0,
             "negative_days": 0, "system_missing_days": 0, "no_record_days": window_days, "not_carried_days": 0,
             "not_sellable_days": 0, "duplicate_dates": 0, "is_complete_observation": False, "daily_rate": None,
             "negative_semantics": negative_semantics or NO_NEGATIVES}
    if cutoff is not None and end > cutoff:
        return {**base, **empty, "gate_status": INVALID_VALUE, "reason_codes": _reasons(*reasons, "WINDOW_AFTER_AS_OF")}
    if len(datasets) > 1:
        return {**base, **empty, "gate_status": NOT_COMPARABLE, "reason_codes": _reasons(*reasons, "DATASET_MISMATCH")}
    if len(types) > 1:
        return {**base, **empty, "gate_status": NOT_COMPARABLE, "reason_codes": _reasons(*reasons, "SIGNAL_TYPE_MISMATCH")}
    if len(units) > 1:
        return {**base, **empty, "gate_status": UNIT_MISMATCH, "reason_codes": _reasons(*reasons, "UNIT_MISMATCH")}
    counts = Counter(_day(r.get("date")) for r in in_window)
    duplicates = sum(1 for n in counts.values() if n > 1)
    if duplicates:
        return {**base, **empty, "duplicate_dates": duplicates, "no_record_days": window_days - len(counts),
                "gate_status": INVALID_VALUE, "reason_codes": _reasons(*reasons, "DUPLICATE_DATE")}
    states: Counter[str] = Counter()
    values: list[float] = []
    for row in in_window:
        declared = _text(row.get("day_state")).upper()
        if declared in DECLARED_DAY_STATES:
            states[declared] += 1
            continue
        quantity = _finite(row.get("quantity"))
        if quantity is None:
            states[SYSTEM_MISSING] += 1
            continue
        states[OBSERVED_ZERO if quantity == 0 else OBSERVED_VALUE] += 1
        values.append(quantity)
    no_record = window_days - len(counts)
    negatives = [v for v in values if v < 0]
    if negatives:
        neg_sem = negative_semantics if negative_semantics not in (None, NO_NEGATIVES) else NEGATIVE_UNKNOWN
    else:
        neg_sem = NO_NEGATIVES
    observed = len(values)
    applicable_days = window_days - states[NOT_CARRIED] - states[NOT_SELLABLE]
    result = {**base, "observed_days": observed, "zero_days": states[OBSERVED_ZERO], "negative_days": len(negatives),
              "system_missing_days": states[SYSTEM_MISSING], "no_record_days": no_record,
              "not_carried_days": states[NOT_CARRIED], "not_sellable_days": states[NOT_SELLABLE],
              "duplicate_dates": 0, "negative_semantics": neg_sem,
              "gross_positive": _round(sum(v for v in values if v > 0)) if values else None,
              "negative_total": _round(sum(negatives)) if values else None}
    gate = VALID
    if states[NOT_CARRIED] == window_days:
        reasons.append("NOT_CARRIED_NOT_ZERO")
        return {**result, "quantity": None, "is_complete_observation": False, "daily_rate": None,
                "gate_status": MISSING_DATA, "reason_codes": _reasons(*reasons), "applicable": False}
    if states[NOT_CARRIED]:
        reasons.append("NOT_CARRIED_NOT_ZERO")
    if states[NOT_SELLABLE]:
        reasons.append("NOT_SELLABLE_EXCLUDED")
    if states[SYSTEM_MISSING]:
        reasons.append("SYSTEM_MISSING_DAYS")
    if no_record:
        reasons.append("NO_RECORD_DAYS")
    if not observed:
        reasons.append("NO_OBSERVATION")
        return {**result, "quantity": None, "is_complete_observation": False, "daily_rate": None,
                "gate_status": MISSING_DATA, "reason_codes": _reasons(*reasons), "applicable": True}
    complete = observed == applicable_days and applicable_days > 0
    if not complete:
        reasons.append("PARTIAL_OBSERVATION")
        gate = combine_gate(gate, VALID_WITH_LIMITATION)
    if negatives:
        if neg_sem in NETTABLE_NEGATIVES:
            reasons.append("RETURNS_NETTED_DOCUMENTED")
            gate = combine_gate(gate, VALID_WITH_LIMITATION)
        else:
            reasons.append("NEGATIVE_UNRESOLVED")
            gate = combine_gate(gate, INVALID_VALUE)
    if states[OBSERVED_ZERO] and zero_semantics == ZERO_RECORDED_POSSIBLY_CENSORED:
        reasons.append("ZERO_POSSIBLY_CENSORED")
    sig = resolved_type or UNKNOWN_SIGNAL
    if sig == UNKNOWN_SIGNAL:
        reasons.append("SIGNAL_TYPE_UNKNOWN")
        gate = combine_gate(gate, NOT_COMPARABLE)
    elif sig != OBSERVED_RETAIL_SALES:
        reasons.append("PROXY_SIGNAL")
        gate = combine_gate(gate, PROXY_ONLY)
    if result["quantity_unit"] is None:
        reasons.append("UNIT_UNKNOWN")
        gate = combine_gate(gate, VALID_WITH_LIMITATION)
    if semantics_evidence == COLUMN_HEADER_CLAIM:
        reasons.append("SEMANTICS_FROM_COLUMN_NAME_ONLY")
        gate = combine_gate(gate, VALID_WITH_LIMITATION)
    net = sum(values)
    rate = net / observed
    if not complete:
        reasons.append("RATE_OVER_OBSERVED_DAYS")
    if window_days < LEGACY_RATE_WINDOW_DAYS:
        reasons.append("RATE_WINDOW_SHORTER_THAN_LEGACY_30D")
    return {**result, "quantity": _round(net), "is_complete_observation": complete, "daily_rate": _round(rate),
            "gate_status": gate, "reason_codes": _reasons(*reasons), "applicable": True}


def window_signal(window: Mapping[str, Any]) -> SalesSignal:
    """The SalesSignal of an aggregate_daily_window result."""
    return SalesSignal(dataset_id=window.get("dataset_id"), location_id=window.get("location_id"),
                       product_id=window.get("product_id"), quantity=window.get("quantity"),
                       signal_type=window.get("signal_type") or UNKNOWN_SIGNAL, aggregation=WINDOW_SUM,
                       period_days=window.get("period_days"), observation_start=window.get("window_start"),
                       observation_end=window.get("window_end"), as_of_date=window.get("as_of"),
                       quantity_unit=window.get("quantity_unit"), observed_days=window.get("observed_days"),
                       is_complete_observation=window.get("is_complete_observation"),
                       zero_semantics=window.get("zero_semantics") or ZERO_SEMANTICS_UNDECLARED,
                       negative_semantics=window.get("negative_semantics") or NO_NEGATIVES,
                       semantics_evidence=window.get("semantics_evidence") or NO_EVIDENCE,
                       provenance=window.get("provenance") or "", source_lineage=window.get("source_lineage") or "",
                       quality_flags=tuple(window.get("reason_codes") or ()))


# ------------------------------------------------------------------------------------------------ average inventory (§15)


def average_inventory(snapshots: Iterable[Mapping[str, Any]], *, period_start: Any, period_end: Any, as_of: Any = None,
                      dataset_id: str | None = None, quantity_unit: str | None = None) -> dict[str, Any]:
    """Average units held over [period_start, period_end] from the snapshots that exist; none is generated.

    Snapshots are mappings with ``date`` and ``quantity`` (optionally ``quantity_unit``, ``dataset_id``). The value is
    the mean of the observed snapshots when the opening and closing snapshots exist; one snapshot is not an average.
    """
    start, end, cutoff = _day(period_start), _day(period_end), _day(as_of)
    if start is None or end is None or end < start:
        raise ValueError("a valid period is required")
    period_days = (end - start).days + 1
    reasons: list[str] = []
    rows = [dict(r) for r in snapshots]
    if cutoff is not None:
        future = [r for r in rows if _day(r.get("date")) is not None and _day(r.get("date")) > cutoff]
        if future:
            reasons.append("FUTURE_DATA_EXCLUDED")
            rows = [r for r in rows if r not in future]
    rows = [r for r in rows if _day(r.get("date")) is not None and start <= _day(r.get("date")) <= end]
    base = {"dataset_id": dataset_id, "period_start": _iso(start), "period_end": _iso(end), "period_days": period_days,
            "quantity_unit": quantity_unit, "value": None, "method": None, "snapshot_count": 0, "has_start": False,
            "has_end": False, "intermediate_count": 0, "start_qty": None, "end_qty": None, "endpoint_average": None,
            "null_snapshots": 0}
    datasets = {r.get("dataset_id", dataset_id) for r in rows}
    units = {r.get("quantity_unit", quantity_unit) for r in rows}
    if len(datasets) > 1:
        return {**base, "status": NOT_COMPARABLE, "gate_status": NOT_COMPARABLE, "reason_codes": _reasons(*reasons, "DATASET_MISMATCH")}
    if len(units) > 1:
        return {**base, "status": UNIT_MISMATCH, "gate_status": UNIT_MISMATCH, "reason_codes": _reasons(*reasons, "UNIT_MISMATCH")}
    if units:
        base["quantity_unit"] = next(iter(units))
    if datasets:
        base["dataset_id"] = next(iter(datasets))
    counts = Counter(_day(r.get("date")) for r in rows)
    if any(n > 1 for n in counts.values()):
        return {**base, "status": INVALID_VALUE, "gate_status": INVALID_VALUE, "reason_codes": _reasons(*reasons, "DUPLICATE_DATE")}
    observed = {}
    nulls = 0
    for row in rows:
        quantity = _finite(row.get("quantity"))
        if quantity is None:
            nulls += 1
        else:
            observed[_day(row.get("date"))] = quantity
    if nulls:
        reasons.append("SNAPSHOT_NULL")
    base.update({"snapshot_count": len(observed), "null_snapshots": nulls, "has_start": start in observed,
                 "has_end": end in observed, "intermediate_count": sum(1 for d in observed if start < d < end),
                 "start_qty": observed.get(start), "end_qty": observed.get(end)})
    if any(v < 0 for v in observed.values()):
        return {**base, "status": INVALID_VALUE, "gate_status": INVALID_VALUE, "reason_codes": _reasons(*reasons, "NEGATIVE_INVENTORY")}
    if not observed:
        return {**base, "status": AVG_INVENTORY_UNAVAILABLE, "gate_status": MISSING_DATA,
                "reason_codes": _reasons(*reasons, "NO_SNAPSHOT")}
    if len(observed) == 1:
        code = "START_SNAPSHOT_ONLY" if base["has_start"] else "END_SNAPSHOT_ONLY" if base["has_end"] else "SINGLE_MID_SNAPSHOT"
        return {**base, "status": AVG_INVENTORY_UNAVAILABLE, "gate_status": MISSING_DATA, "reason_codes": _reasons(*reasons, code)}
    gate = VALID
    if not (base["has_start"] and base["has_end"]):
        reasons.append("PERIOD_ENDPOINT_MISSING")
        gate = VALID_WITH_LIMITATION
    if len(observed) < period_days:
        reasons.append("SPARSE_SNAPSHOTS")
        gate = VALID_WITH_LIMITATION
    if base["quantity_unit"] is None:
        reasons.append("UNIT_UNKNOWN")
        gate = combine_gate(gate, VALID_WITH_LIMITATION)
    mean = sum(observed.values()) / len(observed)
    endpoint = (observed[start] + observed[end]) / 2 if base["has_start"] and base["has_end"] else None
    return {**base, "value": _round(mean), "method": "SNAPSHOT_MEAN", "endpoint_average": _round(endpoint),
            "status": "AVG_INVENTORY_AVAILABLE", "gate_status": gate, "reason_codes": _reasons(*reasons)}


def declared_average_inventory(value: Any, *, dataset_id: str | None = None, quantity_unit: str | None = None) -> dict[str, Any]:
    """An uploaded average inventory (e.g. the workbook avg_inventory column) whose averaging window is undeclared."""
    number = _finite(value)
    base = {"dataset_id": dataset_id, "period_start": None, "period_end": None, "period_days": None,
            "quantity_unit": quantity_unit, "value": number, "method": "DECLARED_AVERAGE", "snapshot_count": None,
            "has_start": None, "has_end": None, "intermediate_count": None, "start_qty": None, "end_qty": None,
            "endpoint_average": None, "null_snapshots": 0}
    if number is None:
        return {**base, "status": AVG_INVENTORY_UNAVAILABLE, "gate_status": MISSING_DATA, "reason_codes": _reasons("NO_SNAPSHOT")}
    if number < 0:
        return {**base, "status": INVALID_VALUE, "gate_status": INVALID_VALUE, "reason_codes": _reasons("NEGATIVE_INVENTORY")}
    reasons = ["DECLARED_AVERAGE_WINDOW_UNDECLARED"] + (["UNIT_UNKNOWN"] if quantity_unit is None else [])
    return {**base, "status": "AVG_INVENTORY_AVAILABLE", "gate_status": VALID_WITH_LIMITATION, "reason_codes": _reasons(*reasons)}


# ------------------------------------------------------------------------------------------------ classification (§17-18)


def _legacy_turnover_module():
    try:
        return load_legacy_module("turnover_analyzer")
    except LegacyAlgorithmUnavailable:  # pragma: no cover - the module ships with the repository
        return None


def legacy_grade(days: float) -> str | None:
    """The legacy grade of a days-of-supply value, by the legacy function itself (thresholds unchanged)."""
    module = _legacy_turnover_module()
    if module is None:  # pragma: no cover
        return None
    return module._assign_grade(float(days))


def days_of_supply(stock: Any, rate: Any) -> dict[str, Any]:
    """Current units / units per day, with the cases the legacy formula folds into 0 or 999 kept apart."""
    s, r = _finite(stock), _finite(rate)
    if s is None:
        return {"value": None, "status": DOS_STOCK_UNAVAILABLE, "reasons": _reasons("STOCK_MISSING")}
    if s < 0:
        return {"value": None, "status": INVALID_VALUE, "reasons": _reasons("NEGATIVE_INVENTORY")}
    if r is None:
        return {"value": None, "status": DOS_RATE_UNAVAILABLE, "reasons": _reasons("SALES_MISSING")}
    if r < 0:
        return {"value": None, "status": DOS_NEGATIVE_RATE, "reasons": _reasons("NET_SALES_NEGATIVE")}
    if s == 0 and r == 0:
        return {"value": None, "status": DOS_NO_STOCK_NO_MOVEMENT, "reasons": _reasons("NO_STOCK_NO_MOVEMENT")}
    if s == 0:
        return {"value": 0.0, "status": DOS_ZERO_STOCK, "reasons": _reasons("ZERO_STOCK_NOT_CLASSIFIED")}
    if r == 0:
        return {"value": math.inf, "status": DOS_NO_MOVEMENT, "reasons": _reasons("ZERO_SALES_OBSERVED")}
    return {"value": _round(s / r, 4), "status": DOS_AVAILABLE, "reasons": []}


def classify(dos: Mapping[str, Any], *, basis: str, evidence: str, window_days: int | None,
             complete: bool | None) -> dict[str, Any]:
    """Legacy grade of a validated days of supply, with its scope and confidence; UNCLASSIFIED when not classifiable."""
    if dos.get("status") not in CLASSIFIABLE_DOS or basis == NO_BASIS:
        return {"classification": UNCLASSIFIED, "confidence": "NONE", "scope": SCOPE_NONE, "reasons": []}
    grade = legacy_grade(dos["value"])
    reasons = ["LEGACY_THRESHOLD_HEURISTIC"]
    if basis != REAL_SALES_TURNOVER:
        return {"classification": grade, "confidence": "LOW", "scope": SCOPE_PROXY, "reasons": _reasons(*reasons, "PROXY_SIGNAL")}
    scope = SCOPE_CLAIMED if evidence == COLUMN_HEADER_CLAIM else SCOPE_REAL
    if complete is False:
        return {"classification": grade, "confidence": "LOW", "scope": scope, "reasons": _reasons(*reasons, "PARTIAL_OBSERVATION")}
    if window_days is not None and window_days < LEGACY_RATE_WINDOW_DAYS:
        return {"classification": grade, "confidence": "LOW", "scope": scope,
                "reasons": _reasons(*reasons, "RATE_WINDOW_SHORTER_THAN_LEGACY_30D")}
    if evidence in (SOURCE_DOCUMENTED, DECLARED_SEMANTICS) and complete is True and window_days is not None:
        return {"classification": grade, "confidence": "HIGH", "scope": scope, "reasons": _reasons(*reasons)}
    extra = []
    if complete is None:
        extra.append("WINDOW_COMPLETENESS_UNDECLARED")
    if window_days is None:
        extra.append("AVERAGE_WINDOW_UNDECLARED")
    return {"classification": grade, "confidence": "MEDIUM", "scope": scope, "reasons": _reasons(*reasons, *extra)}


# ------------------------------------------------------------------------------------------------ turnover (§14, 16, 26)


def _unit_check(sales_unit: str | None, inventory_unit: str | None, same_dataset: bool) -> tuple[str | None, list[str]]:
    if sales_unit is not None and inventory_unit is not None:
        return (None, []) if sales_unit == inventory_unit else (UNIT_MISMATCH, ["UNIT_MISMATCH"])
    if sales_unit is None and inventory_unit is None:
        return (VALID_WITH_LIMITATION, ["UNIT_UNKNOWN_SAME_DATASET"]) if same_dataset else (UNIT_MISMATCH, ["UNIT_UNVERIFIABLE"])
    return UNIT_MISMATCH, ["UNIT_UNVERIFIABLE"]


def physical_turnover(sales: Mapping[str, Any], inventory: Mapping[str, Any], *, end_stock: Any = None,
                      per_day_only: bool = False) -> dict[str, Any]:
    """Physical unit turnover = units moved out in the period / average units held over the same period.

    ``sales`` is an :func:`aggregate_daily_window` result (or a mapping with the same keys); ``inventory`` an
    :func:`average_inventory` / :func:`declared_average_inventory` result. The numerator's signal type decides the
    basis: only observed retail sales is real-sales turnover; outbound / forecast / estimates are proxies.
    ``end_stock`` (the closing units) gives the days of supply the legacy grade is defined on. ``per_day_only`` marks a
    sales value that is a per-day rate without a period total: days of supply only, no ratio.
    """
    reasons: list[str] = []
    signal = sales.get("signal_type") or UNKNOWN_SIGNAL
    basis = SIGNAL_TURNOVER_BASIS.get(signal, NO_BASIS)
    if signal == PROXY_DEMAND:
        basis = FLOW_BASED_TURNOVER_PROXY if "outbound" in _text(sales.get("provenance")).lower() else FORECAST_DEMAND_PROXY
    period_days = sales.get("period_days")
    provenance = {"sales_signal_type": signal, "sales_window": [sales.get("window_start"), sales.get("window_end")],
                  "sales_observed_days": sales.get("observed_days"), "sales_provenance": sales.get("provenance"),
                  "sales_evidence": sales.get("semantics_evidence"), "inventory_method": inventory.get("method"),
                  "inventory_snapshots": inventory.get("snapshot_count"),
                  "inventory_period": [inventory.get("period_start"), inventory.get("period_end")]}
    stock = end_stock if end_stock is not None else inventory.get("end_qty")
    rate = sales.get("daily_rate")
    dos = days_of_supply(stock, rate) if sales.get("quantity") is not None else days_of_supply(stock, None)
    result = {"turnover_value": None, "turnover_period_days": period_days, "turnover_annualized_365": None,
              "turnover_basis": basis, "turnover_provenance": provenance, "average_inventory": inventory.get("value"),
              "sales_quantity": sales.get("quantity"), "daily_rate": rate, "days_of_supply": dos["value"],
              "days_of_supply_status": dos["status"]}

    def done(status: str, gate: str, *codes: str, value: float | None = None, classifiable: bool = True) -> dict[str, Any]:
        out = dict(result)
        if value is not None:
            out["turnover_value"] = _round(value)
            out["turnover_annualized_365"] = _round(value * 365.0 / period_days) if period_days else None
        cls = (classify(dos, basis=basis, evidence=sales.get("semantics_evidence") or NO_EVIDENCE,
                        window_days=period_days, complete=sales.get("is_complete_observation"))
               if classifiable else {"classification": UNCLASSIFIED, "confidence": "NONE", "scope": SCOPE_NONE, "reasons": []})
        all_reasons = _reasons(*reasons, *codes, *dos["reasons"], *cls["reasons"])
        out.update({"turnover_status": status, "gate_status": gate, "turnover_classification": cls["classification"],
                    "classification_confidence": cls["confidence"], "classification_scope": cls["scope"],
                    "classification_threshold_source": LEGACY_GRADE_SOURCE if cls["classification"] != UNCLASSIFIED else None,
                    "reason_codes": all_reasons})
        return out

    sales_gate = sales.get("gate_status") or VALID
    if sales.get("applicable") is False:
        return done(NOT_APPLICABLE, MISSING_DATA, *sales.get("reason_codes", []), classifiable=False)
    ds_sales, ds_inv = sales.get("dataset_id"), inventory.get("dataset_id")
    if ds_sales and ds_inv and ds_sales != ds_inv:
        return done(NOT_COMPARABLE, NOT_COMPARABLE, "DATASET_MISMATCH", classifiable=False)
    if signal == OBSERVED_ORDER:
        return done(NOT_COMPARABLE, NOT_COMPARABLE, "ORDER_IS_NOT_OUTFLOW", classifiable=False)
    if sales_gate in (NOT_COMPARABLE, UNIT_MISMATCH, INVALID_VALUE):
        status = {NOT_COMPARABLE: NOT_COMPARABLE, UNIT_MISMATCH: UNIT_MISMATCH, INVALID_VALUE: INVALID_VALUE}[sales_gate]
        if "NEGATIVE_UNRESOLVED" in (sales.get("reason_codes") or []):
            reasons.append("NEGATIVE_UNRESOLVED")
        return done(status, sales_gate, *(sales.get("reason_codes") or []), classifiable=False)
    if basis == NO_BASIS:
        return done(NOT_COMPARABLE, NOT_COMPARABLE, "SIGNAL_TYPE_UNKNOWN", classifiable=False)
    if sales.get("quantity") is None:
        return done(TURNOVER_UNAVAILABLE, MISSING_DATA, "SALES_MISSING", *(sales.get("reason_codes") or []), classifiable=False)
    unit_gate, unit_reasons = _unit_check(sales.get("quantity_unit"), inventory.get("quantity_unit"),
                                          same_dataset=(ds_sales or None) == (ds_inv or None))
    reasons += unit_reasons
    if unit_gate == UNIT_MISMATCH:
        return done(UNIT_MISMATCH, UNIT_MISMATCH, classifiable=False)
    gate = combine_gate(sales_gate, unit_gate)
    if basis != REAL_SALES_TURNOVER:
        gate = combine_gate(gate, PROXY_ONLY)
        reasons.append("PROXY_SIGNAL")
    if per_day_only:
        net = float(sales["quantity"])
        if net < 0:
            return done(NET_SALES_NEGATIVE, combine_gate(gate, INVALID_VALUE), "NET_SALES_NEGATIVE", classifiable=False)
        # No ratio is computed, so the average inventory does not enter the gate; days of supply uses end_stock.
        return done(DAYS_OF_SUPPLY_ONLY, gate, "RATIO_NEEDS_PERIOD_TOTAL", *(sales.get("reason_codes") or []))
    gate = combine_gate(gate, inventory.get("gate_status"))
    inv_period = (inventory.get("period_start"), inventory.get("period_end"))
    if inventory.get("method") == "SNAPSHOT_MEAN" and inv_period != (sales.get("window_start"), sales.get("window_end")):
        return done(PERIOD_MISMATCH, PERIOD_MISMATCH, "PERIOD_NOT_ALIGNED", classifiable=False)
    if inventory.get("status") in (INVALID_VALUE, UNIT_MISMATCH, NOT_COMPARABLE):
        return done(inventory["status"], inventory["gate_status"], *(inventory.get("reason_codes") or []), classifiable=False)
    net = float(sales["quantity"])
    if net < 0:
        return done(NET_SALES_NEGATIVE, combine_gate(gate, INVALID_VALUE), "NET_SALES_NEGATIVE", classifiable=False)
    avg = _finite(inventory.get("value"))
    partial = sales.get("is_complete_observation") is False
    if avg is None:
        return done(AVG_INVENTORY_UNAVAILABLE, combine_gate(gate, MISSING_DATA), *(inventory.get("reason_codes") or []))
    if avg == 0:
        return done(DIVISION_BY_ZERO, combine_gate(gate, INVALID_VALUE), "AVERAGE_INVENTORY_ZERO")
    if inventory.get("method") == "DECLARED_AVERAGE":
        reasons += list(inventory.get("reason_codes") or [])
    if partial:
        return done(PARTIAL_PERIOD, combine_gate(gate, VALID_WITH_LIMITATION), *(sales.get("reason_codes") or []),
                    value=net / avg)
    if net == 0:
        return done(ZERO_TURNOVER, gate, "ZERO_SALES_OBSERVED", *(sales.get("reason_codes") or []), value=0.0)
    return done(TURNOVER_AVAILABLE, gate, *(sales.get("reason_codes") or []), value=net / avg)


def financial_turnover(*, cogs: Any = None, cogs_currency: str | None = None, average_inventory_value: Any = None,
                       value_currency: str | None = None, period_days: int | None = None) -> dict[str, Any]:
    """Financial inventory turnover = cost of goods sold / average inventory value of the same period.

    Computed only from an explicit COGS amount and an explicit inventory valuation in the same currency; a sales
    quantity times a price or a unit cost is not COGS and is never used in its place.
    """
    c, v = _finite(cogs), _finite(average_inventory_value)
    reasons = []
    if c is None:
        reasons.append("COGS_MISSING")
    if v is None:
        reasons.append("INVENTORY_VALUE_MISSING")
    if reasons:
        return {"value": None, "status": FINANCIAL_TURNOVER_UNAVAILABLE, "gate_status": MISSING_DATA,
                "reason_codes": _reasons(*reasons), "period_days": period_days}
    if not cogs_currency or cogs_currency != value_currency:
        return {"value": None, "status": FINANCIAL_TURNOVER_UNAVAILABLE, "gate_status": UNIT_MISMATCH,
                "reason_codes": _reasons("CURRENCY_MISMATCH"), "period_days": period_days}
    if v == 0:
        return {"value": None, "status": DIVISION_BY_ZERO, "gate_status": INVALID_VALUE,
                "reason_codes": _reasons("AVERAGE_INVENTORY_ZERO"), "period_days": period_days}
    return {"value": _round(c / v), "status": FINANCIAL_TURNOVER_AVAILABLE, "gate_status": VALID, "reason_codes": [],
            "period_days": period_days, "currency": cogs_currency}


# ------------------------------------------------------------------------------------------------ upload field semantics


# Raw upload column -> (aggregation, period_days). Names and Korean aliases from services.column_aliases.
FIELD_PERIODS = {
    "avg_daily_sales": (DAILY_MEAN, None), "일평균판매량": (DAILY_MEAN, None), "일판매량": (DAILY_MEAN, None),
    "sales_30d": (WINDOW_SUM, 30), "sales_30": (WINDOW_SUM, 30), "30일판매량": (WINDOW_SUM, 30),
    "월판매량": (CALENDAR_MONTH_SUM, None), "sales_7d": (WINDOW_SUM, 7), "주판매량": (WINDOW_SUM, 7),
    "7일판매량": (WINDOW_SUM, 7), "demand_forecast_7d": (FORECAST_SUM, 7),
    "sales_qty": (PERIOD_UNDECLARED, None), "sales": (PERIOD_UNDECLARED, None), "판매량": (PERIOD_UNDECLARED, None),
    "demand_qty": (PERIOD_UNDECLARED, None), "수요량": (PERIOD_UNDECLARED, None), "수요": (PERIOD_UNDECLARED, None),
}
SALES_FIELDS = ("sales_qty", "avg_daily_sales", "sales_7d", "sales_30d", "sales_30", "demand_qty", "demand_forecast_7d")
# The inventory alias chain applied at upload, in order (data_loader._normalize_inventory, then
# column_aliases.normalize_columns with INVENTORY_ALIASES). Replayed by resolve_loader_aliases; a test checks the replay
# against normalize_loaded_data for every combination of sales columns.
LOADER_ALIAS_CHAIN = (
    ("data_loader._normalize_inventory", "sales_qty", ("avg_daily_sales", "sales_30d", "sales_30", "sales_7d", "sales"), "exact"),
    ("data_loader._normalize_inventory", "sales_30d", ("sales_30", "sales_qty", "avg_daily_sales"), "exact"),
    *(("column_aliases.INVENTORY_ALIASES", target, tuple(INVENTORY_ALIASES[target]), "canonical")
      for target in ("sales_qty", "avg_daily_sales", "demand_qty", "sales_30d", "sales_7d")),
)
LEGACY_PREPARE_ALIAS = ("legacy data_adapter.prepare_legacy_data", "sales_30d", ("sales_30", "sales_qty", "avg_daily_sales"))


def resolve_loader_aliases(raw_columns: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Replay the upload alias chain on a raw column set: standard field -> the raw column it is a copy of."""
    columns = [str(c).strip() for c in raw_columns]
    origin: dict[str, str] = {c: c for c in columns}
    stage_of: dict[str, str] = {c: "RAW" for c in columns}
    for stage, target, candidates, matcher in LOADER_ALIAS_CHAIN:
        if target in origin:
            continue
        if matcher == "exact":
            source = next((c for c in candidates if c in origin), None)
        else:
            lookup = {_canonical(c): c for c in origin}
            source = next((lookup[_canonical(c)] for c in candidates if _canonical(c) in lookup), None)
        if source is not None:
            origin[target] = origin[source]
            stage_of[target] = stage
    out = {}
    for name in SALES_FIELDS:
        if name not in origin:
            continue
        raw = origin[name]
        declared, raw_period = FIELD_PERIODS.get(name, (PERIOD_UNDECLARED, None)), FIELD_PERIODS.get(raw, (PERIOD_UNDECLARED, None))
        out[name] = {"raw_source_column": raw, "alias_stage": stage_of[name], "field_period": declared,
                     "source_period": raw_period, "period_conflict": _period_conflict(declared, raw_period)}
    return out


def _period_conflict(field: tuple[str, int | None], source: tuple[str, int | None]) -> bool:
    """A field holding a value of another declared period (an undeclared side is ambiguous, not a conflict)."""
    if PERIOD_UNDECLARED in (field[0], source[0]):
        return False
    per_day = {DAILY_MEAN, DAILY_QUANTITY}
    if field[0] in per_day and source[0] in per_day:
        return False
    return field != source


def _equal(a: pd.Series, b: pd.Series) -> bool:
    """Element-wise identical numeric columns with at least one observed value. An all-zero pair counts: a copy of
    a field whose values are all 0 cannot be told apart from an independent zero, so it is not proven independent."""
    x, y = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    same = (x == y) | (x.isna() & y.isna())
    return bool(same.all()) and bool(x.notna().any())


def upload_sales_fields(inventory: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    """Per sales field of a normalized upload: its declared period, declared semantics and alias-copy suspicion.

    The upload keeps no record of which column was aliased, so a copy is detected as an exact equality (with at least
    one non-zero value) between fields of different declared periods. Which side is the original is not guessed: both
    are excluded from period-based use unless a declared semantics column resolves one of them.
    """
    if inventory is None or not isinstance(inventory, pd.DataFrame) or inventory.empty:
        return {}
    present = [c for c in SALES_FIELDS if c in inventory.columns]
    report: dict[str, dict[str, Any]] = {}
    for name in present:
        aggregation, period = FIELD_PERIODS.get(name, (PERIOD_UNDECLARED, None))
        values = pd.to_numeric(inventory[name], errors="coerce")
        report[name] = {"aggregation": aggregation, "period_days": period, "rows": int(len(values)),
                        "null_rows": int(values.isna().sum()), "zero_rows": int((values == 0).sum()),
                        "negative_rows": int((values < 0).sum()), "equal_to": [], "period_conflict": False,
                        "declared_semantics": _text(inventory[f"{name}_semantics"].dropna().iloc[0])
                        if f"{name}_semantics" in inventory.columns and inventory[f"{name}_semantics"].notna().any() else "",
                        "declared_provenance": sorted({_text(v) for v in inventory[f"{name}_provenance"].dropna()} - {""})
                        if f"{name}_provenance" in inventory.columns else []}
    for name in present:
        report[name]["identical_all_zero_with"] = []
    for i, left in enumerate(present):
        for right in present[i + 1:]:
            if _equal(inventory[left], inventory[right]):
                report[left]["equal_to"].append(right)
                report[right]["equal_to"].append(left)
                if not bool((pd.to_numeric(inventory[left], errors="coerce").fillna(0) != 0).any()):
                    report[left]["identical_all_zero_with"].append(right)
                    report[right]["identical_all_zero_with"].append(left)
    # A field that is an exact copy of a field with declared semantics carries those semantics.
    for name, item in report.items():
        item["inherited_semantics"] = ""
        if not item["declared_semantics"] and not item["declared_provenance"]:
            for other in item["equal_to"]:
                source = report[other]
                if source["declared_semantics"] or source["declared_provenance"]:
                    item["inherited_semantics"] = " ".join([source["declared_semantics"], *source["declared_provenance"]]).strip()
                    item["inherited_from"] = other
                    break
    declared_daily = _declared_daily_fields(inventory, report)
    for name, item in report.items():
        mine = (item["aggregation"], item["period_days"])
        if name in declared_daily:
            mine = (DAILY_QUANTITY, 1)
            item["resolved_aggregation"], item["resolved_period_days"] = DAILY_QUANTITY, 1
        # A field equal to a declared one-day quantity holds that one-day value whatever its name says.
        item["copy_of_declared_daily"] = sorted(o for o in item["equal_to"] if o in declared_daily and name not in declared_daily)
        if name in declared_daily:
            continue  # the declaration fixes the direction: the declared field is the original
        # A copy of a declared original is judged against that original only; otherwise against every equal field.
        others = item["copy_of_declared_daily"] or item["equal_to"]
        for other in others:
            theirs = (DAILY_QUANTITY, 1) if other in declared_daily else (report[other]["aggregation"], report[other]["period_days"])
            if _period_conflict(mine, theirs):
                item["period_conflict"] = True
    return report


# The production convention of legacy_adapters.data_adapter.prepare_legacy_data: an upload sales_qty labelled with
# this semantics is the snapshot day's outbound used as a one-day demand proxy (the Suhyup E2E rows).
DAILY_PROXY_SEMANTICS = "demand_proxy_not_retail_sales"


def _declared_daily_fields(inventory: pd.DataFrame, report: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """Fields declared to be one day's observed flow (the data_adapter one-day proxy convention, or an outbound
    provenance on a dated row)."""
    out = set()
    dated = "snapshot_date" in inventory.columns or "date" in inventory.columns
    for name, item in report.items():
        text = " ".join([item.get("declared_semantics") or "", *item.get("declared_provenance", [])]).lower()
        if item["aggregation"] == PERIOD_UNDECLARED and dated and (DAILY_PROXY_SEMANTICS in text or "outbound" in text):
            out.add(name)
    return out


def field_signal_type(name: str, report: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """(signal type, evidence) of an upload sales field from its declarations; the header is only a claim."""
    semantics = _text((row or {}).get(f"{name}_semantics")) or report.get("declared_semantics") or ""
    provenance = " ".join([_text((row or {}).get(f"{name}_provenance")), *report.get("declared_provenance", []),
                           report.get("inherited_semantics") or ""]).lower()
    text = f"{semantics} {provenance}".lower()
    if "proxy" in text or "outbound" in text or "not_retail" in text:
        return PROXY_DEMAND, DECLARED_SEMANTICS
    if "forecast" in text:
        return FORECAST_DEMAND, DECLARED_SEMANTICS
    if "user" in text or "estimate" in text:
        return USER_ESTIMATED_DEMAND, DECLARED_SEMANTICS
    if text.strip() and any(k in text for k in ("actual", "observed", "pos", "retail")):
        return OBSERVED_RETAIL_SALES, DECLARED_SEMANTICS
    if name == "demand_qty":
        return UNKNOWN_SIGNAL, NO_EVIDENCE   # overloaded: shortage, 7-day forecast or 30-day sales depending on the file
    if name == "demand_forecast_7d":
        return FORECAST_DEMAND, COLUMN_HEADER_CLAIM
    return OBSERVED_RETAIL_SALES, COLUMN_HEADER_CLAIM


# Preference for the validated daily rate: the 30-day window the legacy grade is written on, then the declared daily
# mean, then the 7-day window, then a declared one-day quantity.
RATE_FIELD_ORDER = ("sales_30d", "sales_30", "avg_daily_sales", "sales_7d", "sales_qty")


def row_sales_signal(row: Mapping[str, Any], fields: Mapping[str, Mapping[str, Any]], *, dataset_id: str | None = None,
                     quantity_unit: str | None = None) -> tuple[SalesSignal | None, str | None, list[str]]:
    """The sales signal of one upload inventory row: the first usable field in RATE_FIELD_ORDER."""
    skipped: list[str] = []
    for name in RATE_FIELD_ORDER:
        item = fields.get(name)
        if item is None:
            continue
        if item["period_conflict"] or item.get("copy_of_declared_daily"):
            skipped.append("ALIAS_COPY_ACROSS_PERIODS")
            continue
        aggregation = item.get("resolved_aggregation", item["aggregation"])
        period = item.get("resolved_period_days", item["period_days"])
        if aggregation == PERIOD_UNDECLARED:
            skipped.append("PERIOD_UNDECLARED")
            continue
        signal_type, evidence = field_signal_type(name, item, row)
        snapshot = _text(row.get("snapshot_date") or row.get("date")) or None
        complete = True if aggregation == DAILY_QUANTITY and _finite(row.get(name)) is not None else None
        signal = SalesSignal(dataset_id=dataset_id, location_id=_text(row.get("store_id")) or None,
                             product_id=_text(row.get("product_id")) or None, quantity=_finite(row.get(name)),
                             signal_type=signal_type, aggregation=aggregation, period_days=period,
                             observation_start=snapshot if aggregation == DAILY_QUANTITY else None,
                             observation_end=snapshot, as_of_date=snapshot, quantity_unit=quantity_unit,
                             observed_days=1 if complete else None, is_complete_observation=complete,
                             negative_semantics=NO_NEGATIVES if (_finite(row.get(name)) or 0) >= 0 else NEGATIVE_UNKNOWN,
                             semantics_evidence=evidence,
                             provenance=_text(row.get(f"{name}_provenance")) or ";".join(item.get("declared_provenance", [])),
                             source_lineage=f"upload.inventory.{name}")
        return signal, name, list(dict.fromkeys(skipped))
    return None, None, list(dict.fromkeys(skipped))


def signal_window(signal: SalesSignal) -> dict[str, Any]:
    """An aggregate-shaped mapping of an upload signal (no daily records), for physical_turnover."""
    rate = daily_rate(signal)
    quantity = signal.quantity
    gate = rate["gate"]
    reasons = list(rate["reasons"])
    if quantity is not None and quantity < 0:
        gate, reasons = combine_gate(gate, INVALID_VALUE), reasons + ["NEGATIVE_UNRESOLVED"]
    if signal.aggregation in PER_DAY_AGGREGATIONS:
        period_quantity = None  # a per-day value carries no period total
    else:
        period_quantity = quantity
    return {"dataset_id": signal.dataset_id, "location_id": signal.location_id, "product_id": signal.product_id,
            "signal_type": signal.signal_type, "aggregation": signal.aggregation, "period_days": rate["window_days"],
            "window_start": signal.observation_start, "window_end": signal.observation_end, "as_of": signal.as_of_date,
            "quantity_unit": signal.quantity_unit, "quantity": period_quantity if quantity is not None else None,
            "rate_quantity": quantity, "daily_rate": rate["value"], "observed_days": signal.observed_days,
            "is_complete_observation": signal.is_complete_observation, "negative_semantics": signal.negative_semantics,
            "semantics_evidence": signal.semantics_evidence, "provenance": signal.provenance,
            "gate_status": gate, "reason_codes": _reasons(*reasons), "applicable": True}


# ------------------------------------------------------------------------------------------------ legacy diagnostics


def _first_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    present = set(columns)
    return next((c for c in candidates if c in present), None)


def legacy_turnover_inputs(row: Mapping[str, Any], columns: Iterable[str], upload_fields: Mapping[str, Mapping[str, Any]]
                           ) -> dict[str, Any]:
    """Which input the legacy turnover analyzer read for this row and how it treated it (diagnostics only)."""
    columns = list(columns)
    sales_col = _first_column(columns, ("sales_30d", "state_source_sales_30d", "source_sales_30d"))
    via = sales_col
    if sales_col is None and "avg_daily_sales" in columns:
        via = "avg_daily_sales x 30"
    stock_col = _first_column(columns, ("stock_qty", "state_source_stock", "source_stock_qty"))
    raw_sales = _finite(row.get(sales_col)) if sales_col else (_finite(row.get("avg_daily_sales")) if via else None)
    raw_stock = _finite(row.get(stock_col)) if stock_col else None
    codes: list[str] = []
    if via is None:
        codes.append("LEGACY_SALES_COLUMN_ABSENT_AS_ZERO")
    elif raw_sales is None:
        codes.append("LEGACY_SALES_NULL_AS_ZERO")
    elif raw_sales < 0:
        codes.append("LEGACY_NEGATIVE_SALES_NEGATIVE_DAYS")
    if stock_col is None or raw_stock is None:
        codes.append("LEGACY_STOCK_NULL_AS_ZERO")
    sales_used = raw_sales if raw_sales is not None else 0.0
    stock_used = raw_stock if raw_stock is not None else 0.0
    if stock_used == 0 and sales_used > 0:
        codes.append("LEGACY_ZERO_STOCK_GRADED_FAST")
    elif stock_used == 0 and sales_used == 0:
        codes.append("LEGACY_ZERO_STOCK_ZERO_SALES_DEAD")
    elif sales_used == 0 and raw_sales is not None:
        codes.append("LEGACY_ZERO_SALES_DEAD")
    avg = _finite(row.get("avg_inventory")) if "avg_inventory" in columns else None
    if "avg_inventory" not in columns or avg is None or avg <= 0:
        codes.append("LEGACY_SINGLE_SNAPSHOT_AS_AVERAGE")
    if (avg is None or avg <= 0) and stock_used == 0:
        codes.append("LEGACY_AVG_INVENTORY_ZERO_RATE_ZERO")
    item = upload_fields.get("sales_30d", {})
    if _text(row.get("sales_30d_provenance")) == "derived_30d_projection_from_daily_demand_proxy":
        codes.append("LEGACY_30D_PROJECTED_FROM_ONE_DAY")
    elif item.get("copy_of_declared_daily") or (item.get("period_conflict") and any(
            upload_fields.get(o, {}).get("aggregation") in PER_DAY_AGGREGATIONS for o in item.get("equal_to", []))):
        codes.append("LEGACY_DAILY_VALUE_IN_30D_FIELD")
    daily = upload_fields.get("avg_daily_sales", {})
    if daily.get("period_conflict") and any(upload_fields.get(o, {}).get("aggregation") in (WINDOW_SUM, CALENDAR_MONTH_SUM)
                                            for o in daily.get("equal_to", [])):
        codes.append("LEGACY_30D_FIELD_IN_DAILY_FIELD")
    proxy = any(field_signal_type(n, upload_fields[n], row)[0] in (PROXY_DEMAND, OBSERVED_OUTBOUND_FLOW)
                for n in upload_fields if n in ("sales_qty", "sales_30d", "avg_daily_sales"))
    if proxy:
        codes.append("LEGACY_PROXY_AS_SALES")
    return {"legacy_sales_input": via, "legacy_sales_raw": raw_sales, "legacy_sales_used": sales_used,
            "legacy_stock_input": stock_col, "legacy_stock_raw": raw_stock, "legacy_stock_used": stock_used,
            "legacy_codes": _reasons(*codes)}


GRADE_CAUSES = {
    "SAME": "validated grade equals the legacy grade",
    "MISSING_SALES_GRADED_DEAD": "legacy DEAD from NULL / absent sales read as 0",
    "NEGATIVE_SALES_GRADED_FAST": "legacy FAST from negative sales",
    "ZERO_STOCK_GRADED": "legacy grade on stock 0 (stock-out, not a speed)",
    "MISSING_STOCK_GRADED": "legacy grade on NULL stock read as 0",
    "PERIOD_ALIAS_DISTORTION": "legacy sales_30d holds a daily value / projection; rate distorted",
    "PROXY_SIGNAL_ONLY": "the only sales signal is a proxy (outbound / forecast)",
    "RATE_UNAVAILABLE": "no usable sales signal",
    "DIFFERENT_RATE_FIELD": "validated rate taken from another field than legacy sales_30d",
}


def grade_cause(legacy: str | None, validated: Mapping[str, Any], legacy_codes: Sequence[str]) -> str:
    """Why a legacy grade differs from (or is not confirmed by) the validated classification."""
    codes = set(legacy_codes)
    if "LEGACY_SALES_NULL_AS_ZERO" in codes or "LEGACY_SALES_COLUMN_ABSENT_AS_ZERO" in codes:
        return "MISSING_SALES_GRADED_DEAD" if legacy == "DEAD" else "RATE_UNAVAILABLE"
    if "LEGACY_NEGATIVE_SALES_NEGATIVE_DAYS" in codes:
        return "NEGATIVE_SALES_GRADED_FAST"
    if "LEGACY_STOCK_NULL_AS_ZERO" in codes:
        return "MISSING_STOCK_GRADED"
    if "LEGACY_ZERO_STOCK_GRADED_FAST" in codes or "LEGACY_ZERO_STOCK_ZERO_SALES_DEAD" in codes:
        return "ZERO_STOCK_GRADED"
    if codes & {"LEGACY_DAILY_VALUE_IN_30D_FIELD", "LEGACY_30D_PROJECTED_FROM_ONE_DAY", "LEGACY_30D_FIELD_IN_DAILY_FIELD"}:
        return "PERIOD_ALIAS_DISTORTION"
    if validated.get("classification_scope") == SCOPE_PROXY:
        return "PROXY_SIGNAL_ONLY"
    if validated.get("turnover_classification") == UNCLASSIFIED:
        return "RATE_UNAVAILABLE"
    if validated.get("turnover_classification") == legacy:
        return "SAME"
    return "DIFFERENT_RATE_FIELD"


# ------------------------------------------------------------------------------------------------ ABC (§19, 23-24)

ABC_REVENUE = "ABC_REVENUE"                    # observed retail sales x selling price
ABC_COST_VALUED = "ABC_COST_VALUED_SALES"      # sales x unit cost (a COGS-like value; the legacy formula)
ABC_QUANTITY_PROXY = "ABC_QUANTITY_PROXY"      # units only (price and cost absent)
ABC_NOT_VERIFIABLE = "ABC_NOT_VERIFIABLE"
LEGACY_ABC_FORMULA = "abc_analyzer: abc_revenue_value = sales_30d x unit_cost (cost column absent -> 1.0; NULL -> 0)"


def _abc_module():
    try:
        return load_legacy_module("abc_analyzer")
    except LegacyAlgorithmUnavailable:  # pragma: no cover
        return None


def abc_thresholds() -> tuple[float, float] | None:
    module = _abc_module()
    return (module._A_THRESHOLD, module._B_THRESHOLD) if module is not None else None


def abc_grades(values: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """Legacy cumulative-share grading (thresholds from abc_analyzer) of per-product values; zero total -> none."""
    module = _abc_module()
    items = sorted(((k, float(v)) for k, v in values.items()), key=lambda kv: (-kv[1], kv[0]))
    total = sum(v for _, v in items)
    if module is None or total <= 0:
        return {}
    out, cumulative = {}, 0.0
    for key, value in items:
        cumulative += value
        share = cumulative / total
        out[key] = {"value": _round(value), "cumulative_share": _round(share), "grade": module._assign_grade(share)}
    return out


def legacy_abc_basis(columns: Iterable[str]) -> dict[str, Any]:
    """What the legacy ABC actually multiplies, from the columns it would see."""
    columns = list(columns)
    sales_col = _first_column(columns, ("sales_30d", "state_source_sales_30d", "source_sales_30d"))
    if sales_col is None and "avg_daily_sales" in columns:
        sales_col = "avg_daily_sales x 30"
    cost_col = _first_column(columns, ("unit_cost", "state_unit_cost"))
    basis = ABC_COST_VALUED if cost_col else ABC_QUANTITY_PROXY
    codes = []
    if cost_col is None:
        codes.append("LEGACY_ABC_COST_ABSENT_TIMES_ONE")
    if sales_col is None:
        codes.append("LEGACY_SALES_COLUMN_ABSENT_AS_ZERO")
    return {"legacy_formula": LEGACY_ABC_FORMULA, "legacy_sales_input": sales_col, "legacy_cost_input": cost_col or "1.0",
            "legacy_basis": basis, "legacy_label": "abc_revenue_value (named revenue; the value is cost-valued sales)"
            if cost_col else "abc_revenue_value (named revenue; the value is a unit count)", "legacy_codes": codes}


def validated_abc(rows: Sequence[Mapping[str, Any]], *, product_key: str = "product_id") -> dict[str, Any]:
    """ABC on a stated basis: revenue when every row has observed sales and a price, cost-valued when every row has a
    unit cost, otherwise a quantity proxy; rows with a missing value are excluded and counted, never valued at 1 or 0."""
    def complete(field: str) -> bool:
        return bool(rows) and all(_finite(r.get(field)) is not None and _finite(r.get(field)) > 0 for r in rows)

    sales_ok = [r for r in rows if _finite(r.get("period_sales")) is not None]
    reasons = []
    if len(sales_ok) < len(rows):
        reasons.append("SALES_MISSING")
    if any(_finite(r.get("period_sales")) is not None and _finite(r.get("period_sales")) < 0 for r in rows):
        reasons.append("NEGATIVE_UNRESOLVED")
    signal_types = {r.get("signal_type") for r in rows}
    proxy = any(t != OBSERVED_RETAIL_SALES for t in signal_types)
    units = {r.get("quantity_unit") for r in rows}
    if complete("unit_price"):
        basis, value_field = ABC_REVENUE, "unit_price"
    elif complete("unit_cost"):
        basis, value_field = ABC_COST_VALUED, "unit_cost"
    else:
        basis, value_field = ABC_QUANTITY_PROXY, None
    gate = VALID
    if basis == ABC_QUANTITY_PROXY:
        gate = PROXY_ONLY
        if len(units) > 1:
            return {"basis": ABC_NOT_VERIFIABLE, "gate_status": UNIT_MISMATCH, "grades": {},
                    "reason_codes": _reasons(*reasons, "UNIT_MISMATCH"), "excluded_rows": len(rows)}
    if proxy:
        gate = combine_gate(gate, PROXY_ONLY)
        reasons.append("PROXY_SIGNAL")
    if "NEGATIVE_UNRESOLVED" in reasons:
        return {"basis": ABC_NOT_VERIFIABLE, "gate_status": INVALID_VALUE, "grades": {},
                "reason_codes": _reasons(*reasons), "excluded_rows": len(rows)}
    values: dict[str, float] = {}
    for row in sales_ok:
        multiplier = _finite(row.get(value_field)) if value_field else 1.0
        values[str(row.get(product_key))] = values.get(str(row.get(product_key)), 0.0) + float(row["period_sales"]) * multiplier
    if reasons and "SALES_MISSING" in reasons:
        gate = combine_gate(gate, VALID_WITH_LIMITATION)
    grades = abc_grades(values)
    return {"basis": basis if grades else ABC_NOT_VERIFIABLE, "gate_status": gate if grades else MISSING_DATA,
            "grades": grades, "reason_codes": _reasons(*reasons), "excluded_rows": len(rows) - len(sales_ok),
            "thresholds": abc_thresholds(), "value_field": value_field}


# ------------------------------------------------------------------------------------------------ disposal risk (§20-21)

DISPOSAL_MEANING = "HEURISTIC_RISK_SCORE"
DISPOSAL_MEANING_NOTE = ("disposal_risk_analyzer: weighted points for remaining shelf life (40), days of supply (30), "
                         "days since inbound (20), category keyword (10), rescaled to 100 without an expiry column; "
                         "not a probability and not a forecast of disposed units")


def disposal_input_audit(row: Mapping[str, Any], columns: Iterable[str], upload_fields: Mapping[str, Mapping[str, Any]],
                         raw_columns: Iterable[str] | None = None) -> dict[str, Any]:
    """Inputs of the legacy disposal score for one row: observed, placeholder (missing read as 0 / 999) or proxy."""
    columns = list(columns)
    codes, components = [], {}
    expiry_col = _first_column(columns, ("expiry_days", "days_to_expiry"))
    if expiry_col is None:
        components["expiry"] = "ABSENT_REWEIGHTED"
        codes.append("DISPOSAL_EXPIRY_ABSENT_REWEIGHTED")
    elif _finite(row.get(expiry_col)) is None:
        components["expiry"] = "NULL_AS_999_SAFE"
        codes.append("DISPOSAL_EXPIRY_NULL_AS_SAFE")
    else:
        components["expiry"] = "OBSERVED"
    raw = set(raw_columns or [])
    if raw and "shelf_life_days" in raw and not ({"expiry_days", "days_to_expiry"} & raw):
        codes.append("DISPOSAL_SHELF_LIFE_ALIASED_AS_REMAINING")
    inbound_col = _first_column(columns, ("inbound_days", "state_inbound_days"))
    if inbound_col is None or _finite(row.get(inbound_col)) is None:
        components["inbound_age"] = "MISSING_AS_ZERO_FRESH"
        codes.append("DISPOSAL_INBOUND_DAYS_MISSING_AS_ZERO")
    else:
        components["inbound_age"] = "OBSERVED"
    sales_col = _first_column(columns, ("sales_30d", "state_source_sales_30d"))
    sales = _finite(row.get(sales_col)) if sales_col else None
    if sales is None:
        components["days_of_supply"] = "SALES_MISSING_AS_ZERO_MAX_POINTS"
        codes.append("DISPOSAL_SALES_MISSING_AS_ZERO")
    else:
        signal_type = OBSERVED_RETAIL_SALES
        for name in ("sales_qty", "sales_30d", "avg_daily_sales"):
            if name in upload_fields:
                kind, _ = field_signal_type(name, upload_fields[name], row)
                if kind != OBSERVED_RETAIL_SALES:
                    signal_type = kind
        item = upload_fields.get("sales_30d", {})
        if _text(row.get("sales_30d_provenance")) == "derived_30d_projection_from_daily_demand_proxy" \
                or item.get("copy_of_declared_daily") or item.get("period_conflict"):
            components["days_of_supply"] = "PERIOD_MISMATCH"
            codes.append("DISPOSAL_SALES_PERIOD_MISMATCH")
        elif signal_type != OBSERVED_RETAIL_SALES:
            components["days_of_supply"] = "PROXY"
        else:
            components["days_of_supply"] = "OBSERVED_CLAIMED"
        if signal_type != OBSERVED_RETAIL_SALES:
            codes.append("DISPOSAL_SALES_PROXY")
    components["category"] = "KEYWORD_HEURISTIC" if "category" in columns else "ABSENT_ZERO"
    stock_col = _first_column(columns, ("stock_qty", "state_source_stock"))
    if stock_col is None or _finite(row.get(stock_col)) is None:
        codes.append("DISPOSAL_STOCK_MISSING_AS_ZERO")
    return {"meaning": DISPOSAL_MEANING, "is_probability": False, "predicts_disposed_units": False,
            "disposal_history_used": False, "components": components, "codes": codes}


DISPOSAL_PROBABILITY_UNAVAILABLE = "DISPOSAL_PROBABILITY_UNAVAILABLE"


def disposal_probability(disposal_history: Sequence[Mapping[str, Any]] | None = None,
                         expiry_dates_observed: bool = False) -> dict[str, Any]:
    """No disposal probability is produced here. Without disposal history and observed expiry it cannot be; with them it
    would need a separately validated model. The legacy score stays a heuristic score."""
    missing = []
    if not disposal_history:
        missing.append("NO_DISPOSAL_HISTORY")
    if not expiry_dates_observed:
        missing.append("NO_OBSERVED_EXPIRY")
    return {"value": None, "status": DISPOSAL_PROBABILITY_UNAVAILABLE,
            "reason_codes": missing or ["NO_VALIDATED_DISPOSAL_MODEL"], "legacy_score_meaning": DISPOSAL_MEANING}


DISPOSAL_CODES = {
    "DISPOSAL_EXPIRY_ABSENT_REWEIGHTED": "no expiry column: the other components are rescaled to 100 points",
    "DISPOSAL_EXPIRY_NULL_AS_SAFE": "NULL remaining shelf life read as 999 days (no expiry points)",
    "DISPOSAL_SHELF_LIFE_ALIASED_AS_REMAINING": "the loader copies total shelf_life_days into the remaining-days field",
    "DISPOSAL_INBOUND_DAYS_MISSING_AS_ZERO": "missing days-since-inbound read as 0 (fresh)",
    "DISPOSAL_SALES_MISSING_AS_ZERO": "missing sales read as 0: the days-of-supply component takes its maximum",
    "DISPOSAL_SALES_PERIOD_MISMATCH": "sales_30d holds a daily value / projection",
    "DISPOSAL_SALES_PROXY": "the sales input is an outbound / forecast proxy",
    "DISPOSAL_STOCK_MISSING_AS_ZERO": "missing stock read as 0",
    "NO_DISPOSAL_HISTORY": "no observed disposal / waste records",
    "NO_OBSERVED_EXPIRY": "no observed expiry dates per lot",
    "NO_VALIDATED_DISPOSAL_MODEL": "no validated disposal model exists in this step",
}
REASONS.update({k: v for k, v in DISPOSAL_CODES.items()})
REASONS["LEGACY_ABC_COST_ABSENT_TIMES_ONE"] = "legacy ABC: no unit cost column, every value multiplied by 1.0 (a unit count)"


# ------------------------------------------------------------------------------------------------ safety stock (§22)

SAFETY_CODES = {
    "SS_DEMAND_MEAN_FROM_DAILY_FIELD": "demand mean = avg_daily_sales",
    "SS_DEMAND_MEAN_FROM_30D_FIELD": "demand mean = sales_30d / 30",
    "SS_DEMAND_MEAN_MISSING_AS_ZERO": "no / NULL demand mean read as 0",
    "SS_DEMAND_MEAN_PROXY": "demand mean is an outbound / proxy quantity",
    "SS_DEMAND_MEAN_PERIOD_MISMATCH": "the field read as a daily mean holds another period",
    "SS_STD_OBSERVED_COLUMN": "demand_std column supplied",
    "SS_STD_PLACEHOLDER_30PCT": "demand_std absent / <= 0: 0.30 x mean placeholder",
    "SS_LEAD_TIME_COLUMN": "lead_time_days supplied",
    "SS_LEAD_TIME_DEFAULT_3": "lead time absent / <= 0: 3-day default",
    "SS_Z_SERVICE_LEVEL_TABLE": "z from the 0.90/0.95/0.99 table (default 1.65)",
}
REASONS.update(SAFETY_CODES)


def safety_stock_input_audit(row: Mapping[str, Any], columns: Iterable[str],
                             upload_fields: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    columns = list(columns)
    codes = ["SS_Z_SERVICE_LEVEL_TABLE"]
    if "avg_daily_sales" in columns:
        source, value = "avg_daily_sales", _finite(row.get("avg_daily_sales"))
        codes.append("SS_DEMAND_MEAN_FROM_DAILY_FIELD")
        item = upload_fields.get("avg_daily_sales", {})
        if item.get("period_conflict") and not item.get("copy_of_declared_daily"):
            codes.append("SS_DEMAND_MEAN_PERIOD_MISMATCH")
    else:
        col = _first_column(columns, ("sales_30d", "state_source_sales_30d", "source_sales_30d"))
        source, value = (f"{col} / 30", _finite(row.get(col))) if col else (None, None)
        if col:
            codes.append("SS_DEMAND_MEAN_FROM_30D_FIELD")
    if value is None:
        codes.append("SS_DEMAND_MEAN_MISSING_AS_ZERO")
    kinds = {field_signal_type(n, upload_fields[n], row)[0] for n in upload_fields if n in ("sales_qty", "avg_daily_sales", "sales_30d")}
    if kinds - {OBSERVED_RETAIL_SALES}:
        codes.append("SS_DEMAND_MEAN_PROXY")
    std = _finite(row.get("demand_std")) if "demand_std" in columns else None
    codes.append("SS_STD_OBSERVED_COLUMN" if std is not None and std > 0 else "SS_STD_PLACEHOLDER_30PCT")
    lead = _finite(row.get("lead_time_days")) if "lead_time_days" in columns else None
    codes.append("SS_LEAD_TIME_COLUMN" if lead is not None and lead > 0 else "SS_LEAD_TIME_DEFAULT_3")
    return {"demand_mean_source": source, "codes": codes,
            "placeholder_inputs": [c for c in codes if c in ("SS_STD_PLACEHOLDER_30PCT", "SS_LEAD_TIME_DEFAULT_3",
                                                             "SS_DEMAND_MEAN_MISSING_AS_ZERO")]}


# ------------------------------------------------------------------------------------------------ store-product matching (§23)

MATCH_CODES = {
    "MATCH_TARGET_SALES_NOT_SUPPLIED_NEUTRAL": "no target sales column: target demand set equal to the source (neutral)",
    "MATCH_TARGET_SALES_NULL_AS_ZERO": "NULL target sales read as 0 (a location without history scored as no demand)",
    "MATCH_SOURCE_ZERO_ZEROES_DEMAND_FIT": "source sales 0 with no target column: demand fit 0 although the target is unobserved",
    "MATCH_SOURCE_SALES_ABSENT_CONSTANT": "no source sales column: constant 1.0",
    "MATCH_RUNS_ON_INVENTORY_ROWS": "the matcher scores source inventory rows, not transfer candidates",
}
REASONS.update(MATCH_CODES)


def store_product_match_audit(row: Mapping[str, Any], columns: Iterable[str]) -> dict[str, Any]:
    columns = list(columns)
    codes = ["MATCH_RUNS_ON_INVENTORY_ROWS"]
    src_col = _first_column(columns, ("avg_daily_sales", "sales_30d", "state_source_sales_30d"))
    tgt_col = _first_column(columns, ("state_target_sales_30d", "target_sales_30d"))
    src = _finite(row.get(src_col)) if src_col else None
    if src_col is None:
        codes.append("MATCH_SOURCE_SALES_ABSENT_CONSTANT")
    if tgt_col is None:
        codes.append("MATCH_TARGET_SALES_NOT_SUPPLIED_NEUTRAL")
        if src_col is not None and (src is None or src == 0):
            codes.append("MATCH_SOURCE_ZERO_ZEROES_DEMAND_FIT")
    elif _finite(row.get(tgt_col)) is None:
        codes.append("MATCH_TARGET_SALES_NULL_AS_ZERO")
    usable = tgt_col is not None and _finite(row.get(tgt_col)) is not None
    return {"target_demand_observed": usable, "use": "RELOCATION_INPUT_CANDIDATE" if usable else "REFERENCE_SCORE_ONLY",
            "codes": codes}


# ------------------------------------------------------------------------------------------------ forecast inputs (§24)

FORECAST_CODES = {
    "FORECAST_V1_AGGREGATE_INPUTS": "V1 reads sales_7d / avg_daily_sales / sales_30d aggregates",
    "FORECAST_V1_INPUT_PROXY": "V1 inputs are an outbound / proxy quantity",
    "FORECAST_V1_INPUT_PERIOD_MISMATCH": "a V1 input field holds another period (loader alias)",
    "FORECAST_ROUTER_NO_SIGNAL_TYPE": "the router's daily_sales_history carries no signal type; outbound would be read as sales",
}
REASONS.update(FORECAST_CODES)


def forecast_input_audit(row: Mapping[str, Any], upload_fields: Mapping[str, Mapping[str, Any]],
                         history_supplied: bool) -> dict[str, Any]:
    codes = ["FORECAST_V1_AGGREGATE_INPUTS"]
    for name in ("sales_7d", "avg_daily_sales", "sales_30d"):
        item = upload_fields.get(name)
        if item and (item["period_conflict"] or item.get("copy_of_declared_daily")):
            codes.append("FORECAST_V1_INPUT_PERIOD_MISMATCH")
            break
    kinds = {field_signal_type(n, upload_fields[n], row)[0] for n in upload_fields if n in ("sales_qty", "avg_daily_sales", "sales_30d", "sales_7d")}
    if kinds - {OBSERVED_RETAIL_SALES}:
        codes.append("FORECAST_V1_INPUT_PROXY")
    if history_supplied:
        codes.append("FORECAST_ROUTER_NO_SIGNAL_TYPE")
    return {"version": _text(row.get("demand_forecast_version")) or None,
            "router_reason": _text(row.get("demand_forecast_reason")) or None, "codes": list(dict.fromkeys(codes))}


# ------------------------------------------------------------------------------------------------ action dependency


ACTION_DEPENDENCY = {
    "TURNOVER_INDEPENDENT": "the legacy action is the same under every turnover grade",
    "DEPENDS_ON_REPRODUCED_TURNOVER": "the action changes with the grade; the validated grade confirms it (MEDIUM/HIGH, real or claimed sales)",
    "DEPENDS_ON_UNSUPPORTED_TURNOVER": "the action changes with the grade, and the grade is unclassified, proxy, low-confidence or different",
    "DISPOSAL_INDEPENDENT": "the legacy action is the same under every disposal grade",
    "DEPENDS_ON_HEURISTIC_DISPOSAL": "the action changes with the heuristic disposal grade/score",
}
_DISPOSAL_LEVELS = (("LOW", 0.0), ("MEDIUM", 50.0), ("HIGH", 70.0), ("CRITICAL", 90.0))


def _vhs_module():
    try:
        return load_legacy_module("varo_hybrid_score")
    except LegacyAlgorithmUnavailable:  # pragma: no cover
        return None


def action_dependency(candidate: Mapping[str, Any], validated: Mapping[str, Any] | None) -> dict[str, Any]:
    """Whether the legacy rule action (varo_hybrid_score._recommend_action) depends on the turnover / disposal grade.

    The rule is evaluated unchanged on the candidate row with each possible grade substituted; no new action is chosen.
    """
    module = _vhs_module()
    if module is None:  # pragma: no cover
        return {"status": "UNAVAILABLE"}
    row = pd.Series(dict(candidate))
    actual = module._recommend_action(row)
    by_turnover = {g: module._recommend_action(pd.Series({**candidate, "turnover_grade": g}))
                   for g in ("FAST", "NORMAL", "SLOW", "DEAD")}
    by_disposal = {g: module._recommend_action(pd.Series({**candidate, "disposal_risk_grade": g, "disposal_risk_score": s}))
                   for g, s in _DISPOSAL_LEVELS}
    turnover_sensitive = len(set(by_turnover.values())) > 1
    disposal_sensitive = len(set(by_disposal.values())) > 1
    legacy = _text(candidate.get("turnover_grade")) or None
    val = validated or {}
    confirmed = (val.get("turnover_classification") == legacy and val.get("classification_confidence") in ("HIGH", "MEDIUM")
                 and val.get("classification_scope") in (SCOPE_REAL, SCOPE_CLAIMED))
    turnover_status = ("TURNOVER_INDEPENDENT" if not turnover_sensitive else
                       "DEPENDS_ON_REPRODUCED_TURNOVER" if confirmed else "DEPENDS_ON_UNSUPPORTED_TURNOVER")
    return {"legacy_rule_action": actual, "turnover_dependency": turnover_status,
            "disposal_dependency": "DEPENDS_ON_HEURISTIC_DISPOSAL" if disposal_sensitive else "DISPOSAL_INDEPENDENT",
            "action_by_turnover_grade": by_turnover, "legacy_turnover_grade": legacy,
            "validated_turnover_grade": val.get("turnover_classification"),
            "validated_confidence": val.get("classification_confidence"), "validated_scope": val.get("classification_scope")}


# ------------------------------------------------------------------------------------------------ field inventory (§2)

FIELD_INVENTORY_COLUMNS = ("field", "producer", "actual_source", "period", "unit", "null_handling", "downstream",
                           "recommendation_impact")
# Every sales / demand / stock / turnover field the code reads or writes (verified by reading the code).
FIELD_INVENTORY = (
    ("sales_qty", "upload; data_loader._normalize_inventory alias <- avg_daily_sales|sales_30d|sales_30|sales_7d|sales; "
     "Suhyup E2E: outbound of the snapshot day", "workbook column (synthetic samples) | Suhyup outbound_qty "
     "(sales_qty_semantics=demand_proxy_not_retail_sales)", "undeclared (whatever field it was copied from)", "workbook unit / UNKNOWN",
     "kept NULL in the upload; prepare_legacy_data proxy branch fillna(0).clip(>=0)", "candidate_generator demand (if no "
     "avg_daily_sales), optimality gap / transition / sensitivity daily demand, seller loss provenance",
     "candidate need (x7), target caps"),
    ("avg_daily_sales", "upload; column_aliases alias <- sales_qty|일평균판매량|일판매량", "workbook column | alias copy of sales_qty "
     "(Suhyup: one day's outbound)", "per day, window undeclared", "workbook unit", "legacy readers fillna(0)",
     "safety stock mean, EOQ, V1 forecast (hist), demand_fit_score via matcher, clustering, candidate generator demand",
     "candidate need, VHS components, demand forecast"),
    ("sales_7d", "upload; column_aliases alias <- 주판매량|7일판매량", "workbook column", "7-day sum", "workbook unit",
     "V1: fillna(0) then /7", "V1 forecast (SMA/WMA, trend), optimality gap demand_7d", "demand forecast, trend label"),
    ("sales_30d", "upload; data_loader alias <- sales_30|sales_qty|avg_daily_sales; prepare_legacy_data proxy branch = 30 x "
     "daily proxy", "workbook column | alias copy of a daily field | 30 x one day's outbound (Suhyup)", "30-day sum by name; "
     "holds a daily value when aliased", "workbook unit", "turnover / ABC / disposal / matcher fillna(0)",
     "turnover days & grade, ABC value, disposal days-of-supply points, VHS demand_fit_score (x0.12), transfer path, cutline",
     "turnover_grade DEAD/SLOW -> legacy action 할인/폐기; VHS rank"),
    ("sales_30", "upload alias of sales_30d", "workbook column", "30-day sum", "workbook unit", "as sales_30d",
     "cutline / transfer path compatibility", "none beyond sales_30d"),
    ("demand_qty", "upload | prepare_legacy_data = sales_30d (if absent) | transfer_path max(sales_30d - stock, 0)",
     "network sample: max(0, sales_30d - stock) (a 30-day shortfall); V2 samples: ~7-day forecast; legacy copy: 30-day sales",
     "overloaded (shortfall / 7-day / 30-day)", "workbook unit", "cutline fills 0", "optimality gap & transition read it as "
     "7-day demand; min_cost_network shortage", "target caps, network shortage"),
    ("demand_forecast_daily", "demand_forecast_analyzer (V1) / demand_forecast_router (V2)", "V1: WMA of sales_7d/7 and "
     "avg_daily_sales; V2: daily history", "per day, next 7 days", "inventory unit", "V1 inputs fillna(0)",
     "demand risk score, stock-out days", "VHS demand_forecast_score"),
    ("demand_forecast_7d", "V1 / router; also an upload column in V2 samples", "forecast", "7-day forecast total", "inventory unit",
     "-", "VHS demand_fit_score, optimality gap demand_7d, transition", "VHS rank, target caps"),
    ("outbound_qty", "canonical inventory_flow; Suhyup flow CSV", "observed logistics outbound (출고량)", "one day",
     "count UNKNOWN (kg DIRECT in outbound_kg)", "source zero kept; absent key-day = no record",
     "benchmark target need (7 x outbound), GeneratorRebuild sales_qty", "benchmark caps; legacy grades via sales_qty"),
    ("inbound_qty", "canonical inventory_flow", "observed inbound (입고량)", "one day", "count UNKNOWN", "source zero kept",
     "none in production", "none"),
    ("stock_qty / current_stock / inventory_qty", "upload aliases; canonical inventory_qty", "reported holding (Suhyup 재고량)",
     "snapshot", "workbook unit / UNKNOWN", "legacy readers fillna(0)", "every inventory analyzer, generator, caps",
     "everything"),
    ("avg_inventory", "upload column (samples)", "declared average", "window undeclared", "workbook unit",
     "<= 0 or NULL -> stock_qty", "turnover_rate_annual denominator", "none (rate is not graded)"),
    ("turnover_days", "turnover_analyzer", "stock_qty / (sales_30d / 30)", "days of supply", "days",
     "sales 0/NULL -> 999; negative -> negative", "turnover_grade/score, disposal reason text", "grade"),
    ("turnover_rate_annual", "turnover_analyzer", "sales_30d x 12 / avg_inventory", "annualised (x12)", "per year",
     "avg 0 -> 0.0", "dashboard summary", "none"),
    ("turnover_grade / turnover_score / turnover_risk_flag", "turnover_analyzer", "grade of turnover_days (30/60/90)",
     "-", "-", "inherits 999/negative", "VHS turnover_score (0.06), sit_DEAD_STOCK weights, _recommend_action 폐기/할인",
     "legacy action label (varo_action) and VHS rank"),
    ("abc_revenue_value / abc_grade / abc_score", "abc_analyzer", "sales_30d x unit_cost (cost absent -> 1.0)", "30-day",
     "cost-valued units (named revenue)", "NULL sales/cost -> 0", "VHS abc_score (0.04), _recommend_action 폐기, matcher ABC",
     "legacy action 폐기 branch; VHS rank"),
    ("disposal_risk_score / disposal_risk_grade", "disposal_risk_analyzer", "heuristic points (expiry, days of supply, inbound age, category)",
     "-", "points 0-100", "missing inputs -> 0 / 999", "VHS disposal (0.09), auto VHS disposal_risk_score input, _recommend_action, matcher urgency",
     "legacy action 할인/폐기; Varo Final rank"),
    ("safety_stock / reorder_point / reorder_status", "safety_stock_analyzer", "z x std x sqrt(L); mean x L + SS",
     "per day x lead days", "inventory unit", "std 0.3 x mean, L 3 placeholders", "VHS, matcher urgency, _recommend_action 재배치",
     "legacy action / VHS"),
    ("match_score / match_grade", "store_product_matcher", "demand ratio, urgency, ABC, cluster, distance", "-", "points",
     "target sales absent -> source (neutral)", "_recommend_action 재배치 branch, VHS", "재배치 unreachable on Suhyup"),
    ("source_daily_demand / target_daily_demand", "seller_loss_engine inputs", "seller input or forecast rows",
     "per day", "declared", "MISSING stays MISSING; proxy -> PROXY", "Seller Loss strategies", "parallel field only"),
    ("target_daily_demand_proxy / target_need_7d", "suhyup_algorithm_revalidation.enrich_inventory_constraints",
     "same-day outbound; max(0, median + 7 x outbound - stock)", "one day / 7 days", "UNKNOWN", "None when no row",
     "T1 / T3 / P1 caps", "benchmark plans (PROXY)"),
    ("sales_velocity / turnover_rate", "not present in the code", "-", "-", "-", "-", "-", "-"),
)


# ------------------------------------------------------------------------------------------------ pipeline field (§32)


def _inventory_frame(data: Mapping[str, Any]) -> pd.DataFrame | None:
    frame = data.get("inventory") if isinstance(data, Mapping) else None
    return frame if isinstance(frame, pd.DataFrame) and not frame.empty else None


def _aligned(upload: pd.DataFrame, analyzed: pd.DataFrame | None) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    rows = upload.reset_index(drop=True).to_dict("records")
    if analyzed is None or len(analyzed) != len(upload):
        return [(row, None) for row in rows]
    out = []
    for row, legacy in zip(rows, analyzed.reset_index(drop=True).to_dict("records")):
        same = (_text(row.get("store_id")), _text(row.get("product_id"))) == (_text(legacy.get("store_id")), _text(legacy.get("product_id")))
        out.append((row, legacy if same else None))
    return out


def evaluate_inventory_row(row: Mapping[str, Any], legacy: Mapping[str, Any] | None, fields: Mapping[str, Mapping[str, Any]],
                           *, upload_columns: Sequence[str], legacy_columns: Sequence[str],
                           history_supplied: bool = False, dataset_id: str | None = None) -> dict[str, Any]:
    """Validated sales signal, days of supply, turnover and grade of one upload row next to the legacy outputs."""
    signal, field_used, skipped = row_sales_signal(row, fields, dataset_id=dataset_id)
    stock = _finite(row.get("stock_qty"))
    if "avg_inventory" in upload_columns and _finite(row.get("avg_inventory")) is not None:
        inventory = declared_average_inventory(row.get("avg_inventory"), dataset_id=dataset_id)
    else:
        inventory = {"dataset_id": dataset_id, "value": None, "method": "SINGLE_SNAPSHOT", "status": AVG_INVENTORY_UNAVAILABLE,
                     "gate_status": MISSING_DATA, "reason_codes": ["SINGLE_SNAPSHOT_AS_AVERAGE"], "quantity_unit": None,
                     "snapshot_count": 1 if stock is not None else 0, "period_start": None, "period_end": None}
    if signal is None:
        gate = PERIOD_MISMATCH if "ALIAS_COPY_ACROSS_PERIODS" in skipped or "PERIOD_UNDECLARED" in skipped else MISSING_DATA
        window = {"period_days": None, "gate_status": gate, "daily_rate": None}
        dos = days_of_supply(stock, None)
        turnover = {"turnover_value": None, "turnover_period_days": None, "days_of_supply": dos["value"],
                    "days_of_supply_status": dos["status"],
                    "turnover_status": TURNOVER_UNAVAILABLE if gate == MISSING_DATA else PERIOD_MISMATCH,
                    "gate_status": gate, "turnover_basis": NO_BASIS, "turnover_classification": UNCLASSIFIED,
                    "classification_confidence": "NONE", "classification_scope": SCOPE_NONE,
                    "reason_codes": _reasons(*(skipped or ["SALES_MISSING"]), *dos["reasons"])}
    else:
        window = signal_window(signal)
        if signal.aggregation in PER_DAY_AGGREGATIONS:
            turnover = physical_turnover({**window, "quantity": signal.quantity}, inventory, end_stock=stock,
                                         per_day_only=True)
        else:
            turnover = physical_turnover(window, inventory, end_stock=stock)
        if skipped:
            turnover["reason_codes"] = _reasons(*turnover["reason_codes"], *skipped)
    columns = legacy_columns or upload_columns
    legacy_inputs = legacy_turnover_inputs(legacy or row, columns, fields)
    legacy_grade_value = _text((legacy or {}).get("turnover_grade")) or None
    cause = grade_cause(legacy_grade_value, turnover, legacy_inputs["legacy_codes"])
    disposal = disposal_input_audit(legacy or row, columns, fields)
    safety = safety_stock_input_audit(legacy or row, columns, fields)
    match = store_product_match_audit(legacy or row, columns)
    return {
        "store_id": _text(row.get("store_id")) or None, "product_id": _text(row.get("product_id")) or None,
        "sales_field_used": field_used, "sales_signal_type": signal.signal_type if signal else None,
        "sales_semantics_evidence": signal.semantics_evidence if signal else None,
        "sales_aggregation": signal.aggregation if signal else None, "sales_period_days": window.get("period_days"),
        "sales_gate": window.get("gate_status"), "validated_daily_rate": window.get("daily_rate"),
        "stock_qty": stock, "avg_inventory_method": inventory.get("method"),
        "legacy_turnover_days": _finite((legacy or {}).get("turnover_days")),
        "legacy_turnover_rate_annual": _finite((legacy or {}).get("turnover_rate_annual")),
        "legacy_turnover_grade": legacy_grade_value, "legacy_sales_input": legacy_inputs["legacy_sales_input"],
        "legacy_sales_used": legacy_inputs["legacy_sales_used"], "legacy_codes": legacy_inputs["legacy_codes"],
        "days_of_supply": turnover["days_of_supply"] if turnover["days_of_supply"] is None or not math.isinf(turnover["days_of_supply"]) else "INF",
        "days_of_supply_status": turnover["days_of_supply_status"], "turnover_value": turnover["turnover_value"],
        "turnover_period_days": turnover.get("turnover_period_days"), "turnover_status": turnover["turnover_status"],
        "turnover_basis": turnover["turnover_basis"], "turnover_classification": turnover["turnover_classification"],
        "classification_confidence": turnover["classification_confidence"],
        "classification_scope": turnover["classification_scope"], "gate_status": turnover["gate_status"],
        "reason_codes": turnover["reason_codes"], "grade_cause": cause,
        "abc_legacy_grade": _text((legacy or {}).get("abc_grade")) or None,
        "disposal_legacy_grade": _text((legacy or {}).get("disposal_risk_grade")) or None,
        "disposal_legacy_score": _finite((legacy or {}).get("disposal_risk_score")),
        "disposal_components": disposal["components"], "disposal_codes": disposal["codes"],
        "safety_stock_mean_source": safety["demand_mean_source"], "safety_stock_codes": safety["codes"],
        "match_use": match["use"], "match_codes": match["codes"],
        "forecast_codes": forecast_input_audit(legacy or row, fields, history_supplied)["codes"],
    }


def build_inventory_metric_integrity(uploaded_data: Mapping[str, Any], analyzed_inventory: pd.DataFrame | None = None,
                                     candidates: pd.DataFrame | None = None) -> dict[str, Any]:
    """The parallel PipelineResult.inventory_metric_integrity field; it reads and never writes production values."""
    base = {"status": "parallel_only", "contract_version": CONTRACT_VERSION, "production_action_applied": False,
            "legacy_metrics_replaced": False, "thresholds_changed": False}
    upload = _inventory_frame(uploaded_data)
    if upload is None:
        return {**base, "status": "no_inventory", "rows": [], "counts": {}}
    fields = upload_sales_fields(upload)
    analyzed = analyzed_inventory if isinstance(analyzed_inventory, pd.DataFrame) and not analyzed_inventory.empty else None
    legacy_columns = list(analyzed.columns) if analyzed is not None else []
    history = uploaded_data.get("daily_sales_history") if isinstance(uploaded_data, Mapping) else None
    history_supplied = isinstance(history, pd.DataFrame) and not history.empty
    rows = [evaluate_inventory_row(row, legacy, fields, upload_columns=list(upload.columns),
                                   legacy_columns=legacy_columns, history_supplied=history_supplied)
            for row, legacy in _aligned(upload, analyzed)]
    aligned = sum(1 for _, legacy in _aligned(upload, analyzed) if legacy is not None)
    abc_rows = []
    for row in upload.to_dict("records"):
        signal, _, _ = row_sales_signal(row, fields)
        window = signal_window(signal) if signal is not None else {}
        period_sales = window.get("quantity") if window.get("aggregation") in (WINDOW_SUM, CALENDAR_MONTH_SUM) else None
        if period_sales is None and window.get("daily_rate") is not None and window.get("aggregation") == DAILY_MEAN:
            period_sales = None
        abc_rows.append({"product_id": _text(row.get("product_id")), "period_sales": period_sales,
                         "unit_price": row.get("unit_price"), "unit_cost": row.get("unit_cost"),
                         "signal_type": signal.signal_type if signal else UNKNOWN_SIGNAL, "quantity_unit": None})
    validated = validated_abc(abc_rows)
    legacy_abc = {}
    if analyzed is not None and {"product_id", "abc_grade"} <= set(analyzed.columns):
        for pid, grade in zip(analyzed["product_id"].map(_text), analyzed["abc_grade"].map(_text)):
            legacy_abc.setdefault(pid, grade)
    abc_compare = Counter()
    for pid, item in validated.get("grades", {}).items():
        abc_compare["same" if legacy_abc.get(pid) == item["grade"] else "different"] += 1
    actions = []
    if isinstance(candidates, pd.DataFrame) and not candidates.empty:
        by_key = {(r["store_id"], r["product_id"]): r for r in rows}
        for cand in candidates.to_dict("records"):
            key = (_text(cand.get("source_id")), _text(cand.get("product_id")))
            dep = action_dependency(cand, by_key.get(key))
            actions.append({"route_id": _text(cand.get("route_id")) or None, "source_id": key[0], "product_id": key[1],
                            "varo_action": _text(cand.get("varo_action")) or None, **{k: dep.get(k) for k in (
                                "legacy_rule_action", "turnover_dependency", "disposal_dependency", "legacy_turnover_grade",
                                "validated_turnover_grade", "validated_confidence", "validated_scope")}})
    counts = {
        "rows": len(rows), "legacy_rows_aligned": aligned,
        "legacy_dead_from_missing_sales": sum(1 for r in rows if r["grade_cause"] == "MISSING_SALES_GRADED_DEAD"),
        "legacy_fast_from_negative_sales": sum(1 for r in rows if r["grade_cause"] == "NEGATIVE_SALES_GRADED_FAST"),
        "legacy_graded_on_zero_stock": sum(1 for r in rows if r["grade_cause"] == "ZERO_STOCK_GRADED"),
        "legacy_period_alias_rows": sum(1 for r in rows if r["grade_cause"] == "PERIOD_ALIAS_DISTORTION"),
        "observed_zero_sales_rows": sum(1 for r in rows if r["days_of_supply_status"] == DOS_NO_MOVEMENT),
        "average_inventory_zero_rows": sum(1 for r in rows if r["turnover_status"] == DIVISION_BY_ZERO),
        "unit_mismatch_rows": sum(1 for r in rows if r["turnover_status"] == UNIT_MISMATCH),
        "partial_period_rows": sum(1 for r in rows if r["turnover_status"] == PARTIAL_PERIOD),
        "grade_causes": dict(sorted(Counter(r["grade_cause"] for r in rows).items())),
        "turnover_status": dict(sorted(Counter(r["turnover_status"] for r in rows).items())),
        "turnover_basis": dict(sorted(Counter(r["turnover_basis"] for r in rows).items())),
        "gate_status": dict(sorted(Counter(r["gate_status"] for r in rows).items())),
        "classification_confidence": dict(sorted(Counter(r["classification_confidence"] for r in rows).items())),
        "legacy_vs_validated_grade": dict(sorted(Counter(f"{r['legacy_turnover_grade']}->{r['turnover_classification']}"
                                                         for r in rows).items())),
        "real_sales_turnover_rows": sum(1 for r in rows if r["turnover_basis"] == REAL_SALES_TURNOVER
                                        and r["classification_scope"] == SCOPE_REAL),
    }
    return {
        **base,
        "sales_fields": {name: {k: v for k, v in item.items()} for name, item in fields.items()},
        "rows": rows,
        "counts": counts,
        "abc": {**legacy_abc_basis(legacy_columns or upload.columns), "validated_basis": validated["basis"],
                "validated_gate": validated["gate_status"], "validated_reasons": validated["reason_codes"],
                "validated_grades": validated["grades"], "legacy_vs_validated": dict(abc_compare),
                "thresholds": validated.get("thresholds")},
        "disposal_risk": {"meaning": DISPOSAL_MEANING, "note": DISPOSAL_MEANING_NOTE, "is_probability": False,
                          "code_counts": dict(sorted(Counter(c for r in rows for c in r["disposal_codes"]).items()))},
        "safety_stock": {"code_counts": dict(sorted(Counter(c for r in rows for c in r["safety_stock_codes"]).items()))},
        "store_product_match": {"code_counts": dict(sorted(Counter(c for r in rows for c in r["match_codes"]).items())),
                                "reference_only_rows": sum(1 for r in rows if r["match_use"] == "REFERENCE_SCORE_ONLY")},
        "forecast_inputs": {"history_supplied": history_supplied,
                            "code_counts": dict(sorted(Counter(c for r in rows for c in r["forecast_codes"]).items()))},
        "financial_turnover": financial_turnover(),
        "action_dependency": {"rows": actions,
                              "counts": dict(sorted(Counter(a["turnover_dependency"] for a in actions).items())),
                              "disposal_counts": dict(sorted(Counter(a["disposal_dependency"] for a in actions).items()))},
    }
