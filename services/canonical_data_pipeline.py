"""Bounded-memory offline canonical exports and evidence reports.

Run: python -m services.canonical_data_pipeline --data-root C:/VARO_V2_REAL_DATA
Only canonical_* files and new reports are written; raw is read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from services.canonical_schema import (COMMON, KEYS, LINEAGE, NUMERIC, TABLE_FIELDS, VERSION, add_flag, canonicalize,
    finalize_quality, schema_document)
from services.real_data_adapters import (AIHUB_MEASUREMENT_CONTEXT, DATA_ROOT, DATASETS, EXTERNAL_DATASETS, LOGISALL_COLLAPSE, NFQS_FLOAT_RESIDUE_TOLERANCE_TON,
    NFQS_IDENTITY_TOLERANCE_TON, REGIONS, UNIT_EVIDENCE, adapt_aihub, adapt_aihub_measurements, adapt_jangbogo, adapt_logisall, adapt_nfqs,
    adapt_suhyup, base_frame, classify_name_versions, collapse_source_records, csv_chunks, csv_encoding, mapped, nfqs_flow_identity_diff,
    nfqs_panel_coverage, pack_spec_tokens, suhyup_to_existing, verified_identifier_crosswalk)

ALGORITHMS = {
    "ABC": ["inventory_qty", "unit_cost"], "Turnover": ["inventory_qty", "sales_qty", "aligned_period"],
    "Disposal Risk": ["inventory_qty", "expiry_date"], "Demand Forecast": ["sales_qty", "date", "consistent_unit"],
    "Safety Stock": ["demand_qty", "lead_time_days", "demand_std"], "EOQ": ["demand_qty", "ordering_cost", "holding_cost"],
    "Store/Product Matching": ["inventory_qty", "location_id", "product_id", "target_need"],
    "Transport Cost": ["distance_km", "transport_cost", "currency"],
    "Greedy": ["recommended_qty", "source_surplus", "target_need", "transport_cost"],
    "VHS": ["recommended_qty", "transport_cost", "feasibility_score"],
    "Pareto": ["recommended_qty", "transport_cost", "source_surplus", "target_need"],
    "DQN": ["actual_state_transition", "reward_validation", "compatible_model"],
    "MILP": ["recommended_qty", "transport_cost", "source_surplus", "target_need"],
    "Optimality Gap": ["recommended_qty", "transport_cost", "source_surplus", "target_need"],
    "Varo Final": ["recommended_qty", "transport_cost", "vhs_score", "source_surplus", "target_need"],
}
ROUTING = {"Transport Cost", "Greedy", "VHS", "Pareto", "MILP", "Optimality Gap", "Varo Final", "DQN"}
# FULL needs a recorded semantic review, never column presence alone: {(dataset, algorithm): evidence}.
# The gate below still applies: every required field (including verified pseudo-fields such as consistent_unit)
# must be present and the quantity unit known; otherwise the review alone grants nothing.
FULL_APPROVALS: dict = {
    ("m5", "Demand Forecast"): ("Semantic review 2026-09-29: target is observed daily store x item unit sales (M5 guide: 'The number of units sold at day i', "
                                "the official M5 forecasting target); unit item SOURCE_METADATA; key (date, store, item) verified unique over all 59,181,090 cells; "
                                "no NULL, negative or fractional cell; calendar.csv maps every d_* column to exactly one date; holdout ground truth exists in the data "
                                "(d_1914-d_1941, the official validation window). Limitation: sales are censored by unobserved stock-outs; this is sales forecasting, "
                                "not latent-demand recovery."),
}
# Evidence used to gate coverage; FULL additionally needs observed (non-proxy) constraints and a known quantity unit.
DATASET_PROFILES = {
    "suhyup": {"grain": "daily center x product x processing-state stock and flow; 620 existing benchmark route candidates (31 days)",
               "quantity_unit_status": "UNKNOWN (parallel kg DIRECT)", "location_identification": "9 identified centers",
               "observed_network_constraints": False, "quantity_unit_known": False},
    "logisall": {"grain": "monthly national stock; zone-daily sales; zone-to-zone daily distribution (masked-ZIP sub-zone records summed)",
                 "quantity_unit_status": "UNKNOWN", "location_identification": "7 first-digit ZIP zones, not facilities",
                 "observed_network_constraints": False, "quantity_unit_known": False},
    "jangbogo": {"grain": "warehouse x category monthly sales/purchase requests; product x barcode monthly purchase requests; warehouse x SKU daily occurrences",
                 "quantity_unit_status": "UNKNOWN", "location_identification": "warehouse codes for category/occurrence tables; none for product purchase requests",
                 "observed_network_constraints": False, "quantity_unit_known": False},
    "nfqs": {"grain": "quarterly snapshot, 14 regions + national total, 13-14 product groups", "quantity_unit_status": "SOURCE_METADATA (ton)",
             "location_identification": "NFQS branch regions (aggregates of cooperating firms)", "observed_network_constraints": False, "quantity_unit_known": True},
    "aihub": {"grain": "61 daily site-aggregate rows; 20,480 item measurements", "quantity_unit_status": "SOURCE_METADATA (item); volume DIRECT (m3)",
              "location_identification": "none for flow rows; 2 metadata sites cannot be joined", "observed_network_constraints": False, "quantity_unit_known": True},
    "m5": {"grain": "daily store x item unit sales (10 stores, 3,049 items, 1,941 days); weekly store x item sell price joined to its days; daily calendar events and state SNAP indicators",
           "quantity_unit_status": "SOURCE_METADATA (item: 'number of units sold')", "location_identification": "10 identified stores in 3 states",
           "observed_network_constraints": False, "quantity_unit_known": True},
    "favorita": {"grain": "daily store x item unit sales (zero-sales rows absent in source); daily store transactions; daily oil price; dated holidays/events",
                 "quantity_unit_status": "UNKNOWN (item-dependent count or kg, not labelled)", "location_identification": "54 identified stores (city/state/type/cluster)",
                 "observed_network_constraints": False, "quantity_unit_known": False},
    "freshretailnet": {"grain": "daily store x product sales with 24 hourly sales and hourly stockout values; store-day weather; national daily holiday indicator",
                       "quantity_unit_status": "SOURCE_METADATA (globally normalised sales amount; coefficient undisclosed)",
                       "location_identification": "898 encoded stores in 18 encoded cities", "observed_network_constraints": False, "quantity_unit_known": True,
                       "quantity_unit_blocker": "quantity is a globally normalised amount, not a physical unit"},
    "kamp": {"grain": "shipment event (date x ordering project x building part) x rebar grade", "quantity_unit_status": "UNKNOWN (guidebook unit column empty)",
             "location_identification": "none: the shipping plant is not keyed; 공사/부위 are an ordering project and building part, not locations",
             "observed_network_constraints": False, "quantity_unit_known": False},
}
# External datasets: explicit decisions; every other algorithm is UNSUPPORTED with the dataset default reason.
EXTERNAL_COVERAGE = {
    "m5": {"default": "M5 provides observed store x item unit sales, weekly sell prices and calendar context only; no on-hand inventory, cost, lead time, capacity or store-to-store network exists, so the required inputs are absent (no synthetic stock or route is generated).",
           "ABC": ("PARTIAL", "A revenue ranking is computable from observed unit sales x weekly sell_price (USD per the M5 guide's dollar-sales weighting); Varo ABC multiplies sales by unit_cost, and a sell price is not a cost; no inventory value exists."),
           "Demand Forecast": ("PARTIAL", "Observed daily store x item unit sales with unit item; FULL only when the recorded semantic review and the full gate hold.")},
    "favorita": {"default": "Favorita provides observed store x item unit sales, promotions, store transactions, oil price and holidays only; no inventory, price, cost, lead time, capacity or network exists, so the required inputs are absent.",
                 "Demand Forecast": ("PARTIAL", "Observed daily store x item unit_sales (125,497,040 rows) with promotion and store transactions; the unit is item-dependent (count or kg per the Kaggle description) and unlabelled, so unit UNKNOWN and items cannot be pooled; zero-sales days are absent from the source (absence is not zero; stock availability is unknown); negative values are documented returns; test.csv has no ground truth, so validation must hold out train dates.")},
    "freshretailnet": {"default": "FreshRetailNet provides normalised sales, hourly stockout status, discounts and weather only; the stock level itself is not released (only out-of-stock hours), and no cost, lead time, capacity or network exists.",
                       "Demand Forecast": ("PARTIAL", "Observed daily and hourly store x product sales for 50,000 series with hourly stockout annotations (censored-demand structure) and a 7-day eval split with actuals; sales are globally normalised by an undisclosed coefficient, so forecasts cannot be converted to physical quantities for replenishment.")},
    "kamp": {"default": "KAMP releases order-based shipment quantities only: no inventory (the guidebook's MS-SQL stock collection is not in the workbook), no cost, lead time, capacity, identified shipping location or source/target network.",
             "Demand Forecast": ("PARTIAL", "Order-based (바리스트) outbound shipment quantity per rebar grade for one masked construction project, dated per shipment event (186 dates); a shipment series, not retail sales or demand; unit not stated (UNKNOWN).")},
}


def algorithm_coverage(dataset, present):
    """Conservative semantic gate. A union of column names never proves FULL."""
    profile = DATASET_PROFILES.get(dataset, {})
    rows = []
    for algorithm, required in ALGORITHMS.items():
        have = sorted(set(required) & set(present))
        missing = sorted(set(required) - set(present))
        level, reason = "UNSUPPORTED", "Required semantics/fields absent; no forced algorithm execution."
        if dataset in EXTERNAL_COVERAGE:
            decision = EXTERNAL_COVERAGE[dataset]
            level, reason = decision.get(algorithm, ("UNSUPPORTED", decision["default"]))
        elif dataset == "suhyup" and algorithm in ROUTING:
            level = "BENCHMARK_ONLY"
            reason = "Existing 620-candidate/31-day benchmark only; outbound is not retail demand, target_need is the existing proxy, route costs are enriched reference estimates."
        elif dataset == "logisall" and algorithm == "Demand Forecast":
            level = "PARTIAL"
            reason = "Actual zone-daily sales; repeated masked-ZIP sub-zone records summed with full lineage; quantity unit UNKNOWN; 7 coarse zones, not stores."
        elif dataset == "jangbogo" and algorithm == "Demand Forecast":
            level = "PARTIAL"
            reason = "Actual warehouse x category monthly sales; unit UNKNOWN; category and SKU grains cannot be pooled; cross-release conflicting months are ineligible."
        elif dataset in {"suhyup", "aihub"} and algorithm == "Turnover":
            level = "PARTIAL"
            reason = ("Stock and outbound available for logistics-flow diagnostics; outbound cannot substitute for retail sales_30d."
                      + (" AI Hub rows are one unidentified site aggregate." if dataset == "aihub" else ""))
        elif dataset == "suhyup" and algorithm == "Store/Product Matching":
            level = "PARTIAL"
            reason = "Actual location/product/stock exists; unmet retail demand is unavailable."
        blockers = [f"missing fields: {', '.join(missing)}"] if missing else []
        if not profile.get("quantity_unit_known", False):
            blockers.append("quantity unit UNKNOWN")
        elif profile.get("quantity_unit_blocker"):
            blockers.append(profile["quantity_unit_blocker"])
        if algorithm in ROUTING and not profile.get("observed_network_constraints", False):
            blockers.append("no observed source surplus/target need/cost constraints")
        if not blockers and (dataset, algorithm) not in FULL_APPROVALS:
            blockers.append("no recorded semantic review approving FULL")
        if not blockers:
            level, reason = "FULL", FULL_APPROVALS[(dataset, algorithm)]
        rows.append({"algorithm": algorithm, "supported": level != "UNSUPPORTED", "support_level": level,
                     "required_fields_present": "|".join(have), "missing_fields": "|".join(missing), "reason": reason,
                     "full_gate_blockers": "; ".join(blockers), "data_grain": profile.get("grain"),
                     "quantity_unit_status": profile.get("quantity_unit_status"), "location_identification": profile.get("location_identification")})
    return pd.DataFrame(rows)


MASTER_ATTRIBUTES = {"product_master": ("product_id", "product_name", "category", "product_state", "product_id_namespace"),
                     "location_master": ("location_id", "location_name", "location_type", "region", "address", "location_id_namespace")}
IDENTITY_TABLES = {"inventory_snapshot", "inventory_flow", "transfer_network", "demand_series", "location_product_observation", "product_measurement"}
FLAG_COLUMNS = ("validation_flags", "quality_flags", "quality_status", "analysis_eligible")


def flag_statistics(path, batch_size=1_000_000):
    """Flag/quality tallies of a written table, streamed in batches (bounded memory on large tables)."""
    import pyarrow.parquet as pq
    whole = {c: Counter() for c in ("validation_flags", "quality_flags")}
    counts = {c: Counter() for c in ("quality_status", "analysis_eligible")}
    nonnull = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(FLAG_COLUMNS)):
        frame = batch.to_pandas()
        for col in FLAG_COLUMNS:
            nonnull[col] += int(frame[col].notna().sum())
        for col in whole:
            whole[col].update(frame[col].fillna("").astype(str).value_counts().to_dict())
        for col in counts:
            counts[col].update(frame[col].value_counts().to_dict())
    split = {col: Counter() for col in whole}
    for col, tally in whole.items():
        for value, n in tally.items():
            for part in str(value).split("|"):
                if part:
                    split[col][part] += n
    ordered = lambda c: dict(sorted(c.items(), key=lambda kv: -kv[1]))
    return {"flags": split["validation_flags"], "nonnull": nonnull,
            "quality": {"quality_status": ordered(counts["quality_status"]), "quality_flags": dict(split["quality_flags"]),
                        "analysis_eligible": ordered(counts["analysis_eligible"])}}


def adjacent_identical_positions(frame):
    """1-based record ids identical in every column to the record directly above."""
    same = (frame == frame.shift()).fillna(False).astype(bool).all(axis=1).to_numpy(dtype=bool, copy=True)
    if len(same):
        same[0] = False
    return [int(i) + 1 for i in same.nonzero()[0]]


def key_hashes(frame, table):
    # String-normalised so hashes written in chunks match hashes of re-read parquet.
    return pd.util.hash_pandas_object(frame[list(KEYS[table])].astype("string"), index=False)


def parquet_schema(pa, columns):
    return pa.schema([(c, pa.float64() if c in NUMERIC else pa.string()) for c in columns])


class Exporter:
    def __init__(self, root, dataset):
        import pyarrow as pa
        self.pa = pa
        self.root, self.dataset = Path(root), dataset
        self.folder = self.root / DATASETS[dataset]
        self.processed, self.results = self.folder / "processed", self.folder / "results"
        self.transform_version = VERSION
        self.unit_evidence = UNIT_EVIDENCE.get(dataset)
        # Tables whose key uniqueness is checked with one 64-bit hash per row at finish() instead of a per-row
        # Counter (bounded memory for 10^7-10^8 row sources); repeats are flagged in the same reflag pass.
        self.bulk_key_tables = set()
        self.bulk_hashes = defaultdict(list)
        self.processed.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.writers, self.key_counts = {}, defaultdict(Counter)
        self.counts, self.flags, self.nonnull = Counter(), defaultdict(Counter), defaultdict(Counter)
        self.sources, self.mappings, self.conservation = [], [], []
        self.master_parts = defaultdict(list)
        self.identity_parts, self.location_parts = [], []
        self.notes = []
        self.present = set()
        self.date_ranges = {}
        self.checks = {}
        self.entity_references = defaultdict(set)
        self.units = defaultdict(set)
        self.unit_status = defaultdict(Counter)
        self.quality = {}
        # table -> [(source_file, {1-based source record ids}, flag)] applied after export.
        self.post_flags = defaultdict(list)

    def record_mapping(self, table, mapping, source, measures, grain, scope, unit=None, derived=None, unit_status=None):
        derived = derived or {}
        unit_status = unit_status or ("UNKNOWN" if unit is None else None)
        fields = {}
        metadata = {"date_grain": grain, "scope": scope, "transform_version": self.transform_version, "source_dataset": self.dataset,
                    "source_file": source, "source_sheet_or_table": "source table", "source_row_id": "1-based source record"}
        for field in (*COMMON, *TABLE_FIELDS[table]):
            if field in derived:
                fields[field] = {"coverage": "DERIVED", **derived[field]}
            elif field in mapping:
                fields[field] = {"coverage": "DIRECT" if field == mapping[field] else "RENAMED", "source_columns": [mapping[field]], "unit": unit if field in measures else None,
                                 **({"unit_status": unit_status} if field in measures else {})}
            elif field in metadata:
                fields[field] = {"coverage": "DERIVED", "source_columns": [], "formula": str(metadata[field]), "assumptions": "Source structure metadata; not a fabricated observation", "unit": None}
            elif field == "unit" and unit:
                fields[field] = {"coverage": "DERIVED", "source_columns": [], "formula": unit, "assumptions": "Explicit source header/documentation unit", "unit": unit, "unit_status": unit_status}
            else:
                fields[field] = {"coverage": "UNAVAILABLE", "source_columns": [], "unit": None}
        self.mappings.append({"source_file": source, "table": table, "date_grain": grain, "scope": scope, "fields": fields})

    def write(self, table, frame, raw=None, measures=None, masters=True):
        import pyarrow.parquet as pq
        frame = frame.copy()
        if table in self.bulk_key_tables:
            self.bulk_hashes[table].append(key_hashes(frame, table).to_numpy(dtype="uint64", copy=True))
        else:
            counts = self.key_counts[table]
            duplicate = []
            for key in key_hashes(frame, table).tolist():
                duplicate.append(counts[key] > 0)
                counts[key] += 1
            mask = pd.Series(duplicate, index=frame.index)
            if mask.any():
                # Repeats beyond the first are marked now; finish() classifies every member.
                finalize_quality(add_flag(frame, mask, "duplicate_key"))
        self.counts[table] += len(frame)
        for col in frame:
            n = int(frame[col].notna().sum())
            self.nonnull[table][col] += n
            if n:
                self.present.add(col)
        datecol = "snapshot_date" if table == "inventory_snapshot" else "date"
        if datecol in frame and frame[datecol].notna().any():
            low, high = frame[datecol].dropna().min(), frame[datecol].dropna().max()
            old = self.date_ranges.get(table, [low, high])
            self.date_ranges[table] = [min(low, old[0]), max(high, old[1])]
        if raw is not None:
            records = (pd.to_numeric(frame["source_record_count"], errors="coerce").fillna(1) if "source_record_count" in frame
                       else pd.Series(1.0, index=frame.index))
            represented = int(records.sum())
            if len(raw) != represented:
                raise AssertionError(f"Record conservation failure: {table} represents {represented} of {len(raw)} source records")
            for target, origin in (measures or {}).items():
                before = pd.to_numeric(raw[origin], errors="coerce")
                after = pd.to_numeric(frame[target], errors="coerce")
                a, b = before.sum(min_count=1), after.sum(min_count=1)
                ok = (pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-7))
                nonnull_records = int(records[after.notna()].sum())
                self.conservation.append({"source_file": str(frame.source_file.iloc[0]), "table": table, "field": target,
                    "source_column": origin, "rows_source": len(raw), "rows_canonical": len(frame), "source_records_represented": represented,
                    "source_sum": None if pd.isna(a) else float(a), "canonical_sum": None if pd.isna(b) else float(b),
                    "source_nonnull": int(before.notna().sum()), "canonical_nonnull": int(after.notna().sum()), "canonical_nonnull_records": nonnull_records,
                    "passed": bool(ok and len(raw) == represented and before.notna().sum() == nonnull_records)})
                if not self.conservation[-1]["passed"]:
                    raise AssertionError(f"Conservation failure: {table}.{target}")
            if represented != len(frame):
                ids = frame["source_row_id"].astype(str).str.split("|").explode()
                self.conservation.append({"source_file": str(frame.source_file.iloc[0]), "table": table, "field": "source_record_count",
                    "source_column": "(all records)", "rows_source": len(raw), "rows_canonical": len(frame), "source_records_represented": represented,
                    "distinct_source_row_ids": int(ids.nunique()), "passed": len(raw) == represented == ids.nunique() == len(ids)})
                if not self.conservation[-1]["passed"]:
                    raise AssertionError(f"Record lineage failure: {table}")
        schema = parquet_schema(self.pa, frame.columns)
        arr = self.pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        if table not in self.writers:
            self.writers[table] = pq.ParquetWriter(self.processed / f"canonical_{table}.parquet", schema, compression="zstd")
        self.writers[table].write_table(arr)
        if masters:
            self.collect_masters(frame)
        for col, entity in [("product_id", "product_master"), ("location_id", "location_master"), ("source_id", "location_master"), ("target_id", "location_master")]:
            if col in frame and table not in {"product_master", "location_master", "product_identity_version"}:
                self.entity_references[entity].update(frame[col].dropna().astype(str).unique())
        self.units[table].update(frame.unit.dropna().astype(str).unique())
        self.unit_status[table].update(frame.unit_status.fillna("<NULL>").astype(str).value_counts().to_dict())
        if table in IDENTITY_TABLES:
            self.collect_identity(table, frame)

    def collect_identity(self, table, frame):
        """Every (id, name) pair with its observed period; exact, not first-per-chunk."""
        if "product_id" in frame and "product_name" in frame:
            datecol = "snapshot_date" if table == "inventory_snapshot" else ("date" if "date" in frame else None)
            part = frame[["product_id", "product_name", "source_file"]].copy()
            if datecol:
                part["first"] = frame["period_start"].fillna(frame[datecol])
                part["last"] = frame["period_end"].fillna(frame[datecol])
            else:
                part["first"] = part["last"] = pd.NA
            part = part.dropna(subset=["product_id", "product_name"])
            if not part.empty:
                grouped = part.groupby(["product_id", "product_name", "source_file"], dropna=False).agg(
                    first=("first", "min"), last=("last", "max"), n=("product_id", "size")).reset_index()
                grouped["table"], grouped["dated"] = table, datecol is not None
                self.identity_parts.append(grouped)
        if "location_id" in frame and "location_name" in frame:
            loc = frame[["location_id", "location_name"]].dropna().drop_duplicates()
            if not loc.empty:
                self.location_parts.append(loc)

    def collect_masters(self, frame):
        for table, idcol in [("product_master", "product_id"), ("location_master", "location_id")]:
            if idcol not in frame:
                continue
            cols = list(dict.fromkeys([*LINEAGE, "date_grain", "scope", "product_grain", "unit", "unit_status", *[c for c in MASTER_ATTRIBUTES[table] if c in frame]]))
            part = frame.loc[frame[idcol].notna(), cols].drop_duplicates(list(c for c in KEYS[table] if c in cols))
            if not part.empty:
                self.master_parts[table].append(part)
        if "source_id" in frame:
            for field in ("source_id", "target_id"):
                part = frame[[*LINEAGE, "date_grain", "scope", field]].rename(columns={field: "location_id"}).dropna(subset=["location_id"]).drop_duplicates("location_id")
                self.master_parts["location_master"].append(part)

    def identity_summary(self):
        if not self.identity_parts:
            return pd.DataFrame(columns=["product_id", "product_name", "dated", "first", "last", "n", "tables", "files"])
        data = pd.concat(self.identity_parts, ignore_index=True)
        return data.groupby(["product_id", "product_name", "dated"]).agg(
            first=("first", "min"), last=("last", "max"), n=("n", "sum"),
            tables=("table", lambda s: "|".join(sorted(set(s)))), files=("source_file", lambda s: "|".join(sorted(set(s))))).reset_index()

    def identity_versions(self, summary):
        rows = []
        if summary.empty:
            return pd.DataFrame(rows)
        for pid, group in summary[summary.dated.astype(bool)].groupby("product_id", sort=True):
            group = group.sort_values(["first", "product_name"], na_position="last")
            records = [{"product_name": r.product_name, "valid_from": r.first, "valid_to": r.last} for r in group.itertuples()]
            temporal, change = classify_name_versions(records)
            for seq, r in enumerate(group.itertuples(), 1):
                files = r.files.split("|")
                rows.append({"source_dataset": self.dataset, "source_file": files[0], "source_sheet_or_table": "derived:product_identity_version",
                             "source_row_id": "all_observations", "transform_version": self.transform_version, "date_grain": "static", "scope": "dataset_product_identity",
                             "product_grain": "product", "product_id": pid, "product_name": r.product_name, "version_seq": seq,
                             "valid_from": r.first, "valid_to": r.last, "observation_count": r.n, "source_tables": r.tables, "source_files": r.files,
                             "pack_spec_tokens": "|".join(pack_spec_tokens(r.product_name)), "temporal_relation": temporal, "name_change_class": change,
                             "validation_flags": "product_identity_versioned" if len(group) > 1 else ""})
        return pd.DataFrame(rows)

    def reflag(self, versioned_ids, label_variant_ids):
        """Classify every member of a repeated logical key and mark versioned products."""
        import pyarrow.parquet as pq
        summary = {}
        for table in list(self.writers):
            path = self.processed / f"canonical_{table}.parquet"
            repeated = {h for h, n in self.key_counts[table].items() if n > 1}
            version_ids = versioned_ids if table in IDENTITY_TABLES | {"product_master"} else set()
            label_ids = label_variant_ids if table == "product_master" else set()
            post = self.post_flags.get(table, [])
            if not (repeated or version_ids or label_ids or post):
                continue
            df = pd.read_parquet(path)
            classes = Counter()
            if repeated:
                hashes = key_hashes(df, table)
                member = hashes.isin(repeated)
                if table in self.bulk_key_tables:
                    # Bulk-checked tables were written unflagged; mark every occurrence after the first, as write() does.
                    add_flag(df, member & hashes.duplicated(keep="first"), "duplicate_key")
                sub = df.loc[member].assign(_h=hashes[member])
                numeric_cols = [c for c in df.columns if c in NUMERIC and c != "source_record_count"]
                signature = sub[numeric_cols].astype("string").fillna("<NA>").agg("|".join, axis=1) if numeric_cols else pd.Series("", index=sub.index)
                groups = sub.groupby("_h")
                n_files = groups["source_file"].nunique()
                n_values = signature.groupby(sub["_h"]).nunique()
                missing_sub = pd.Series(False, index=sub.index)
                if "product_variant_id" in sub:
                    files_with_variant = set(df.loc[df.product_variant_id.notna(), "source_file"])
                    missing_sub = sub.product_variant_id.isna() & sub.source_file.isin(files_with_variant)
                missing_sub = missing_sub.groupby(sub["_h"]).any()
                label = pd.Series("repeated_key_unexplained", index=n_files.index)
                label[n_files.gt(1) & n_values.le(1)] = "cross_release_repeat"
                label[n_files.gt(1) & n_values.gt(1)] = "cross_release_conflict"
                label[missing_sub] = "repeated_key_missing_subkey"
                row_label = sub["_h"].map(label)
                for name in row_label.unique():
                    add_flag(df, row_label.reindex(df.index).eq(name), name)
                classes.update(row_label.tolist())
            if version_ids:
                mask = df.product_id.isin(version_ids)
                add_flag(df, mask, "product_identity_versioned")
                classes["product_identity_versioned"] += int(mask.sum())
            for source_file, record_ids, flag in post:
                wanted = {str(i) for i in record_ids}
                rows = df.source_file.eq(source_file)
                mask = rows & df.source_row_id.where(rows, "").astype(str).str.split("|").map(lambda ids: not wanted.isdisjoint(ids))
                add_flag(df, mask, flag)
                classes[flag] += int(mask.sum())
            if label_ids:
                mask = df.product_id.isin(label_ids)
                add_flag(df, mask, "product_label_variants")
                classes["product_label_variants"] += int(mask.sum())
            finalize_quality(df)
            pq.write_table(self.pa.Table.from_pandas(df, schema=parquet_schema(self.pa, df.columns), preserve_index=False), path, compression="zstd")
            summary[table] = {name: n for name, n in classes.items() if n}
        return summary

    def finalize_bulk_keys(self):
        """Exact key uniqueness of bulk tables from one 64-bit hash per row."""
        for table, parts in self.bulk_hashes.items():
            hashes = np.concatenate(parts) if parts else np.array([], dtype="uint64")
            unique, counts = np.unique(hashes, return_counts=True)
            repeated = counts > 1
            self.key_counts[table] = Counter({int(h): int(n) for h, n in zip(unique[repeated], counts[repeated])})
            self.checks.setdefault("key_uniqueness", {})[table] = {
                "rows": int(len(hashes)), "distinct_keys": int(len(unique)), "keys_repeated": int(repeated.sum()),
                "key": list(KEYS[table]), "method": "64-bit hash of the string-normalised canonical key over every written row; repeats are flagged and classified in the reflag pass"}
        self.bulk_hashes.clear()

    def finish(self):
        self.finalize_bulk_keys()
        master_ids = {}
        identity = self.identity_summary()
        names_per_id = identity.groupby("product_id").product_name.nunique() if not identity.empty else pd.Series(dtype=int)
        dated_names = identity[identity.dated].groupby("product_id").product_name.nunique() if not identity.empty else pd.Series(dtype=int)
        versioned_ids = set(dated_names[dated_names.gt(1)].index)
        label_variant_ids = set(names_per_id[names_per_id.gt(1)].index) - versioned_ids
        locations = pd.concat(self.location_parts).drop_duplicates() if self.location_parts else pd.DataFrame(columns=["location_id", "location_name"])
        conflict_counts = {"product_master": int(names_per_id.gt(1).sum()),
                           "location_master": int(locations.groupby("location_id").location_name.nunique().gt(1).sum()) if not locations.empty else 0}
        for table, parts in self.master_parts.items():
            if not parts:
                continue
            data = pd.concat(parts, ignore_index=True)
            keys = [c for c in KEYS[table] if c in data]
            name_col = "product_name" if table == "product_master" else "location_name"
            if name_col in data:
                self.checks[table + "_attribute_conflicts"] = {"conflicting_name_keys": conflict_counts[table],
                    "method": "exact: every distinct (id, name) pair across all canonical fact rows (1.0.0 compared only the first row per 50k chunk)",
                    "policy": "First observed label retained on the master; fact rows keep their own labels; dated conflicts are versioned in product_identity_version."}
            data = data.drop_duplicates(keys, keep="first")
            idcol = "product_id" if table == "product_master" else "location_id"
            master_ids[table] = set(data[idcol].dropna().astype(str))
            self.write(table, canonicalize(data, table), masters=False)
            self.mappings.append({"table": table, "operation": "distinct observed identifiers; first-source lineage preserved; not an authoritative cross-dataset master", "fields": {field: {"coverage": "AGGREGATED" if field in data and data[field].notna().any() else "UNAVAILABLE", "source_columns": [field] if field in data else [], "formula": "distinct source identifier, first observed attribute", "unit": None, "assumptions": "Dataset-local identity only"} for field in (*COMMON, *TABLE_FIELDS[table])}})
        versions = self.identity_versions(identity)
        if not versions.empty:
            self.write("product_identity_version", canonicalize(versions, "product_identity_version"), masters=False)
            self.mappings.append({"table": "product_identity_version", "operation": "one row per observed (product_id, product_name); ordered by first observation; product_id never renumbered",
                                  "fields": {"valid_from": "min period_start/date of rows with that name", "valid_to": "max period_end/date", "pack_spec_tokens": "regex quantity+unit tokens printed in the name",
                                             "temporal_relation": "SEQUENTIAL if name periods never overlap, CONCURRENT otherwise", "name_change_class": "PACK_SPEC_CHANGED if printed pack/grade tokens differ, else LABEL_CHANGED_SAME_PACK_TOKENS"}})
            multi = versions[versions.version_seq.gt(1)].product_id.unique()
            classes = versions[versions.product_id.isin(multi)].drop_duplicates("product_id")
            self.checks["product_identity"] = {"products": int(versions.product_id.nunique()), "versioned_products": len(multi),
                "temporal_relation": classes.temporal_relation.value_counts().to_dict(), "name_change_class": classes.name_change_class.value_counts().to_dict(),
                "versioned_product_ids": sorted(multi.tolist())}
        for writer in self.writers.values():
            writer.close()
        self.checks["repeated_key_classification"] = self.reflag(versioned_ids, label_variant_ids)
        if label_variant_ids:
            self.checks["product_label_variants"] = {"products": len(label_variant_ids), "policy": "Undated label variants of one identifier (e.g. measurement contexts); master keeps the first label and is flagged."}
        for table, references in self.entity_references.items():
            if table not in master_ids and table in self.writers:
                col = "product_id" if table == "product_master" else "location_id"
                master_ids[table] = set(pd.read_parquet(self.processed / f"canonical_{table}.parquet", columns=[col])[col].dropna().astype(str))
            self.checks[table + "_mapping"] = {"observed_references": len(references), "unmatched_identifiers": sorted(references - master_ids.get(table, set()))}
        for table in self.writers:
            stats = flag_statistics(self.processed / f"canonical_{table}.parquet")
            self.flags[table], self.quality[table] = stats["flags"], stats["quality"]
            for col in FLAG_COLUMNS:
                self.nonnull[table][col] = stats["nonnull"][col]
        self.checks["unit_consistency"] = {table: {"observed_units": sorted(units), "unit_status": dict(self.unit_status[table]), "unknown_unit_rows": self.counts[table] - self.nonnull[table]["unit"], "policy": "Never combine quantities across units or unknown-unit source scopes"} for table, units in self.units.items()}
        report = {"dataset": self.dataset, "transform_version": self.transform_version, "row_counts": dict(self.counts), "semantic_checks": self.checks,
            "date_ranges": self.date_ranges, "flags": {k: dict(v) for k, v in self.flags.items()}, "quality": self.quality,
            "field_coverage": {t: {c: "MISSING" if n == 0 else "FULL" if n == self.counts[t] else "PARTIAL" for c,n in self.nonnull[t].items()} for t in self.counts},
            "null_counts": {t: {c: self.counts[t]-n for c,n in self.nonnull[t].items()} for t in self.counts},
            "quantity_conservation": self.conservation,
            "all_quantity_checks_passed": all(c["passed"] for c in self.conservation),
            "duplicate_policy": "Every source record stays traceable. Records collapse only under a documented dataset policy (source_record_count, full source_row_id list). Remaining repeated keys keep every row; all members are classified (cross_release_conflict/cross_release_repeat/repeated_key_missing_subkey/repeated_key_unexplained) and none is summed or picked.",
            "quarantine_policy": "Rows with invalid_date/negative/invalid_numeric flags remain traceable in canonical with analysis_eligible=false. No correction or deletion.",
            "notes": self.notes, "selected_sources": self.sources}
        mapping = {"schema": schema_document(), "dataset": self.dataset, "unit_evidence": self.unit_evidence, "mappings": self.mappings,
                   "scope": "Selected usable full tables/numeric subset, not a claim that all collected files are converted", "notes": self.notes}
        for name, obj in [("canonical_mapping.json", mapping), ("data_quality_report.json", report)]:
            (self.results / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
        algorithm_coverage(self.dataset, self.present).to_csv(self.results / "algorithm_coverage.csv", index=False, encoding="utf-8-sig")
        return report


def source_inventory(exporter):
    """Use the existing verified manifest for all sources; never scan 100 GB."""
    path = exporter.results / "raw_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    inventory = {"dataset": exporter.dataset, "manifest": str(path.relative_to(exporter.root)),
        "status": manifest.get("download_status"), "notes": manifest.get("notes"),
        "files": manifest.get("files", []), "measurement_semantics": {
            "missing_units": "Remain NULL; quantity field name is not evidence of items/kg",
            "sales_vs_outbound": "Never equated", "purchase_request_vs_receipt": "Never equated",
            "capacity": "Area/observed volume is not vehicle or storage capacity"}}
    (exporter.results / "canonical_source_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")


def stream_file(e, path, adapter, family, collapse=None):
    """collapse=(keys or None for all non-measure columns, measures): whole-file record collapse."""
    source = path.relative_to(e.root).as_posix()
    rows, columns, dtypes, dates = 0, [], {}, []
    source_monotonic, previous_date = True, None
    mapped_once = False
    if collapse:
        full = pd.concat(list(csv_chunks(path)))
        keys, collapse_measures = collapse
        keys = keys or [c for c in full.columns if c not in collapse_measures]
        batches = [(collapse_source_records(full, keys, collapse_measures), full)]
    else:
        batches = ((raw, raw) for raw in csv_chunks(path))
    canonical_rows = 0
    for raw, original in batches:
        result = adapter(raw, source, family)
        table, frame, measures = result[:3]
        work = result[3] if len(result) == 4 else raw
        # Conservation always compares against the uncollapsed source records.
        reference = original.rename(columns={c: c.upper() for c in original.columns if c not in work and c.upper() in work})
        e.write(table, frame, reference, measures)
        rows += len(original)
        canonical_rows += len(frame)
        columns = list(original.columns)
        dtypes = {c: ("numeric text" if pd.to_numeric(original[c].dropna(), errors="coerce").notna().all() else "string") for c in original}
        date_col = "snapshot_date" if table == "inventory_snapshot" else "date"
        if date_col in frame:
            dates.extend(frame[date_col].dropna().agg(["min", "max"]).dropna().tolist())
            valid_dates = frame[date_col].dropna()
            if not valid_dates.empty:
                source_monotonic = source_monotonic and bool(valid_dates.is_monotonic_increasing) and (previous_date is None or previous_date <= valid_dates.iloc[0])
                previous_date = valid_dates.iloc[-1]
        if not mapped_once:
            # Recover exact renames by comparing raw columns; explicit measures win.
            mapping = dict(measures)
            for target in frame:
                if target in mapping or target in COMMON or target in {"date", "snapshot_date"}:
                    continue
                for origin in work:
                    if frame[target].reset_index(drop=True).astype("string").equals(work[origin].reset_index(drop=True).astype("string")) and frame[target].notna().any():
                        mapping[target] = origin
                        break
            datesource = next((c for c in work if c.upper() in {"BASE_YMD", "CFMTN_YMD", "CFMTN_YM", "기준일자"}), "BASE_YR+BASE_MM")
            derived = {c: {"source_columns": [datesource], "formula": "strict calendar parse; month -> first/last day, retain monthly grain", "unit": None, "assumptions": "No date correction"} for c in [date_col, "period_start", "period_end"]}
            if mapping.get("product_id") == "_category_id":
                derived["product_id"] = {"source_columns": ["FDMT_PDLT_LGLS_CODE"], "formula": "'category:' + source category code", "unit": None, "assumptions": "Namespace only; category is not a SKU"}
            if mapping.get("product_variant_id") == "_barcode":
                derived["product_variant_id"] = {"source_columns": ["BRCD_INFO"], "formula": "BRCD_INFO; literal 'NULL' and empty are missing markers", "unit": None, "assumptions": "Official grain is 상품 및 바코드별; barcode variants are never summed"}
            if collapse:
                derived["source_row_id"] = {"source_columns": ["(record position)"], "formula": "'|'.join of every collapsed 1-based record id", "unit": None, "assumptions": "Lossless lineage of collapsed records"}
                derived["source_record_count"] = {"source_columns": ["(record position)"], "formula": "number of source records sharing the complete key", "unit": None, "assumptions": "Records with NULL key or NULL/invalid/negative measure are never merged"}
            mapping["raw_date"] = datesource
            for name in ("validation_flags", "quality_flags", "quality_status", "analysis_eligible"):
                derived[name] = {"source_columns": list(work.columns), "formula": "canonical schema validation", "unit": None, "assumptions": "Flags do not repair observations"}
            unit = frame.unit.dropna().iloc[0] if frame.unit.notna().any() else None
            e.record_mapping(table, mapping, source, measures, str(frame.date_grain.iloc[0]), str(frame.scope.iloc[0]), unit=unit, derived=derived,
                             unit_status=str(frame.unit_status.mode().iloc[0]))
            mapped_once = True
    e.sources.append({"file": source, "sheet_or_table": "csv", "row_count": rows, "canonical_rows": canonical_rows, "columns": columns, "sample_dtypes": dtypes,
                      "date_range": [min(dates), max(dates)] if dates else None, "source_date_monotonic": source_monotonic,
                      "encoding": csv_encoding(path), "status": "full selected file", "family": family,
                      "collapse_policy": None if not collapse else {"keys": keys, "measures": list(collapse[1]), "rule": "sum measures of records sharing a complete key; NULL key or NULL/invalid/negative measure never merged"}})


def run_suhyup(e):
    for path in sorted((e.folder / "raw").glob("*.CSV")):
        stream_file(e, path, adapt_suhyup, "flow" if "입출고" in path.name else "stock")
    from services.suhyup_algorithm_revalidation import enrich_inventory_constraints
    legacy = pd.read_csv(e.folder / "processed/suhyup_logistics_inventory_flow_actual.csv", dtype={"date": str, "center_code": str, "product_code": str})
    path = e.root / "16_VARO_E2E_20260731/multi_snapshot_validation/suhyup_202607_multi_snapshot_recommendations.csv"
    raw = pd.read_csv(path, dtype={"snapshot_date": str, "route_id": str, "product_id": str, "source_id": str, "target_id": str})
    enriched = enrich_inventory_constraints(raw, legacy)
    raw_stock_path = next(p for p in (e.folder / "raw").glob("*.CSV") if "재고현황" in p.name)
    stock_ids = pd.concat(list(csv_chunks(raw_stock_path, columns=["물류센터-공판장 코드", "수산물품목코드"])), ignore_index=True)
    location_crosswalk = verified_identifier_crosswalk(pd.concat([raw.source_id, raw.target_id]), stock_ids["물류센터-공판장 코드"])
    product_crosswalk = verified_identifier_crosswalk(raw.product_id, stock_ids["수산물품목코드"])
    mapping = {"product_id": "product_id", "product_name": "product_name", "source_id": "source_id", "target_id": "target_id", "recommended_qty": "recommended_qty", "distance_km": "distance_km", "travel_time_min": "travel_time_min", "transport_cost": "move_cost", "route_id": "route_id", "route_type": "route_type", "source_surplus": "source_surplus", "target_need": "target_need_7d", "expected_saving": "expected_saving"}
    frame = mapped(enriched, "suhyup", path.relative_to(e.root), mapping, date_field="snapshot_date")
    frame["source_id"] = frame.source_id.map(location_crosswalk)
    frame["target_id"] = frame.target_id.map(location_crosswalk)
    frame["product_id"] = frame.product_id.map(product_crosswalk)
    frame["record_kind"] = "existing_benchmark_candidate"
    route_units = UNIT_EVIDENCE["suhyup"]["benchmark_route"]
    frame["currency"], frame["distance_unit"], frame["time_unit"] = route_units["currency"], route_units["distance"], route_units["time"]
    frame["currency_status"] = frame["distance_unit_status"] = frame["time_unit_status"] = route_units["status"]
    frame["validation_flags"] = "benchmark_proxy_constraints|unit_unspecified"
    measures = {k:v for k,v in mapping.items() if k in NUMERIC}
    e.write("transfer_network", canonicalize(frame, "transfer_network"), enriched, measures)
    deriv = {
        "source_surplus": {"source_columns": ["stock_qty", "center_code", "product_code", "date"], "formula": "existing enrich_inventory_constraints: max(0, source stock - same-day selected-network product median)", "unit": None, "assumptions": "Existing benchmark proxy, not observed transferable stock"},
        "target_need": {"source_columns": ["stock_qty", "outbound_qty"], "formula": "existing enrich_inventory_constraints: max(0, median stock + 7 * daily outbound - target stock)", "unit": None, "assumptions": "Existing benchmark demand proxy, not actual retail unmet demand"},
        "transport_cost": {"source_columns": ["move_cost"], "formula": "identity from existing enriched route benchmark", "unit": "KRW", "unit_status": "DERIVED", "assumptions": "Reference transport engine estimate; not invoiced cost"}}
    for field, crosswalk in [("source_id", location_crosswalk), ("target_id", location_crosswalk), ("product_id", product_crosswalk)]:
        deriv[field] = {"source_columns": [field], "formula": "verified one-to-one raw code crosswalk", "crosswalk": crosswalk, "unit": None, "assumptions": "Existing benchmark removed leading zeroes; restore only actual observed raw identifiers"}
    e.record_mapping("transfer_network", mapping, str(path.relative_to(e.root)), measures, "daily", "location_product", derived=deriv)
    e.checks["unit_evidence"] = UNIT_EVIDENCE["suhyup"]
    e.notes += ["Raw count unit unspecified (unit_status UNKNOWN); parallel kg observations retained separately with weight_unit kg DIRECT.", "State-processing code is part of stock/flow key; never collapse it.", "No unit_weight inferred from stock_kg/stock_qty; packaging mix is unknown.", "Transfer candidates and proxy constraints are existing benchmark-derived records, not observed movements."]


def logisall_repeat_evidence(folder):
    """Row-level evidence for the LogisAll repeated-key and missing-destination policies."""
    def load(path):
        return pd.concat(list(csv_chunks(path)))
    releases, missing, dates, flags = [], [], {}, []
    for sales_path in sorted(folder.glob("TB_MRI_SLE_CW*.csv")):
        release = sales_path.stem.split("-", 1)[1]
        dsbn_path = folder / f"TB_MIR_DSBN_CW-{release}.csv"
        sales, dsbn = load(sales_path), load(dsbn_path)
        by_zone = sales.groupby(["BASE_YMD", "AGFD_PDLT_NM", "AGFD_SLPL_ZIP"], dropna=False).size().sort_index()
        by_sender = dsbn.groupby(["BASE_YMD", "AGFD_PDLT_NM", "FRWAR_ZIP"], dropna=False).size().sort_index()
        paired = bool(by_zone.index.equals(by_sender.index) and (by_zone.values == by_sender.values).all())
        identical = sales.duplicated(keep=False)
        adjacent = {"demand_series": (sales_path, adjacent_identical_positions(sales)), "transfer_network": (dsbn_path, adjacent_identical_positions(dsbn))}
        for table, (path, ids) in adjacent.items():
            if ids:
                flags.append((table, path.relative_to(folder.parents[2]).as_posix(), ids))
        releases.append({"release": release, "sales_rows": len(sales), "distribution_rows": len(dsbn),
            "sales_rows_in_repeated_zone_keys": int(sales.duplicated(["BASE_YMD", "AGFD_PDLT_NM", "AGFD_SLPL_ZIP"], keep=False).sum()),
            "distribution_rows_in_repeated_keys": int(dsbn.duplicated(["BASE_YMD", "AGFD_PDLT_NM", "FRWAR_ZIP", "ARVL_ZIP"], keep=False).sum()),
            "fully_identical_sales_rows": int(identical.sum()), "adjacent_identical_sales_rows": len(adjacent["demand_series"][1]),
            "adjacent_identical_distribution_rows": len(adjacent["transfer_network"][1]),
            "adjacent_identical_sales_qty_share": float(pd.to_numeric(sales.SLE_QY).iloc[[i - 1 for i in adjacent["demand_series"][1]]].sum() / pd.to_numeric(sales.SLE_QY).sum()),
            "adjacent_identical_distribution_qty_share": float(pd.to_numeric(dsbn.DSBN_QY).iloc[[i - 1 for i in adjacent["transfer_network"][1]]].sum() / pd.to_numeric(dsbn.DSBN_QY).sum()),
            "zones": sorted(set(sales.AGFD_SLPL_ZIP.dropna()) | set(dsbn.FRWAR_ZIP.dropna()) | set(dsbn.ARVL_ZIP.dropna())),
            "sales_zone_multiplicity_equals_distribution_sender_multiplicity": paired})
        dates[release] = set(sales.BASE_YMD.dropna())
        for idx, row in dsbn[dsbn.ARVL_ZIP.isna()].iterrows():
            same = dsbn[(dsbn.BASE_YMD == row.BASE_YMD) & (dsbn.AGFD_PDLT_NM == row.AGFD_PDLT_NM) & (dsbn.FRWAR_ZIP == row.FRWAR_ZIP)]
            missing.append({"source_file": dsbn_path.relative_to(folder.parents[2]).as_posix(), "source_row_id": str(idx + 1), "date": row.BASE_YMD,
                            "product": row.AGFD_PDLT_NM, "source_zone": row.FRWAR_ZIP, "shipment_qty": row.DSBN_QY, "raw_arrival_field": "empty",
                            "other_arrival_zones_same_date_product_sender": sorted(same.ARVL_ZIP.dropna().unique().tolist()),
                            "restoration": "NOT_RESTORED: no source field or location master determines the arrival zone; sales zone pairs with the sender, not the destination",
                            "canonical": "target_id NULL, missing:target_id (MISSING_KEY), analysis_eligible=false, excluded from transfer-network use"})
    names = list(dates)
    overlap = sum(len(dates[a] & dates[b]) for i, a in enumerate(names) for b in names[i + 1:])
    return {"releases": releases, "cross_release_date_overlap": overlap,
            "conclusion": "Repeated (date, product, zone) keys are distinct sub-zone records masked to a first-digit ZIP zone, not duplicate rows: the multiplicity is identical in the paired sales/distribution extracts of every release and releases never overlap in date. Policy: sum per complete key with full source_row_id lineage.",
            "residual_uncertainty": "Fully identical rows cannot be proven distinct individually. Records identical to the record directly above occur only where listed per release; they are scattered (no repeated blocks), kept and summed, and every canonical row containing one carries adjacent_identical_record (DUPLICATE_OBSERVATION, informational) so the uncertainty stays visible."}, missing, flags


def run_logisall(e):
    folder = e.folder / "raw/kadx_full_data"
    for prefix, family in [("TB_MI_STC", "stock"), ("TB_MRI_SLE_CW", "sales"), ("TB_MIR_DSBN_CW", "network")]:
        for path in sorted(folder.glob(prefix + "*.csv")):
            stream_file(e, path, adapt_logisall, family, collapse=LOGISALL_COLLAPSE.get(family))
    e.checks["repeated_key_investigation"], e.checks["missing_destination"], adjacent = logisall_repeat_evidence(folder)
    for table, source_file, ids in adjacent:
        e.post_flags[table].append((source_file, ids, "adjacent_identical_record"))
    e.checks["unit_evidence"] = UNIT_EVIDENCE["logisall"]
    e.notes += ["Full received monthly national stock, regional daily sales, regional distribution files used; public sample files excluded.", "Regional stock and regional inbound/outbound full products were not acquired; samples are not production evidence.", "ZIP is masked to a first-digit zone (00000..60000), not a warehouse or precise geolocation. Product names are dataset-local keys.", "Repeated zone keys are sub-zone records; their quantities are summed per key with every source record id retained (source_record_count).", "National monthly stock cannot be joined to regional daily sales as same-grain inventory/demand.", "Production quantity PDTN_QY is not storage; production tables excluded.", "Quantity units are absent from the acquired specifications; unit NULL, unit_status UNKNOWN."]


def jangbogo_negative_evidence(folder, in_scope="TB_PDLT_BUY_YM_231010.csv"):
    """All negative purchase-request rows across every BUY partition, plus in-scope code history."""
    def load(path):
        data = pd.concat(list(csv_chunks(path)))
        data.columns = data.columns.str.upper()
        return data, pd.to_numeric(data.PRCA_DMND_QYT, errors="coerce")

    scope_data, scope_qty = load(folder / in_scope)
    codes = set(scope_data.loc[scope_qty < 0, "FDMT_PDLT_CODE"])
    partitions, history, scope_rows = [], [], []
    for path in sorted(folder.glob("TB_PDLT_BUY_YM*.csv")):
        data, qty = (scope_data, scope_qty) if path.name == in_scope else load(path)
        rows = [{"source_row_id": str(i + 1), "month": r.CFMTN_YM, "product_id": r.FDMT_PDLT_CODE, "product_name": r.FDMT_PDLT_NM,
                 "barcode_raw": r.BRCD_INFO, "order_qty": float(qty[i])} for i, r in data[qty < 0].iterrows()]
        partitions.append({"file": path.name, "months": [data.CFMTN_YM.min(), data.CFMTN_YM.max()], "rows": len(data),
                           "zero_rows": int(qty.eq(0).sum()), "negative_rows": rows})
        scope_rows = rows if path.name == in_scope else scope_rows
        hit = data[data.FDMT_PDLT_CODE.isin(codes)]
        history += [{"file": path.name, "month": r.CFMTN_YM, "product_id": r.FDMT_PDLT_CODE, "order_qty": float(q)} for r, q in zip(hit.itertuples(), qty[hit.index])]
    return {"in_scope_file": in_scope, "in_scope_negative_rows": scope_rows,
            "all_partitions": partitions, "history_of_in_scope_negative_codes": history,
            "specification": "PRCA_DMND_QYT = 구매요청수량 only; no direction, return, cancellation or adjustment column exists (KADX '농산물 매입 기간별 데이터').",
            "classification": "UNDOCUMENTED_SIGNED_REQUEST: negative net purchase-request quantity. Not provably a return/cancellation (no such field, and no prior positive request of that magnitude exists for these codes), not provably an error. Retained unchanged; negative:order_qty (NEGATIVE_QUANTITY), analysis_eligible=false; never abs() or zero.",
            "not_populated": "transaction_direction/adjustment_flag/returns_qty are not derived because no source field supports them."}


def run_jangbogo(e):
    folder = e.folder / "raw/kadx_full_data"
    for prefix, family in [("TB_SPOT_CL_MTH_BUY_YM", "orders"), ("TB_SPOT_CL_MTH_SALES_YM", "sales")]:
        for path in sorted(folder.glob(prefix + "*.csv")):
            stream_file(e, path, adapt_jangbogo, family)
    for name in ["TB_LGTC_PDLT_SALES_YMD_20230301.csv", "TB_LGTC_PDLT_SALES_YMD_20231010.csv"]:
        # Occurrence rows carry no quantity; fully identical rows collapse to one with a record count.
        stream_file(e, folder / name, adapt_jangbogo, "observation", collapse=(None, []))
    stream_file(e, folder / "TB_PDLT_BUY_YM_231010.csv", adapt_jangbogo, "orders")
    e.checks["negative_quantity_investigation"] = jangbogo_negative_evidence(folder)
    ids = pd.concat([p[["product_id"]] for p in e.identity_parts]).product_id.drop_duplicates() if e.identity_parts else pd.Series(dtype=str)
    float_spelled = sorted(i for i in ids if re.fullmatch(r"category:\d+\.0", str(i)))
    e.checks["identifier_spelling"] = {"float_spelled_category_ids": float_spelled,
        "with_integer_counterpart": sorted(i for i in float_spelled if i[:-2] in set(ids)),
        "policy": "Raw spelling kept (e.g. category:1001.0 in 2021-01 releases vs category:1001 later); no crosswalk applied in this release."}
    e.checks["unit_evidence"] = UNIT_EVIDENCE["jangbogo"]
    e.notes += ["Bounded first adapter scope: all small warehouse/category monthly sales+purchase tables; two warehouse/SKU observation releases; latest full SKU purchase-request partition. Large transaction tables are catalogued but not converted.", "PRCA_DMND_QYT is purchase-request quantity (order_qty), never inbound or realized demand.", "Purchase-request grain is month x product x barcode (BRCD_INFO -> product_variant_id); barcode variants are not summed. Repeats with a missing barcode are flagged repeated_key_missing_subkey.", "Warehouse/SKU table has no movement quantity; fully identical occurrence rows collapse to one row with source_record_count and every source row id.", "Product-level purchase-request partition has no warehouse key, so no fabricated warehouse join.", "Category: prefixed identifiers cannot join SKU codes. No stock snapshot exists in selected tables.", "Overlapping category-sales releases with different values are all flagged cross_release_conflict; none is picked or summed.", "Invalid calendar strings become NULL dates with raw_date retained and analysis_eligible=false."]


def run_nfqs(e):
    signatures, product_sets, period_products = {}, {}, {}
    identity_rows, region_rows, negatives, closes = [], 0, [], {}
    region_mismatch, product_sum_mismatch = [], []
    for path in sorted((e.folder / "raw").glob("nfqs_fish_inventory_*.json")):
        year, q = path.stem.split("_")[-2:]
        if not 2020 <= int(year) <= 2025:
            continue
        raw = json.loads(path.read_text(encoding="utf-8-sig"))["LIST"]
        source = path.relative_to(e.root).as_posix()
        period = f"{year}{q}"
        frame = adapt_nfqs(raw, source, year, q[1:])
        e.write("inventory_snapshot", frame)
        source_values = [row.get(k) for row in raw for k in [*REGIONS, "INVENTOTAL"]]
        before = pd.to_numeric(pd.Series(source_values), errors="coerce")
        after = frame.inventory_qty
        e.conservation.append({"source_file": source, "field": "inventory_qty", "table": "inventory_snapshot", "source_sum": float(before.sum()), "canonical_sum": float(after.sum()), "source_nonnull": int(before.notna().sum()), "canonical_nonnull": int(after.notna().sum()), "passed": math.isclose(float(before.sum()), float(after.sum()), abs_tol=1e-7, rel_tol=1e-12) and int(before.notna().sum()) == int(after.notna().sum())})
        signatures[path.name] = sorted(set().union(*(set(row) for row in raw)))
        product_sets[path.name] = sorted(row.get("ICEGDFG") for row in raw)
        period_products[period] = {row.get("ICEGDFG"): row.get("CODEKNM") for row in raw}
        for index, row in enumerate(raw):
            diff = nfqs_flow_identity_diff(row)
            identity_rows.append({"period": period, "record": index + 1, "product_id": row["ICEGDFG"], "product_name": row["CODEKNM"],
                                  "ICEALIST": row["ICEALIST"], "ICEAIN": row["ICEAIN"], "ICEAOUT": row["ICEAOUT"], "INVENTOTAL": row["INVENTOTAL"], "difference_ton": diff})
            gap = sum(float(row[k]) for k in REGIONS) - float(row["INVENTOTAL"])
            region_rows += 1
            if abs(gap) > NFQS_IDENTITY_TOLERANCE_TON:
                region_mismatch.append({"period": period, "product_id": row["ICEGDFG"], "difference_ton": gap})
            for field in [*REGIONS, "INVENTOTAL", "ICEALIST", "ICEAIN", "ICEAOUT"]:
                if float(row[field]) < 0:
                    negatives.append({"period": period, "record": index + 1, "product_id": row["ICEGDFG"], "product_name": row["CODEKNM"], "field": field, "raw_value": repr(row[field]),
                                      "within_residue_tolerance": abs(float(row[field])) < NFQS_FLOAT_RESIDUE_TOLERANCE_TON})
            closes[(period, row["ICEGDFG"])] = (float(row["INVENTOTAL"]), float(row["ICEALIST"]))
        total = next((r for r in raw if r.get("ICEGDFG") == "999"), None)
        if total:
            for field in [*REGIONS, "INVENTOTAL", "ICEALIST", "ICEAIN", "ICEAOUT"]:
                gap = sum(float(r[field]) for r in raw if r.get("ICEGDFG") != "999") - float(total[field])
                if abs(gap) > NFQS_IDENTITY_TOLERANCE_TON:
                    product_sum_mismatch.append({"period": period, "field": field, "difference_ton": gap})
        e.sources.append({"file": source, "sheet_or_table": "LIST", "row_count": len(raw), "canonical_rows": len(frame), "columns": signatures[path.name], "date_range": [str(frame.snapshot_date.min()), str(frame.snapshot_date.max())]})
        e.record_mapping("inventory_snapshot", {"product_id": "ICEGDFG", "product_name": "CODEKNM", "inventory_qty": "region field or INVENTOTAL"}, source, {"inventory_qty": "region field or INVENTOTAL"}, "quarterly", "region/national_total", "ton",
            {"snapshot_date": {"source_columns": ["query year", "query quarter"], "formula": "quarter end", "unit": None, "assumptions": "Query metadata, not daily observations"}, "location_id": {"source_columns": list(REGIONS) + ["INVENTOTAL"], "formula": "unpivot field name", "unit": None, "assumptions": "National total is separately scoped; never sum it with regions"}},
            unit_status=UNIT_EVIDENCE["nfqs"]["quantity"]["status"])
    identity = pd.DataFrame(identity_rows)
    broken = identity[identity.difference_ton.abs() > NFQS_IDENTITY_TOLERANCE_TON]
    periods = sorted(period_products)
    continuity = []
    for prev, cur in zip(periods, periods[1:]):
        for pid in period_products[cur]:
            if (prev, pid) in closes:
                continuity.append(closes[(cur, pid)][1] - closes[(prev, pid)][0])
    continuity = pd.Series(continuity, dtype=float)
    panel = nfqs_panel_coverage(period_products)
    panel.to_csv(e.results / "nfqs_panel_coverage.csv", index=False, encoding="utf-8-sig")
    reported = panel[panel.status.eq("REPORTED")]
    e.checks["quarter_structure"] = {"field_signatures": signatures, "product_sets": product_sets,
        "product_set_changed": len(set(tuple(v) for v in product_sets.values())) > 1,
        "region_sum_vs_reported_total_mismatches": len(region_mismatch)}
    e.checks["within_record_flow_identity"] = {
        "formula": "ICEALIST + ICEAIN - ICEAOUT = INVENTOTAL within the same quarterly record",
        "field_semantics": "SOURCE_METADATA + empirical: data.go.kr 15082985 describes 이월량, 입출하량, 재고량 per quarter; the official page does not display or define the hidden fields; the identity holding exactly in most records fixes the sign of each field.",
        "tolerance_ton": NFQS_IDENTITY_TOLERANCE_TON, "records_checked": len(identity), "mismatches": len(broken),
        "mismatch_rows": broken.to_dict("records"),
        "mismatch_periods": broken.period.value_counts().sort_index().to_dict(),
        "total_row_equals_sum_of_component_gaps": {p: math.isclose(float(g[g.product_id.ne("999")].difference_ton.sum()), float(g[g.product_id.eq("999")].difference_ton.sum()), abs_tol=1e-6)
                                                   for p, g in broken.groupby("period") if g.product_id.eq("999").any()},
        "classification": "SOURCE_ANOMALY reported by the source (not a validation artefact): same-record opening+in-out differs from closing stock; stock values themselves agree with regional sums. Flag source_flow_identity_mismatch on the national-total row (informational).",
        "replaces": "1.0.0 'prior_plus_in_minus_out_vs_current_mismatches' (misnamed: it was this same-record identity, not a cross-quarter flow)."}
    e.checks["cross_quarter_carryover_continuity"] = {
        "comparison": "opening ICEALIST(t) vs closing INVENTOTAL(t-1)", "pairs": int(len(continuity)),
        "equal_within_0.01_ton": int(continuity.abs().lt(0.01).sum()), "different": int(continuity.abs().ge(0.01).sum()),
        "max_abs_difference_ton": float(continuity.abs().max()) if len(continuity) else None,
        "interpretation": "Diagnostic only, not an identity: quarterly carry-over is re-reported by cooperating firms (composition/revision), so inventory(t)+in-out=inventory(t+1) is NOT assumed and no flow is imputed."}
    e.checks["product_components_vs_999_total"] = {"mismatches": product_sum_mismatch, "tolerance_ton": NFQS_IDENTITY_TOLERANCE_TON}
    e.checks["region_sum_vs_national_total"] = {"records": region_rows, "mismatches": region_mismatch}
    e.checks["negative_values"] = {"rows": negatives, "tolerance_ton": NFQS_FLOAT_RESIDUE_TOLERANCE_TON,
        "tolerance_basis": "Raw JSON reports at most 6 decimals of a ton (1e-6 t); tokens with 12-33 decimals are IEEE-754 accumulation residue in the server response. |x| < 1e-9 t is 1/1000 of the finest reported digit.",
        "policy": "Value preserved exactly; within tolerance -> float_residue_negative (SOURCE_ANOMALY, informational); beyond tolerance -> negative:inventory_qty (blocking). Never clamped."}
    e.checks["panel_coverage"] = {"periods": periods, "products": int(panel.product_id.nunique()),
        "balanced": bool(panel.status.eq("REPORTED").all()),
        "not_reported": panel[panel.status.eq("NOT_REPORTED")].groupby("product_id").period.apply(list).to_dict(),
        "reported_span": reported.groupby("product_id").agg(first=("first_reported", "first"), last=("last_reported", "first"), name=("product_name", "first")).to_dict("index"),
        "name_changes": panel[panel.name_variants.gt(1)].product_id.unique().tolist(),
        "special_codes": {"150": "기타 (residual category)", "999": "합계 (reported all-product total)"},
        "policy": "Unbalanced panel: an absent product is NOT_REPORTED, never zero inventory; no rows generated. Detail: results/nfqs_panel_coverage.csv"}
    e.checks["unit_evidence"] = UNIT_EVIDENCE["nfqs"]
    e.notes += ["24 quarterly snapshots, no daily interpolation. National totals and regional rows have distinct scope; summing both double-counts inventory.", "ICEGDFG=999 is a reported all-product total; product_grain=all_products and aggregate_product_total flag prevent treating it as an extra SKU.", "Only cooperating cold-storage firms; not all national inventory (official page notice).", "Unit ton confirmed on the official page ('재고량의 단위는 톤(t) 입니다.').", "Floating negative residues retained unchanged and flagged float_residue_negative; no zero-clipping.", "Hidden ICEALIST/ICEAIN/ICEAOUT are used only for the same-record identity check; they are not promoted to inventory_flow."]


def aihub_location_evidence(e, workbook_header, label_zips):
    site_words = re.compile(r"site|center|centre|warehouse|wrhs|창고|센터|거점|지점|date|일자|날짜", re.IGNORECASE)
    labels = {}
    for path in label_zips:
        keys, members = Counter(), 0
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".json"):
                    continue
                members += 1
                data = json.loads(archive.read(name).decode("utf-8-sig"))
                keys.update(f"info.{k}" for k in data.get("info", {}))
                keys.update(f"images.{k}" for image in data.get("images", []) for k in image)
                keys.update(f"attributes.{k}" for ann in data.get("annotations", []) for k in ann.get("attributes", {}))
        labels[path.name] = {"json_members": members, "field_counts": dict(keys), "site_or_date_fields": sorted(k for k in keys if site_words.search(k.split(".", 1)[1]))}
    return {"daily_flow_workbook_header": workbook_header, "workbook_site_or_date_columns": [h for h in workbook_header if h and site_words.search(str(h)) and str(h) != "일자"],
            "measurement_member_path_levels": ["물품측정데이터", "flow_dir", "category_l1", "category_l2", "kan_code_barcode.csv"],
            "label_archives": labels, "metadata_sites": 2,
            "conclusion": "No deterministic site/warehouse identifier exists for the daily flow rows, measurements or labels; the two metadata sites stay in location_master (scope named_sites_unlinked_to_flow) and are never joined. Flow rows keep location_id NULL (UNKNOWN_LOCATION); no UNKNOWN_LOCATION entity is created and nothing enters network algorithms."}


def run_aihub(e):
    path = next((e.folder / "raw").rglob("*물동량*.xlsx"))
    book = pd.ExcelFile(path)
    header = pd.read_excel(path, sheet_name=book.sheet_names[0], header=None, nrows=2)
    raw = pd.read_excel(path, sheet_name=book.sheet_names[0], header=None, skiprows=2)
    raw = raw.loc[raw[0].notna()].copy()
    source = path.relative_to(e.root).as_posix()
    for table, frame, measures, work in adapt_aihub(raw, source, book.sheet_names[0]):
        e.write(table, frame, work, measures)
        e.record_mapping(table, measures, source, measures, "daily", "unidentified_site_all_products", "item",
            {"date" if table == "inventory_flow" else "snapshot_date": {"source_columns": ["일자"], "formula": "Excel datetime -> ISO date", "assumptions": "Business dates only; missing days are not generated", "unit": None}},
            unit_status=UNIT_EVIDENCE["aihub"]["quantity"]["status"])
    inv = pd.to_numeric(raw[3], errors="coerce")
    diff = inv.diff() - pd.to_numeric(raw[1], errors="coerce") + pd.to_numeric(raw[2], errors="coerce")
    e.notes += [f"Inventory balance between observed rows: {int(diff.dropna().abs().lt(1e-8).sum())}/{int(diff.notna().sum())}; gaps are not imputed.", "Workbook has no site key or SKU key; NULL preserved. Metadata's two named sites cannot be joined to this time series.", "Inventory volume m3 is observed occupied space, not capacity. capacity_m3_implied is deliberately excluded.", "No image archives opened or copied; label JSON archives are read only for field names."]
    e.sources.append({"file": source, "sheet_or_table": book.sheet_names[0], "row_count": len(raw), "date_range": [str(raw[0].min()), str(raw[0].max())], "columns": ["일자", "입고물품", "출고물품", "재고", "size class counts", "occupied cm3", "occupied m3", "utilization ratio"]})
    metadata_path = next((e.folder / "raw").rglob("metadata_warehouse_information.json"))
    metadata = pd.DataFrame(json.loads(metadata_path.read_text(encoding="utf-8-sig"))["warehouse_info"])
    location_mapping = {"location_id": "site_name", "location_name": "site_name", "address": "site_address"}
    locations = mapped(metadata, "aihub", metadata_path.relative_to(e.root), location_mapping, grain="static", scope="named_sites_unlinked_to_flow", sheet="warehouse_info")
    e.write("location_master", canonicalize(locations, "location_master"), masters=False)
    e.record_mapping("location_master", location_mapping, str(metadata_path.relative_to(e.root)), {}, "static", "named_sites_unlinked_to_flow")
    # Reuse the already verified numeric extraction; trace each record to its
    # exact raw ZIP member. Each member is one measurement, not a product row.
    measurement_path = e.folder / "processed/aihub71861_item_measurements.csv"
    zip_path = next((e.folder / "raw").rglob("Other.zip"))
    count, measurements = 0, []
    for raw_measure in csv_chunks(measurement_path):
        frame, measures = adapt_aihub_measurements(raw_measure, zip_path.relative_to(e.root))
        e.write("product_measurement", frame, raw_measure, measures)
        measurements.append(raw_measure)
        count += len(raw_measure)
    e.record_mapping("product_measurement", {"product_id": "barcode", "product_name": "product_name", "category": "category_l2", "category_code": "kan_code", "length": "length_cm", "width": "width_cm", "height": "height_cm", "weight": "weight_kg"},
        str(measurement_path.relative_to(e.root)), measures, "static", "measured_item", unit=None,
        derived={"source_sheet_or_table": {"source_columns": ["source_member"], "formula": "exact member path in Other.zip", "unit": None, "assumptions": "Reuse verified numeric extraction"},
                 "measurement_context": {"source_columns": ["flow_dir"], "formula": str(AIHUB_MEASUREMENT_CONTEXT), "unit": None, "assumptions": "Member folder names the handling context"},
                 "dimension_unit": {"source_columns": ["label caption"], "formula": "cm", "unit": "cm", "unit_status": "SOURCE_METADATA", "assumptions": UNIT_EVIDENCE["aihub"]["measurement"]["evidence"]},
                 "weight_unit": {"source_columns": ["label caption"], "formula": "kg", "unit": "kg", "unit_status": "SOURCE_METADATA", "assumptions": UNIT_EVIDENCE["aihub"]["measurement"]["evidence"]}})
    for field, unit in [("length", "cm"), ("width", "cm"), ("height", "cm"), ("weight", "kg")]:
        e.mappings[-1]["fields"][field].update(unit=unit, unit_status="SOURCE_METADATA")
    e.sources.append({"file": str(measurement_path.relative_to(e.root)), "raw_archive": str(zip_path.relative_to(e.root)), "row_count": count, "status": "existing verified numeric extraction with per-row raw member lineage"})
    m = pd.concat(measurements)
    repeated = m[m.barcode.duplicated(keep=False)]
    groups = repeated.groupby("barcode")
    dims = ["length_cm", "width_cm", "height_cm", "weight_kg"]
    e.checks["repeated_measurement_investigation"] = {
        "measurement_rows": len(m), "unique_barcodes": int(m.barcode.nunique()), "repeated_barcodes": int(groups.ngroups),
        "group_sizes": groups.size().value_counts().to_dict(),
        "flow_dir_combinations": groups.flow_dir.apply(lambda s: "|".join(sorted(s))).value_counts().to_dict(),
        "groups_with_identical_dimensions_and_weight": int((groups[dims].nunique().max(axis=1) == 1).sum()),
        "groups_with_identical_product_name": int((groups.product_name.nunique() == 1).sum()),
        "groups_with_identical_category_l2": int((groups.category_l2.nunique() == 1).sum()),
        "conclusion": "Each repeated barcode was measured once as an inbound item and once as an outbound item with different dimensions/weight: repeated measurements in two handling contexts, not label duplication. Measurements live in product_measurement; product_master has one row per barcode without choosing a weight."}
    label_zips = [p for name in ("TL.zip", "VL.zip") for p in (e.folder / "raw").rglob(name)]
    e.checks["location_identification"] = aihub_location_evidence(e, [str(v) for v in header.iloc[0].tolist()], label_zips)
    e.checks["unit_evidence"] = UNIT_EVIDENCE["aihub"]


def runner(dataset):
    if dataset in EXTERNAL_DATASETS:
        # Imported lazily: the external pipeline builds on this module's Exporter.
        from services import external_canonical_pipeline
        return getattr(external_canonical_pipeline, "run_" + dataset)
    return globals()["run_" + dataset]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    results = {}
    for dataset in args.datasets:
        exporter = Exporter(args.data_root, dataset)
        source_inventory(exporter)
        try:
            runner(dataset)(exporter)
            report = exporter.finish()
            results[dataset] = {"row_counts": report["row_counts"], "quantity_conservation": report["all_quantity_checks_passed"]}
            print(json.dumps({dataset: results[dataset]}, ensure_ascii=False), flush=True)
        finally:
            for writer in exporter.writers.values():
                writer.close()
    from services.external_canonical_pipeline import write_validation_matrix
    print(json.dumps({"external_validation_matrix": str(write_validation_matrix(args.data_root))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
