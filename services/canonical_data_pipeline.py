"""Bounded-memory offline canonical exports and evidence reports.

Run: python -m services.canonical_data_pipeline --data-root C:/VARO_V2_REAL_DATA
Only canonical_* files and new reports are written; raw is read-only.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from services.canonical_schema import COMMON, KEYS, LINEAGE, NUMERIC, TABLE_FIELDS, VERSION, canonicalize, schema_document
from services.real_data_adapters import (DATA_ROOT, DATASETS, REGIONS, adapt_aihub, adapt_jangbogo,
    adapt_logisall, adapt_nfqs, adapt_suhyup, base_frame, csv_chunks, csv_encoding, mapped, suhyup_to_existing, verified_identifier_crosswalk)

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


def algorithm_coverage(dataset, present):
    """Conservative semantic gate. A union of column names never proves FULL."""
    rows = []
    for algorithm, required in ALGORITHMS.items():
        have = sorted(set(required) & set(present))
        missing = sorted(set(required) - set(present))
        level, reason = "UNSUPPORTED", "Required semantics/fields absent; no forced algorithm execution."
        if dataset == "suhyup" and algorithm in {"Transport Cost", "Greedy", "VHS", "Pareto", "MILP", "Optimality Gap", "Varo Final", "DQN"}:
            level = "BENCHMARK_ONLY"
            reason = "Existing 620-candidate/31-day benchmark only; outbound is not retail demand, target_need is the existing proxy, route costs are enriched reference estimates."
        elif dataset in {"logisall", "jangbogo"} and algorithm == "Demand Forecast":
            level = "PARTIAL"
            reason = "Actual sales time series exists; unit is unspecified, duplicate/source overlap must be resolved; category/SKU and daily/monthly grains cannot be pooled."
        elif dataset in {"suhyup", "aihub"} and algorithm == "Turnover":
            level = "PARTIAL"
            reason = "Stock and outbound available for logistics-flow diagnostics; outbound cannot substitute for retail sales_30d."
        elif dataset == "suhyup" and algorithm == "Store/Product Matching":
            level = "PARTIAL"
            reason = "Actual location/product/stock exists; unmet retail demand is unavailable."
        rows.append({"algorithm": algorithm, "supported": level != "UNSUPPORTED", "support_level": level,
                     "required_fields_present": "|".join(have), "missing_fields": "|".join(missing), "reason": reason})
    return pd.DataFrame(rows)


class Exporter:
    def __init__(self, root, dataset):
        import pyarrow as pa
        self.pa = pa
        self.root, self.dataset = Path(root), dataset
        self.folder = self.root / DATASETS[dataset]
        self.processed, self.results = self.folder / "processed", self.folder / "results"
        self.processed.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.writers, self.seen = {}, defaultdict(set)
        self.counts, self.flags, self.nonnull = Counter(), defaultdict(Counter), defaultdict(Counter)
        self.sources, self.mappings, self.conservation = [], [], []
        self.master_parts = defaultdict(list)
        self.notes = []
        self.present = set()
        self.date_ranges = {}
        self.checks = {}
        self.entity_references = defaultdict(set)
        self.units = defaultdict(set)

    def record_mapping(self, table, mapping, source, measures, grain, scope, unit=None, derived=None):
        derived = derived or {}
        fields = {}
        metadata = {"date_grain": grain, "scope": scope, "transform_version": VERSION, "source_dataset": self.dataset,
                    "source_file": source, "source_sheet_or_table": "source table", "source_row_id": "1-based source record"}
        for field in (*COMMON, *TABLE_FIELDS[table]):
            if field in derived:
                fields[field] = {"coverage": "DERIVED", **derived[field]}
            elif field in mapping:
                fields[field] = {"coverage": "DIRECT" if field == mapping[field] else "RENAMED", "source_columns": [mapping[field]], "unit": unit if field in measures else None}
            elif field in metadata:
                fields[field] = {"coverage": "DERIVED", "source_columns": [], "formula": str(metadata[field]), "assumptions": "Source structure metadata; not a fabricated observation", "unit": None}
            elif field == "unit" and unit:
                fields[field] = {"coverage": "DERIVED", "source_columns": [], "formula": unit, "assumptions": "Explicit source header/documentation unit", "unit": unit}
            else:
                fields[field] = {"coverage": "UNAVAILABLE", "source_columns": [], "unit": None}
        self.mappings.append({"source_file": source, "table": table, "date_grain": grain, "scope": scope, "fields": fields})

    def write(self, table, frame, raw=None, measures=None, masters=True):
        import pyarrow.parquet as pq
        frame = frame.copy()
        keys = pd.util.hash_pandas_object(frame[list(KEYS[table])], index=False).tolist()
        duplicate = []
        for key in keys:
            duplicate.append(key in self.seen[table])
            self.seen[table].add(key)
        mask = pd.Series(duplicate, index=frame.index)
        if mask.any():
            frame.loc[mask, "validation_flags"] = frame.loc[mask, "validation_flags"].fillna("").map(lambda s: s + ("|" if s else "") + "duplicate_key")
            frame.loc[mask, "analysis_eligible"] = "false"
        self.counts[table] += len(frame)
        for col in frame:
            n = int(frame[col].notna().sum())
            self.nonnull[table][col] += n
            if n:
                self.present.add(col)
        self.flags[table].update(flag for s in frame.validation_flags.fillna("") for flag in str(s).split("|") if flag)
        datecol = "snapshot_date" if table == "inventory_snapshot" else "date"
        if datecol in frame and frame[datecol].notna().any():
            low, high = frame[datecol].dropna().min(), frame[datecol].dropna().max()
            old = self.date_ranges.get(table, [low, high])
            self.date_ranges[table] = [min(low, old[0]), max(high, old[1])]
        if raw is not None:
            for target, origin in (measures or {}).items():
                before = pd.to_numeric(raw[origin], errors="coerce")
                after = pd.to_numeric(frame[target], errors="coerce")
                a, b = before.sum(min_count=1), after.sum(min_count=1)
                ok = (pd.isna(a) and pd.isna(b)) or math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-7)
                self.conservation.append({"source_file": str(frame.source_file.iloc[0]), "table": table, "field": target,
                    "source_column": origin, "rows_source": len(raw), "rows_canonical": len(frame),
                    "source_sum": None if pd.isna(a) else float(a), "canonical_sum": None if pd.isna(b) else float(b),
                    "source_nonnull": int(before.notna().sum()), "canonical_nonnull": int(after.notna().sum()), "passed": bool(ok and len(raw) == len(frame) and before.notna().sum() == after.notna().sum())})
                if not self.conservation[-1]["passed"]:
                    raise AssertionError(f"Conservation failure: {table}.{target}")
        schema = self.pa.schema([(c, self.pa.float64() if c in NUMERIC else self.pa.string()) for c in frame])
        arr = self.pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        if table not in self.writers:
            self.writers[table] = pq.ParquetWriter(self.processed / f"canonical_{table}.parquet", schema, compression="zstd")
        self.writers[table].write_table(arr)
        if masters:
            self.collect_masters(frame)
        for col, entity in [("product_id", "product_master"), ("location_id", "location_master"), ("source_id", "location_master"), ("target_id", "location_master")]:
            if col in frame and table not in {"product_master", "location_master"}:
                self.entity_references[entity].update(frame[col].dropna().astype(str).unique())
        self.units[table].update(frame.unit.dropna().astype(str).unique())

    def collect_masters(self, frame):
        for table, idcol in [("product_master", "product_id"), ("location_master", "location_id")]:
            if idcol not in frame:
                continue
            cols = list(dict.fromkeys([*LINEAGE, "date_grain", "scope", "product_grain", "unit", *[c for c in TABLE_FIELDS[table] if c in frame]]))
            part = frame.loc[frame[idcol].notna(), cols].drop_duplicates(list(c for c in KEYS[table] if c in cols))
            if not part.empty:
                self.master_parts[table].append(part)
        if "source_id" in frame:
            for field in ("source_id", "target_id"):
                part = frame[[*LINEAGE, "date_grain", "scope", field]].rename(columns={field: "location_id"}).dropna(subset=["location_id"]).drop_duplicates("location_id")
                self.master_parts["location_master"].append(part)

    def finish(self):
        master_ids = {}
        for table, parts in self.master_parts.items():
            if not parts:
                continue
            data = pd.concat(parts, ignore_index=True)
            keys = [c for c in KEYS[table] if c in data]
            name_col = "product_name" if table == "product_master" else "location_name"
            if name_col in data:
                conflicts = data.groupby(keys, dropna=False)[name_col].nunique(dropna=True)
                self.checks[table + "_attribute_conflicts"] = {"conflicting_name_keys": int(conflicts.gt(1).sum()), "policy": "First observed label retained; source fact rows retain their own labels."}
            data = data.drop_duplicates(keys, keep="first")
            idcol = "product_id" if table == "product_master" else "location_id"
            master_ids[table] = set(data[idcol].dropna().astype(str))
            self.write(table, canonicalize(data, table), masters=False)
            self.mappings.append({"table": table, "operation": "distinct observed identifiers; first-source lineage preserved; not an authoritative cross-dataset master", "fields": {field: {"coverage": "AGGREGATED" if field in data and data[field].notna().any() else "UNAVAILABLE", "source_columns": [field] if field in data else [], "formula": "distinct source identifier, first observed attribute", "unit": None, "assumptions": "Dataset-local identity only"} for field in (*COMMON, *TABLE_FIELDS[table])}})
        for writer in self.writers.values():
            writer.close()
        for table, references in self.entity_references.items():
            if table not in master_ids and table in self.writers:
                col = "product_id" if table == "product_master" else "location_id"
                master_ids[table] = set(pd.read_parquet(self.processed / f"canonical_{table}.parquet", columns=[col])[col].dropna().astype(str))
            self.checks[table + "_mapping"] = {"observed_references": len(references), "unmatched_identifiers": sorted(references - master_ids.get(table, set()))}
        self.checks["unit_consistency"] = {table: {"observed_units": sorted(units), "unknown_unit_rows": self.counts[table] - self.nonnull[table]["unit"], "policy": "Never combine quantities across units or unknown-unit source scopes"} for table, units in self.units.items()}
        report = {"dataset": self.dataset, "transform_version": VERSION, "row_counts": dict(self.counts), "semantic_checks": self.checks,
            "date_ranges": self.date_ranges, "flags": {k: dict(v) for k, v in self.flags.items()},
            "field_coverage": {t: {c: "MISSING" if n == 0 else "FULL" if n == self.counts[t] else "PARTIAL" for c,n in self.nonnull[t].items()} for t in self.counts},
            "null_counts": {t: {c: self.counts[t]-n for c,n in self.nonnull[t].items()} for t in self.counts},
            "quantity_conservation": self.conservation,
            "all_quantity_checks_passed": all(c["passed"] for c in self.conservation),
            "duplicate_policy": "Preserve every source row; flag repeated business keys; source_file+source_row_id remains physical key. Do not sum overlapping releases without a source selection policy.",
            "quarantine_policy": "Rows with invalid_date/negative/invalid_numeric flags remain traceable in canonical with analysis_eligible=false. No correction or deletion.",
            "notes": self.notes, "selected_sources": self.sources}
        mapping = {"schema": schema_document(), "dataset": self.dataset, "mappings": self.mappings,
                   "scope": "Selected usable full tables/numeric subset, not a claim that all collected files are converted", "notes": self.notes}
        for name, obj in [("canonical_mapping.json", mapping), ("data_quality_report.json", report)]:
            (self.results / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
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


def stream_file(e, path, adapter, family):
    source = path.relative_to(e.root).as_posix()
    rows, columns, dtypes, dates = 0, [], {}, []
    source_monotonic, previous_date = True, None
    mapped_once = False
    for raw in csv_chunks(path):
        result = adapter(raw, source, family)
        table, frame, measures = result[:3]
        work = result[3] if len(result) == 4 else raw
        e.write(table, frame, work, measures)
        rows += len(raw)
        columns = list(raw.columns)
        dtypes = {c: ("numeric text" if pd.to_numeric(raw[c].dropna(), errors="coerce").notna().all() else "string") for c in raw}
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
            mapping["raw_date"] = datesource
            for name in ("validation_flags", "analysis_eligible"):
                derived[name] = {"source_columns": list(work.columns), "formula": "canonical schema validation", "unit": None, "assumptions": "Flags do not repair observations"}
            e.record_mapping(table, mapping, source, measures, str(frame.date_grain.iloc[0]), str(frame.scope.iloc[0]), derived=derived)
            mapped_once = True
    e.sources.append({"file": source, "sheet_or_table": "csv", "row_count": rows, "columns": columns, "sample_dtypes": dtypes,
                      "date_range": [min(dates), max(dates)] if dates else None, "source_date_monotonic": source_monotonic,
                      "encoding": csv_encoding(path), "status": "full selected file", "family": family})


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
    frame["currency"] = "KRW"
    frame["validation_flags"] = "benchmark_proxy_constraints|unit_unspecified"
    measures = {k:v for k,v in mapping.items() if k in NUMERIC}
    e.write("transfer_network", canonicalize(frame, "transfer_network"), enriched, measures)
    deriv = {
        "source_surplus": {"source_columns": ["stock_qty", "center_code", "product_code", "date"], "formula": "existing enrich_inventory_constraints: max(0, source stock - same-day selected-network product median)", "unit": None, "assumptions": "Existing benchmark proxy, not observed transferable stock"},
        "target_need": {"source_columns": ["stock_qty", "outbound_qty"], "formula": "existing enrich_inventory_constraints: max(0, median stock + 7 * daily outbound - target stock)", "unit": None, "assumptions": "Existing benchmark demand proxy, not actual retail unmet demand"},
        "transport_cost": {"source_columns": ["move_cost"], "formula": "identity from existing enriched route benchmark", "unit": "KRW", "assumptions": "Reference transport engine estimate; not invoiced cost"}}
    for field, crosswalk in [("source_id", location_crosswalk), ("target_id", location_crosswalk), ("product_id", product_crosswalk)]:
        deriv[field] = {"source_columns": [field], "formula": "verified one-to-one raw code crosswalk", "crosswalk": crosswalk, "unit": None, "assumptions": "Existing benchmark removed leading zeroes; restore only actual observed raw identifiers"}
    e.record_mapping("transfer_network", mapping, str(path.relative_to(e.root)), measures, "daily", "location_product", derived=deriv)
    e.notes += ["Raw count unit unspecified; parallel kg observations retained separately.", "State-processing code is part of stock/flow key; never collapse it.", "No unit_weight inferred from stock_kg/stock_qty; packaging mix is unknown.", "Transfer candidates and proxy constraints are existing benchmark-derived records, not observed movements."]


def run_logisall(e):
    for prefix, family in [("TB_MI_STC", "stock"), ("TB_MRI_SLE_CW", "sales"), ("TB_MIR_DSBN_CW", "network")]:
        for path in sorted((e.folder / "raw/kadx_full_data").glob(prefix + "*.csv")):
            stream_file(e, path, adapt_logisall, family)
    e.notes += ["Full received monthly national stock, regional daily sales, regional distribution files used; public sample files excluded.", "Regional stock and regional inbound/outbound full products were not acquired; samples are not production evidence.", "ZIP is a regional identifier, not a warehouse or precise geolocation. Product names are dataset-local keys.", "National monthly stock cannot be joined to regional daily sales as same-grain inventory/demand.", "Production quantity PDTN_QY is not storage; production tables excluded.", "Quantity units are absent from the acquired specifications; left NULL."]


def run_jangbogo(e):
    folder = e.folder / "raw/kadx_full_data"
    for prefix, family in [("TB_SPOT_CL_MTH_BUY_YM", "orders"), ("TB_SPOT_CL_MTH_SALES_YM", "sales")]:
        for path in sorted(folder.glob(prefix + "*.csv")):
            stream_file(e, path, adapt_jangbogo, family)
    for name in ["TB_LGTC_PDLT_SALES_YMD_20230301.csv", "TB_LGTC_PDLT_SALES_YMD_20231010.csv"]:
        stream_file(e, folder / name, adapt_jangbogo, "observation")
    stream_file(e, folder / "TB_PDLT_BUY_YM_231010.csv", adapt_jangbogo, "orders")
    e.notes += ["Bounded first adapter scope: all small warehouse/category monthly sales+purchase tables; two warehouse/SKU observation releases; latest full SKU purchase-request partition. Large transaction tables are catalogued but not converted.", "PRCA_DMND_QYT is purchase-request quantity (order_qty), never inbound or realized demand.", "Warehouse/SKU table has no movement quantity; only location_product_observation is generated.", "Product-level purchase-request partition has no warehouse key, so no fabricated warehouse join.", "Category: prefixed identifiers cannot join SKU codes. No stock snapshot exists in selected tables.", "Invalid calendar strings become NULL dates with raw_date retained and analysis_eligible=false."]


def run_nfqs(e):
    signatures = {}
    product_sets = {}
    regional_mismatches, balance_mismatches = 0, 0
    for path in sorted((e.folder / "raw").glob("nfqs_fish_inventory_*.json")):
        year, q = path.stem.split("_")[-2:]
        if not 2020 <= int(year) <= 2025:
            continue
        raw = json.loads(path.read_text(encoding="utf-8-sig"))["LIST"]
        source = path.relative_to(e.root).as_posix()
        frame = adapt_nfqs(raw, source, year, q[1:])
        e.write("inventory_snapshot", frame)
        source_values = [row.get(k) for row in raw for k in [*REGIONS, "INVENTOTAL"]]
        before = pd.to_numeric(pd.Series(source_values), errors="coerce")
        after = frame.inventory_qty
        e.conservation.append({"source_file": source, "field": "inventory_qty", "table": "inventory_snapshot", "source_sum": float(before.sum()), "canonical_sum": float(after.sum()), "source_nonnull": int(before.notna().sum()), "canonical_nonnull": int(after.notna().sum()), "passed": math.isclose(float(before.sum()), float(after.sum()), abs_tol=1e-7, rel_tol=1e-12)})
        signatures[path.name] = sorted(set().union(*(set(row) for row in raw)))
        product_sets[path.name] = sorted(row.get("ICEGDFG") for row in raw)
        for row in raw:
            regional_mismatches += not math.isclose(sum(float(row[k]) for k in REGIONS), float(row["INVENTOTAL"]), abs_tol=1e-6, rel_tol=1e-9)
            balance_mismatches += not math.isclose(float(row["ICEALIST"]) + float(row["ICEAIN"]) - float(row["ICEAOUT"]), float(row["INVENTOTAL"]), abs_tol=1e-6, rel_tol=1e-9)
        e.sources.append({"file": source, "sheet_or_table": "LIST", "row_count": len(raw), "canonical_rows": len(frame), "columns": signatures[path.name], "date_range": [str(frame.snapshot_date.min()), str(frame.snapshot_date.max())]})
        e.record_mapping("inventory_snapshot", {"product_id": "ICEGDFG", "product_name": "CODEKNM", "inventory_qty": "region field or INVENTOTAL"}, source, {"inventory_qty": "region field or INVENTOTAL"}, "quarterly", "region/national_total", "ton",
            {"snapshot_date": {"source_columns": ["query year", "query quarter"], "formula": "quarter end", "unit": None, "assumptions": "Query metadata, not daily observations"}, "location_id": {"source_columns": list(REGIONS) + ["INVENTOTAL"], "formula": "unpivot field name", "unit": None, "assumptions": "National total is separately scoped; never sum it with regions"}})
    e.checks["quarter_structure"] = {"field_signatures": signatures, "product_sets": product_sets,
        "product_set_changed": len(set(tuple(v) for v in product_sets.values())) > 1,
        "region_sum_vs_reported_total_mismatches": regional_mismatches,
        "prior_plus_in_minus_out_vs_current_mismatches": balance_mismatches}
    e.notes += ["24 quarterly snapshots, no daily interpolation. National totals and regional rows have distinct scope; summing both double-counts inventory.", "ICEGDFG=999 is a reported all-product total; product_grain=all_products and aggregate_product_total flag prevent treating it as an extra SKU.", "Only cooperating cold-storage firms; not all national inventory.", "Floating negative source values retained and flagged; no zero-clipping."]


def run_aihub(e):
    path = next((e.folder / "raw").rglob("*물동량*.xlsx"))
    book = pd.ExcelFile(path)
    raw = pd.read_excel(path, sheet_name=book.sheet_names[0], header=None, skiprows=2)
    raw = raw.loc[raw[0].notna()].copy()
    source = path.relative_to(e.root).as_posix()
    for table, frame, measures, work in adapt_aihub(raw, source, book.sheet_names[0]):
        e.write(table, frame, work, measures)
        e.record_mapping(table, measures, source, measures, "daily", "unidentified_site_all_products", "item",
            {"date" if table == "inventory_flow" else "snapshot_date": {"source_columns": ["일자"], "formula": "Excel datetime -> ISO date", "assumptions": "Business dates only; missing days are not generated", "unit": None}})
    inv = pd.to_numeric(raw[3], errors="coerce")
    diff = inv.diff() - pd.to_numeric(raw[1], errors="coerce") + pd.to_numeric(raw[2], errors="coerce")
    e.notes += [f"Inventory balance between observed rows: {int(diff.dropna().abs().lt(1e-8).sum())}/{int(diff.notna().sum())}; gaps are not imputed.", "Workbook has no site key or SKU key; NULL preserved. Metadata's two named sites cannot be joined to this time series.", "Inventory volume m3 is observed occupied space, not capacity. capacity_m3_implied is deliberately excluded.", "No image/annotation archives opened or copied."]
    e.sources.append({"file": source, "sheet_or_table": book.sheet_names[0], "row_count": len(raw), "date_range": [str(raw[0].min()), str(raw[0].max())], "columns": ["일자", "입고물품", "출고물품", "재고", "size class counts", "occupied cm3", "occupied m3", "utilization ratio"]})
    metadata_path = next((e.folder / "raw").rglob("metadata_warehouse_information.json"))
    metadata = pd.DataFrame(json.loads(metadata_path.read_text(encoding="utf-8-sig"))["warehouse_info"])
    location_mapping = {"location_id": "site_name", "location_name": "site_name", "address": "site_address"}
    locations = mapped(metadata, "aihub", metadata_path.relative_to(e.root), location_mapping, grain="static", scope="named_sites_unlinked_to_flow", sheet="warehouse_info")
    e.write("location_master", canonicalize(locations, "location_master"), masters=False)
    e.record_mapping("location_master", location_mapping, str(metadata_path.relative_to(e.root)), {}, "static", "named_sites_unlinked_to_flow")
    # Reuse the already verified numeric extraction; trace each record to its
    # exact raw ZIP member. No COCO annotation or image data is read.
    measurement_path = e.folder / "processed/aihub71861_item_measurements.csv"
    zip_path = next((e.folder / "raw").rglob("Other.zip"))
    count = 0
    for raw_measure in csv_chunks(measurement_path):
        mapping = {"product_id": "barcode", "product_name": "product_name", "category": "category_l2", "unit_weight": "weight_kg"}
        products = mapped(raw_measure, "aihub", zip_path.relative_to(e.root), mapping, grain="static", scope="measured_item")
        products["source_sheet_or_table"] = raw_measure.source_member
        products["source_row_id"] = "1"
        products["weight_unit"] = "kg"
        e.write("product_master", canonicalize(products, "product_master"), raw_measure, {"unit_weight": "weight_kg"}, masters=False)
        count += len(raw_measure)
    e.record_mapping("product_master", mapping, str(measurement_path.relative_to(e.root)), {"unit_weight": "weight_kg"}, "static", "measured_item",
        derived={"source_sheet_or_table": {"source_columns": ["source_member"], "formula": "exact member path in Other.zip", "unit": None, "assumptions": "Reuse verified numeric extraction, no image annotation"}, "weight_unit": {"source_columns": ["weight_kg"], "formula": "kg", "unit": "kg", "assumptions": "Measurement unit explicit in existing verified extraction"}})
    e.sources.append({"file": str(measurement_path.relative_to(e.root)), "raw_archive": str(zip_path.relative_to(e.root)), "row_count": count, "status": "existing verified numeric extraction with per-row raw member lineage"})


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
            globals()["run_" + dataset](exporter)
            report = exporter.finish()
            results[dataset] = {"row_counts": report["row_counts"], "quantity_conservation": report["all_quantity_checks_passed"]}
            print(json.dumps({dataset: results[dataset]}, ensure_ascii=False), flush=True)
        finally:
            for writer in exporter.writers.values():
                writer.close()


if __name__ == "__main__":
    main()
