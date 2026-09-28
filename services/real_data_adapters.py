"""Explicit source mappings for real data; no UI or decision-rule changes."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from services.canonical_schema import VERSION, canonicalize, parse_period

DATA_ROOT = Path("C:/VARO_V2_REAL_DATA")
DATASETS = {
    "suhyup": "02_Korea_Suhyup_Logistics",
    "logisall": "22_KADX_LOGISALL",
    "jangbogo": "23_KADX_JANGBOGO",
    "nfqs": "29_DOMESTIC_ADDITIONAL/NFQS_FISH_INVENTORY",
    "aihub": "26_AIHUB_INDUSTRIAL_LOGISTICS",
}
REGIONS = {"SEOUL": "서울", "INCHEON": "인천", "JANGHANG": "장항", "YEOSU": "여수", "MOKPO": "목포", "WANDO": "완도", "JEJU": "제주", "BUSAN": "부산", "TONGYEONG": "통영", "POHANG": "포항", "GANGNEUNG": "강릉", "PYEONGTAEK": "평택", "JEONJU": "전주", "INCHEONAIRPORT": "인천공항"}

# Unit evidence. Only what a source header, official page/specification or label states.
UNIT_EVIDENCE = {
    "suhyup": {"quantity": {"unit": None, "status": "UNKNOWN", "evidence": "Raw headers 재고량/입고량/출고량 carry no unit; data.go.kr 15102797/15102798 snapshot has no column unit."},
               "weight": {"unit": "kg", "status": "DIRECT", "evidence": "Raw headers 재고량(킬로그램), 입고량(킬로그램), 출고량(킬로그램)."},
               "benchmark_route": {"currency": "KRW", "distance": "km", "time": "min", "status": "DERIVED", "evidence": "Existing reference transport engine benchmark columns move_cost/distance_km/travel_time_min."}},
    "logisall": {"quantity": {"unit": None, "status": "UNKNOWN", "evidence": "KADX product specifications list STRGE_QY 저장량, SLE_QY 판매량, DSBN_QY 유통량 without any unit (results/kadx_product_metadata.json)."}},
    "jangbogo": {"quantity": {"unit": None, "status": "UNKNOWN", "evidence": "KADX specifications list PRCA_DMND_QYT 구매요청수량 and TOT_SLE_QYT 총판매수량 without any unit; product-name pack text (5KG, 1EA) describes the item, not the quantity column."}},
    "nfqs": {"quantity": {"unit": "ton", "status": "SOURCE_METADATA", "evidence": "Official page https://www.nfqs.go.kr/hpmg/data/actionMarineStockForm.do?menuId=M0000226 states '재고량의 단위는 톤(t) 입니다.' (re-read 2026-09-28)."}},
    "aihub": {"quantity": {"unit": "item", "status": "SOURCE_METADATA", "evidence": "Workbook headers 입고물품/출고물품/재고 with size-class '물품 분류별 수량' counts that sum to the totals."},
              "volume": {"unit": "m3", "status": "DIRECT", "evidence": "Workbook header 보관공간(㎥)."},
              "measurement": {"dimension_unit": "cm", "weight_unit": "kg", "status": "SOURCE_METADATA", "evidence": "Official COCO label caption '가로 …cm, 세로 …cm, 높이 …cm, 무게 …kg' equals the measurement member values; member headers length/width/height/weight carry no unit."}},
}
# NFQS raw JSON reports at most 6 decimals of a ton (1e-6 t); longer tokens are
# IEEE-754 accumulation residue in the server response (e.g. -3.885780586188048E-16).
NFQS_FLOAT_RESIDUE_TOLERANCE_TON = 1e-9
NFQS_IDENTITY_TOLERANCE_TON = 1e-7
AIHUB_MEASUREMENT_CONTEXT = {"01_입고물품": "inbound_item", "02_출고물품": "outbound_item"}


def csv_encoding(path: Path) -> str:
    # Strict decoding, including the entire small buffer; never replacement.
    sample = path.read_bytes()[:65536] if path.stat().st_size < 65536 else None
    if sample is None:
        with path.open("rb") as handle:
            sample = handle.read(65536)
    import codecs
    for enc in ("utf-8-sig", "cp949"):
        try:
            codecs.getincrementaldecoder(enc)().decode(sample, final=False)
            return enc
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Unsupported encoding: {path}")


def csv_chunks(path: Path, chunksize=50000, columns=None):
    return pd.read_csv(path, encoding=csv_encoding(path), dtype="string", keep_default_na=False,
                       na_values=[""], chunksize=chunksize, usecols=columns)


def base_frame(raw, dataset, source, grain="daily", scope="location_product", sheet="csv"):
    out = pd.DataFrame(index=raw.index)
    out["source_dataset"] = dataset
    out["source_file"] = str(source).replace("\\", "/")
    out["source_sheet_or_table"] = sheet
    if "_source_row_ids" in raw:
        # Collapsed records keep every original 1-based record id.
        out["source_row_id"] = raw["_source_row_ids"].astype(str)
        out["source_record_count"] = raw["_source_record_count"]
    else:
        # CSV 1-based data record (header excluded); JSON array position likewise.
        out["source_row_id"] = pd.Series(raw.index + 1, index=raw.index).astype(str)
        out["source_record_count"] = 1
    out["transform_version"] = VERSION
    out["date_grain"] = grain
    out["scope"] = scope
    out["product_grain"] = "product"
    return out


def mapped(raw, dataset, source, mapping, grain="daily", scope="location_product", date_field=None, unit=None, sheet="csv", unit_status=None):
    out = base_frame(raw, dataset, source, grain, scope, sheet)
    for target, origin in mapping.items():
        if origin not in raw:
            raise ValueError(f"Required mapped source column absent: {origin} in {source}")
        out[target] = raw[origin]
    if date_field:
        out["raw_date"] = raw[date_field]
        start, end = parse_period(raw[date_field], grain)
        out["period_start"], out["period_end"] = start, end
        out["snapshot_date" if "inventory_qty" in mapping else "date"] = start
    if unit is not None:
        out["unit"] = unit
    if unit_status is not None:
        out["unit_status"] = unit_status
    return out


def collapse_source_records(raw, keys, measures=()):
    """Collapse raw records that share a complete canonical key.

    Used only where the dataset investigation proved that repeated keys are
    distinct finer-grain records (sum) or identical occurrence records (count).
    Keys must cover every non-measure column, so no differing attribute is
    hidden. A record with a NULL key part or a NULL/non-numeric/negative
    measure is never merged: unknown is not "same" and flagged values stay
    visible. `_source_row_ids` keeps every original 1-based record id.
    """
    keys, measures = list(keys), list(measures)
    extra = set(raw.columns) - set(keys) - set(measures)
    if extra:
        raise ValueError(f"Collapse keys must cover all non-measure columns; uncovered: {sorted(extra)}")
    work = raw.copy()
    work["_id"] = pd.Series(work.index + 1, index=work.index).astype(str)
    work["_order"] = work.index
    numeric = {m: pd.to_numeric(work[m], errors="coerce") for m in measures}
    mergeable = work[keys].notna().all(axis=1)
    for m in measures:
        mergeable &= numeric[m].notna() & numeric[m].ge(0)
    single = work.loc[~mergeable].copy()
    single["_source_row_ids"] = single["_id"]
    single["_source_record_count"] = 1
    groups = work.loc[mergeable].assign(**{m: numeric[m][mergeable] for m in measures}).groupby(keys, sort=False)
    merged = groups.agg(**{m: (m, "sum") for m in measures}, _source_row_ids=("_id", "|".join),
                        _source_record_count=("_id", "size"), _order=("_order", "min")).reset_index()
    out = pd.concat([merged, single[[*keys, *measures, "_source_row_ids", "_source_record_count", "_order"]]], ignore_index=True)
    out = out.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
    return out[[*raw.columns, "_source_row_ids", "_source_record_count"]]


SUHYUP_ID = {"location_id": "물류센터-공판장 코드", "location_name": "물류센터-공판장명", "product_id": "수산물품목코드", "product_name": "수산물품목명", "product_state": "상태가공분류코드"}


def adapt_suhyup(raw, source, kind):
    measures = ({"inventory_qty": "재고량", "inventory_weight_kg": "재고량(킬로그램)"} if kind == "stock" else
                {"inbound_qty": "입고량", "outbound_qty": "출고량", "inbound_weight_kg": "입고량(킬로그램)", "outbound_weight_kg": "출고량(킬로그램)"})
    table = "inventory_snapshot" if kind == "stock" else "inventory_flow"
    out = mapped(raw, "suhyup", source, {**SUHYUP_ID, **measures}, date_field="기준일자")
    # Raw count unit is unspecified; kg is independently observed, not inferred.
    out["weight_unit"], out["weight_unit_status"] = "kg", "DIRECT"
    if kind == "flow":
        out["flow_kind"] = "observed_inbound_outbound"
    return table, canonicalize(out, table), measures


# LogisAll ZIPs are masked to a first-digit zone (00000..60000); several source
# records share a (date, product, zone) key. They are distinct sub-zone records:
# per (date, product, zone) the sales-row multiplicity equals the distribution
# sender-zone multiplicity in every release. Their quantities are summed.
LOGISALL_COLLAPSE = {"sales": (["BASE_YMD", "AGFD_PDLT_NM", "AGFD_SLPL_ZIP"], ["SLE_QY"]),
                     "network": (["BASE_YMD", "AGFD_PDLT_NM", "FRWAR_ZIP", "ARVL_ZIP"], ["DSBN_QY"])}


def adapt_logisall(raw, source, family):
    work = raw.copy()
    grain, scope = "daily", "location_product"
    common = {"product_id": "AGFD_PDLT_NM", "product_name": "AGFD_PDLT_NM"}
    if family == "stock":
        work["_period"] = work["BASE_YR"] + work["BASE_MM"].str.zfill(2)
        date_field, grain, scope = "_period", "monthly", "national"
        table, measures = "inventory_snapshot", {"inventory_qty": "STRGE_QY"}
    elif family == "sales":
        date_field = "BASE_YMD"
        table, measures = "demand_series", {"sales_qty": "SLE_QY"}
        common["location_id"] = "AGFD_SLPL_ZIP"
    else:
        date_field = "BASE_YMD"
        table, measures = "transfer_network", {"shipment_qty": "DSBN_QY"}
        common.update(source_id="FRWAR_ZIP", target_id="ARVL_ZIP")
    out = mapped(work, "logisall", source, {**common, **measures}, grain, scope, date_field)
    if family == "network":
        out["record_kind"] = "observed_distribution"
    return table, canonicalize(out, table), measures


JANGBOGO_MISSING_MARKERS = {"NULL"}


def adapt_jangbogo(raw, source, family):
    work = raw.copy()
    # Source headers vary in case; collapse lineage helpers (_source_*) keep their names.
    work.columns = [c if c.startswith("_") else c.upper() for c in work.columns]
    mapping = {}
    for target, origin in [("location_id", "WRHS_CODE"), ("location_name", "WRHS_NM"), ("product_id", "FDMT_PDLT_CODE"), ("product_name", "FDMT_PDLT_NM"), ("category", "FDMT_PDLT_LGLS_NM")]:
        if origin in work:
            mapping[target] = origin
    if "BRCD_INFO" in work and family != "observation":
        # Official grain is 상품 및 바코드별; the literal "NULL" is a missing marker like an empty cell.
        work["_barcode"] = work["BRCD_INFO"].mask(work["BRCD_INFO"].isin(JANGBOGO_MISSING_MARKERS))
        mapping["product_variant_id"] = "_barcode"
    is_category = "FDMT_PDLT_CODE" not in work
    if is_category:
        # Category identifiers cannot accidentally join SKU identifiers.
        work["_category_id"] = "category:" + work["FDMT_PDLT_LGLS_CODE"]
        mapping.update(product_id="_category_id", product_name="FDMT_PDLT_LGLS_NM")
    grain = "daily" if "CFMTN_YMD" in work else "monthly"
    date_field = "CFMTN_YMD" if grain == "daily" else "CFMTN_YM"
    measures = {}
    if family == "observation":
        table = "location_product_observation"
        if "PRCA_NBOT" in work:
            measures = {"purchase_count": "PRCA_NBOT"}
    elif family == "sales":
        table, measures = "demand_series", {"sales_qty": "TOT_SLE_QYT"}
    else:
        table, measures = "inventory_flow", {"order_qty": "PRCA_DMND_QYT"}
    scope = "location_product" if "WRHS_CODE" in work else "all_locations_unspecified"
    out = mapped(work, "jangbogo", source, {**mapping, **measures}, grain, scope, date_field)
    out["product_grain"] = "category" if is_category else "product"
    if table == "inventory_flow":
        out["flow_kind"] = "purchase_request_not_receipt"
    # Invalid date stays NULL with raw_date and validation flags; no correction.
    return table, canonicalize(out, table), measures, work


def nfqs_flow_identity_diff(item):
    """ICEALIST + ICEAIN - ICEAOUT - INVENTOTAL for one record, or None.

    data.go.kr 15082985 describes the series as 이월량, 입출하량, 재고량; the
    hidden fields are read with that meaning only for this same-record check.
    """
    try:
        values = [float(item[k]) for k in ("ICEALIST", "ICEAIN", "ICEAOUT", "INVENTOTAL")]
    except (KeyError, TypeError, ValueError):
        return None
    return values[0] + values[1] - values[2] - values[3]


def adapt_nfqs(raw, source, year, quarter):
    period = pd.Period(f"{year}Q{quarter}", freq="Q")
    start, end = str(period.start_time.date()), str(period.end_time.date())
    evidence = UNIT_EVIDENCE["nfqs"]["quantity"]
    rows = []
    for index, item in enumerate(raw):
        diff = nfqs_flow_identity_diff(item)
        identity_break = diff is not None and abs(diff) > NFQS_IDENTITY_TOLERANCE_TON
        for code, name in {**REGIONS, "INVENTOTAL": "전국 합계"}.items():
            flags = ["aggregate_product_total"] if item.get("ICEGDFG") == "999" else []
            if code == "INVENTOTAL" and identity_break:
                flags.append("source_flow_identity_mismatch")
            # Missing region field means missing, never zero.
            rows.append({"snapshot_date": end, "period_start": start, "period_end": end,
                         "raw_date": f"{year}Q{quarter}", "date_grain": "quarterly",
                         "location_id": code, "location_name": name, "product_id": item.get("ICEGDFG"),
                         "product_name": item.get("CODEKNM"), "inventory_qty": item.get(code),
                         "scope": "national_total" if code == "INVENTOTAL" else "region",
                         "product_grain": "all_products" if item.get("ICEGDFG") == "999" else "product_group",
                         "unit": evidence["unit"], "unit_status": evidence["status"], "source_dataset": "nfqs",
                         "validation_flags": "|".join(flags), "source_record_count": 1,
                         "source_file": str(source).replace("\\", "/"), "source_sheet_or_table": "LIST",
                         "source_row_id": f"{index + 1}:{code}", "transform_version": VERSION})
    return canonicalize(pd.DataFrame(rows), "inventory_snapshot", residue_tolerance={"inventory_qty": NFQS_FLOAT_RESIDUE_TOLERANCE_TON})


def nfqs_panel_coverage(period_products):
    """Product x period presence. An absent product is NOT_REPORTED, never zero stock.

    period_products maps a period label to {product_id: product_name} as reported.
    """
    periods = sorted(period_products)
    names = {}
    for period in periods:
        for pid, name in period_products[period].items():
            names.setdefault(pid, []).append(name)
    rows = []
    for pid in sorted(names):
        reported = [p for p in periods if pid in period_products[p]]
        for period in periods:
            rows.append({"product_id": pid, "product_name": period_products[period].get(pid), "period": period,
                         "status": "REPORTED" if pid in period_products[period] else "NOT_REPORTED",
                         "first_reported": reported[0], "last_reported": reported[-1],
                         "name_variants": len(set(names[pid]))})
    return pd.DataFrame(rows)


def adapt_aihub(raw, source, sheet):
    # The workbook identifies no site/SKU. The warehouse metadata is not a join key.
    work = raw.copy()
    work.columns = [str(n) for n in range(work.shape[1])]
    work["_date"] = pd.to_datetime(work["0"], errors="coerce").dt.strftime("%Y-%m-%d")
    quantity, volume = UNIT_EVIDENCE["aihub"]["quantity"], UNIT_EVIDENCE["aihub"]["volume"]
    output = []
    for table, measures in [("inventory_snapshot", {"inventory_qty": "3", "inventory_volume_m3": "28", "utilization_ratio": "29"}),
                            ("inventory_flow", {"inbound_qty": "1", "outbound_qty": "2"})]:
        out = mapped(work, "aihub", source, measures, scope="unidentified_site_all_products", date_field="_date", unit=quantity["unit"], sheet=sheet, unit_status=quantity["status"])
        out["source_row_id"] = pd.Series(work.index + 3, index=work.index).astype(str)
        out["product_grain"] = "all_products"
        if table == "inventory_snapshot":
            out["volume_unit"], out["volume_unit_status"] = volume["unit"], volume["status"]
        if table == "inventory_flow":
            out["flow_kind"] = "site_aggregate_inbound_outbound"
        output.append((table, canonicalize(out, table), measures, work))
    return output


def adapt_aihub_measurements(raw, source):
    """One row per physical measurement member; no product-level deduplication."""
    evidence = UNIT_EVIDENCE["aihub"]["measurement"]
    mapping = {"product_id": "barcode", "product_name": "product_name", "category": "category_l2", "category_code": "kan_code",
               "length": "length_cm", "width": "width_cm", "height": "height_cm", "weight": "weight_kg"}
    out = mapped(raw, "aihub", source, mapping, grain="static", scope="measured_item")
    out["source_sheet_or_table"] = raw["source_member"]
    out["source_row_id"] = "1"
    context = raw["flow_dir"].map(AIHUB_MEASUREMENT_CONTEXT)
    if context.isna().any():
        raise ValueError(f"Unrecognised AI Hub measurement folder: {sorted(raw.loc[context.isna(), 'flow_dir'].unique())}")
    out["measurement_context"] = context
    out["dimension_unit"], out["dimension_unit_status"] = evidence["dimension_unit"], evidence["status"]
    out["weight_unit"], out["weight_unit_status"] = evidence["weight_unit"], evidence["status"]
    measures = {"length": "length_cm", "width": "width_cm", "height": "height_cm", "weight": "weight_kg"}
    return canonicalize(out, "product_measurement"), measures


# A unit must not run into a Latin word; 'x' is a multiplication sign (50CMx350M).
_PACK_TOKEN = re.compile(r"(\d+(?:\.\d+)?(?:\s*[~\-]\s*\d+(?:\.\d+)?)?)\s*(KG|ML|MM|CM|EA|BOX|G|L|M|T|팩|포|입|과|수|손|번|매|개)(?![A-WYZ])", re.IGNORECASE)


def pack_spec_tokens(name):
    """Deterministic pack/grade tokens (quantity + unit) printed in a product name."""
    if name is None or (isinstance(name, float) and math.isnan(name)) or pd.isna(name):
        return ()
    return tuple(sorted(re.sub(r"\s+", "", f"{n}{u.upper()}") for n, u in _PACK_TOKEN.findall(str(name))))


def classify_name_versions(versions):
    """versions: rows with product_name, valid_from, valid_to (ISO dates).

    temporal_relation is SEQUENTIAL when no two names share a date range;
    name_change_class is PACK_SPEC_CHANGED when printed pack/grade tokens differ,
    otherwise LABEL_CHANGED_SAME_PACK_TOKENS. Nothing is merged or renamed.
    """
    rows = list(versions)
    if len(rows) < 2:
        return "SINGLE_NAME", "UNCHANGED"
    tokens = {pack_spec_tokens(r["product_name"]) for r in rows}
    change = "PACK_SPEC_CHANGED" if len(tokens) > 1 else "LABEL_CHANGED_SAME_PACK_TOKENS"
    if any(pd.isna(r["valid_from"]) or pd.isna(r["valid_to"]) for r in rows):
        return "INDETERMINATE_UNDATED", change
    overlap = any(a["valid_from"] <= b["valid_to"] and b["valid_from"] <= a["valid_to"]
                  for i, a in enumerate(rows) for b in rows[i + 1:])
    return ("CONCURRENT" if overlap else "SEQUENTIAL"), change


def suhyup_to_existing(snapshot, flow):
    """Strict bridge to the existing Suhyup validator; no demand/sales imputation."""
    keys = ["location_id", "product_id", "product_state"]
    stock = snapshot.rename(columns={"snapshot_date": "date"})
    joined = stock[["date", *keys, "inventory_qty"]].merge(
        flow[["date", *keys, "inbound_qty", "outbound_qty"]], on=["date", *keys], how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Stock/flow key mismatch; cannot construct legacy benchmark input")
    return joined.drop(columns="_merge").rename(columns={"location_id": "center_code", "product_id": "product_code", "product_state": "state_code", "inventory_qty": "stock_qty"})


def verified_identifier_crosswalk(source_values, target_values):
    """Match observed numeric code spellings only; reject missing/ambiguous IDs.

    This is an explicit compatibility crosswalk, never a global canonical cast.
    """
    source = sorted(set(str(v) for v in source_values if pd.notna(v)))
    target = sorted(set(str(v) for v in target_values if pd.notna(v)))
    normalize = lambda v: str(int(v)) if v.isdigit() else v
    lookup = {}
    for value in target:
        key = normalize(value)
        if key in lookup and lookup[key] != value:
            raise ValueError("Ambiguous identifier normalization")
        lookup[key] = value
    result = {}
    for value in source:
        if normalize(value) not in lookup:
            raise ValueError(f"No observed identifier counterpart: {value}")
        result[value] = lookup[normalize(value)]
    if len(set(result.values())) != len(result):
        raise ValueError("Identifier crosswalk is not one-to-one")
    return result
