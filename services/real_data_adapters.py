"""Explicit source mappings for real data; no UI or decision-rule changes."""
from __future__ import annotations

import json
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
    # CSV 1-based data record (header excluded); JSON array position likewise.
    out["source_row_id"] = pd.Series(raw.index + 1, index=raw.index).astype(str)
    out["transform_version"] = VERSION
    out["date_grain"] = grain
    out["scope"] = scope
    out["product_grain"] = "product"
    return out


def mapped(raw, dataset, source, mapping, grain="daily", scope="location_product", date_field=None, unit=None, sheet="csv"):
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
    return out


SUHYUP_ID = {"location_id": "물류센터-공판장 코드", "location_name": "물류센터-공판장명", "product_id": "수산물품목코드", "product_name": "수산물품목명", "product_state": "상태가공분류코드"}


def adapt_suhyup(raw, source, kind):
    measures = ({"inventory_qty": "재고량", "inventory_weight_kg": "재고량(킬로그램)"} if kind == "stock" else
                {"inbound_qty": "입고량", "outbound_qty": "출고량", "inbound_weight_kg": "입고량(킬로그램)", "outbound_weight_kg": "출고량(킬로그램)"})
    table = "inventory_snapshot" if kind == "stock" else "inventory_flow"
    out = mapped(raw, "suhyup", source, {**SUHYUP_ID, **measures}, date_field="기준일자")
    # Raw count unit is unspecified; kg is independently observed, not inferred.
    if kind == "flow":
        out["flow_kind"] = "observed_inbound_outbound"
    return table, canonicalize(out, table), measures


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


def adapt_jangbogo(raw, source, family):
    work = raw.copy()
    work.columns = work.columns.str.upper()
    mapping = {}
    for target, origin in [("location_id", "WRHS_CODE"), ("location_name", "WRHS_NM"), ("product_id", "FDMT_PDLT_CODE"), ("product_name", "FDMT_PDLT_NM"), ("category", "FDMT_PDLT_LGLS_NM")]:
        if origin in work:
            mapping[target] = origin
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


def adapt_nfqs(raw, source, year, quarter):
    period = pd.Period(f"{year}Q{quarter}", freq="Q")
    start, end = str(period.start_time.date()), str(period.end_time.date())
    rows = []
    for index, item in enumerate(raw):
        for code, name in {**REGIONS, "INVENTOTAL": "전국 합계"}.items():
            # Missing region field means missing, never zero.
            rows.append({"snapshot_date": end, "period_start": start, "period_end": end,
                         "raw_date": f"{year}Q{quarter}", "date_grain": "quarterly",
                         "location_id": code, "location_name": name, "product_id": item.get("ICEGDFG"),
                         "product_name": item.get("CODEKNM"), "inventory_qty": item.get(code),
                         "scope": "national_total" if code == "INVENTOTAL" else "region",
                         "product_grain": "all_products" if item.get("ICEGDFG") == "999" else "product_group", "unit": "ton", "source_dataset": "nfqs",
                         "validation_flags": "aggregate_product_total" if item.get("ICEGDFG") == "999" else "",
                         "source_file": str(source).replace("\\", "/"), "source_sheet_or_table": "LIST",
                         "source_row_id": f"{index + 1}:{code}", "transform_version": VERSION})
    return canonicalize(pd.DataFrame(rows), "inventory_snapshot")


def adapt_aihub(raw, source, sheet):
    # The workbook identifies no site/SKU. The warehouse metadata is not a join key.
    work = raw.copy()
    work.columns = [str(n) for n in range(work.shape[1])]
    work["_date"] = pd.to_datetime(work["0"], errors="coerce").dt.strftime("%Y-%m-%d")
    output = []
    for table, measures in [("inventory_snapshot", {"inventory_qty": "3", "inventory_volume_m3": "28", "utilization_ratio": "29"}),
                            ("inventory_flow", {"inbound_qty": "1", "outbound_qty": "2"})]:
        out = mapped(work, "aihub", source, measures, scope="unidentified_site_all_products", date_field="_date", unit="item", sheet=sheet)
        out["source_row_id"] = pd.Series(work.index + 3, index=work.index).astype(str)
        out["product_grain"] = "all_products"
        if table == "inventory_flow":
            out["flow_kind"] = "site_aggregate_inbound_outbound"
        output.append((table, canonicalize(out, table), measures, work))
    return output


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
