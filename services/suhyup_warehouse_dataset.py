"""수협 조합창고 재고 + 입출고 실데이터 안전 결합.

두 원본은 서로 다른 **입도(granularity)** 를 가진다.

* 입출고(flow) `표준코드` — 8자리. 어종 6자리 + 상태 2자리.
* 재고(stock) `표준어종코드` — 6자리. 어종만 있고 상태 구분이 없다.

따라서 두 코드의 **완전 일치는 0건**이고 코드를 그대로 join하면 안 된다. 대신 flow 코드의
앞 6자리가 stock의 어종코드와 같은 체계인지를 **데이터로 검증한 뒤**(원본 상품명이 상태
접미사를 뺀 형태로 정확히 일치하는지) 어종 단위 키로 결합한다. 검증에 실패하면 결합하지
않는다.

결합 키: ``조합코드 + 창고코드 + 기준일자 + 어종코드``

상품명은 **결합 키로 쓰지 않는다.** 같은 창고·날짜에서 서로 다른 어종코드가 같은 이름을
갖는 경우(붕장어·장어류)가 실제로 존재해 이름 키는 모호해지기 때문이다. 정규화된 상품명은
계보(lineage)와 검증에만 사용한다.

이 모듈은 추천 알고리즘(VHS/Greedy/DQN/MILP/실행계획)과 완전히 분리되어 있고 raw 파일을
읽기 전용으로만 다룬다.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# 버전과 상수
# --------------------------------------------------------------------------- #

#: 상품명 정규화 규칙이 바뀌면 반드시 올린다(과거 산출물과 구분하기 위해).
NORMALIZATION_VERSION = "suhyup-warehouse-name-normalize/1.0.0"
#: 결합 규칙(키·집계·판정) 버전.
JOIN_VERSION = "suhyup-warehouse-species-code-join/1.0.0"

DATASET_ID = "03_Korea_Suhyup_Warehouse"

#: flow 원본 컬럼(해양수산부_수협조합창고품목별창고입출고현황).
FLOW_COLUMNS = ("표준코드", "조합코드", "창고코드", "입출고구분", "기준일자",
                "조합명", "창고명", "표준코드명", "입출고구분명", "수량")
#: stock 원본 컬럼(해양수산부_수협조합창고품목별창고재고현황).
STOCK_COLUMNS = ("조합코드", "창고코드", "표준어종코드", "기준일자",
                 "조합명", "창고명", "표준어종명", "수량")

FLOW_FILE_KEYWORDS = ("조합창고", "입출고")
STOCK_FILE_KEYWORDS = ("조합창고", "재고")

#: flow 표준코드에서 어종 코드가 차지하는 앞자리 수. 나머지는 상태 자릿수다.
SPECIES_CODE_LENGTH = 6

#: 원본 상품명 끝에 실제로 관측된 상태 표기. 이름 계층 검증에서만 쓰고 결합에는 쓰지 않는다.
#: 여기에 없는 새로운 표기가 들어오면 stem이 그대로 남아 검증이 "불일치"로 떨어진다.
#: 즉 모르는 값을 조용히 통과시키지 않고 안전한 방향으로 실패한다.
OBSERVED_STATE_LABELS = ("냉장/신선", "냉동", "염장", "활", "건")

INBOUND_LABEL = "입고"
OUTBOUND_LABEL = "출고"

#: 재고 수지 진단에서 0으로 볼 부동소수 오차 한계. 원본에 소수점 수량(예: 408.2)이 있어
#: 뺄셈 결과에 1e-14 수준의 표현 오차가 남는다. 실제 불일치와 구분하기 위한 값이다.
BALANCE_EPSILON = 1e-6

MATCH_MATCHED = "matched"
MATCH_STOCK_ONLY = "stock_only_no_flow_record"
MATCH_UNMATCHED_PRODUCT = "unmatched_product"
MATCH_UNMATCHED_LOCATION = "unmatched_location"
MATCH_UNMATCHED_DATE = "unmatched_date"
MATCH_AMBIGUOUS = "ambiguous_key"

PROVENANCE_ACTUAL = "actual"
PROVENANCE_ASSUMED = "assumed"
PROVENANCE_NOT_AVAILABLE = "not_available"

STATUS_USABLE = "사용 가능"
STATUS_REVIEW = "검토 필요"
STATUS_UNUSABLE = "사용 불가"

JOIN_KEY_COLUMNS = ("coop_code", "warehouse_code", "date", "species_code")


# --------------------------------------------------------------------------- #
# 원본 위치 탐색 (프로젝트 밖 실데이터)
# --------------------------------------------------------------------------- #

def _candidate_real_data_roots(base_dir: Path) -> list[Path]:
    """``VARO_V2_REAL_DATA`` 폴더 후보를 순서대로 돌려준다.

    ``VARO_REAL_DATA_DIR`` 환경변수가 있으면 그 경로만 쓴다(경로일 뿐 비밀값이 아니다).
    없으면 프로젝트 주변과 Desktop/OneDrive Desktop을 훑는다. Windows 알려진 폴더
    이동(Known Folder Move)으로 Desktop이 OneDrive로 옮겨갈 수 있어 둘 다 본다.
    """
    override = os.environ.get("VARO_REAL_DATA_DIR")
    if override:
        path = Path(override)
        return [path.resolve()] if path.is_dir() else []

    candidates: list[Path] = [base_dir / "real_data", base_dir.parent / "VARO_V2_REAL_DATA"]
    home = Path.home()
    desktop_roots = [home / "Desktop", home / "OneDrive" / "Desktop"]
    onedrive = os.environ.get("OneDrive") or os.environ.get("ONEDRIVE")
    if onedrive:
        desktop_roots.append(Path(onedrive) / "Desktop")
    for desktop in desktop_roots:
        candidates.append(desktop / "VARO_V2_REAL_DATA")
        candidates.append(desktop / "Projects" / "VARO_V2_REAL_DATA")

    seen: set[Path] = set()
    existing: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            existing.append(resolved)
    return existing


def real_data_root(base_dir: Path | None = None) -> Path | None:
    """실데이터 루트를 찾는다. 없으면 ``None`` (파일을 만들지 않는다)."""
    base = base_dir or Path(__file__).resolve().parents[1]
    roots = _candidate_real_data_roots(base)
    return roots[0] if roots else None


@dataclass(frozen=True)
class WarehouseSources:
    """조합창고 원본 두 파일과 산출물 폴더."""

    flow_path: Path
    stock_path: Path
    processed_dir: Path


def _pick_raw_file(raw_dir: Path, keywords: tuple[str, ...], exclude: str) -> Path | None:
    matches = [
        path for path in sorted(raw_dir.glob("*"))
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and all(word in path.name for word in keywords)
        and exclude not in path.name
    ]
    return matches[0] if matches else None


def locate_warehouse_sources(root: Path | None = None, base_dir: Path | None = None) -> WarehouseSources | None:
    """조합창고 raw 두 파일을 찾는다. 파일명을 추측해 만들지 않고 실제 존재만 확인한다."""
    data_root = root or real_data_root(base_dir)
    if data_root is None:
        return None
    dataset_dir = Path(data_root) / DATASET_ID
    raw_dir = dataset_dir / "raw"
    if not raw_dir.is_dir():
        return None
    flow_path = _pick_raw_file(raw_dir, FLOW_FILE_KEYWORDS, exclude="재고")
    stock_path = _pick_raw_file(raw_dir, STOCK_FILE_KEYWORDS, exclude="입출고")
    if flow_path is None or stock_path is None:
        return None
    return WarehouseSources(flow_path, stock_path, dataset_dir / "processed")


# --------------------------------------------------------------------------- #
# 원본 읽기 (읽기 전용)
# --------------------------------------------------------------------------- #

#: 공공데이터 CSV에서 실제로 관측되는 인코딩. utf-8 계열을 먼저 시도해야
#: utf-8 파일이 cp949로 잘못 해석되는 일이 없다.
_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


def detect_encoding(data: bytes) -> str:
    """전체 바이트를 실제로 디코딩해 인코딩을 정한다(앞부분만 보고 판정하지 않는다).

    ``utf-8-sig``는 BOM이 없어도 성공하므로 BOM 유무는 바이트로 먼저 가른다.
    """
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    for encoding in _ENCODING_CANDIDATES:
        if encoding == "utf-8-sig":
            continue
        try:
            data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        return encoding
    raise ValueError("지원하는 인코딩으로 읽을 수 없는 파일입니다.")


def read_raw_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """원본 CSV를 문자열 그대로 읽는다. 값을 바꾸지 않고 파일도 건드리지 않는다.

    ``source_row``(스프레드시트 1-based 행번호, 헤더=1)를 계보용으로만 덧붙인다.
    """
    payload = Path(path).read_bytes()
    encoding = detect_encoding(payload)
    import io

    frame = pd.read_csv(io.BytesIO(payload), encoding=encoding, dtype=str, keep_default_na=False, na_values=[""])
    frame = frame.reset_index(drop=True)
    frame["source_row"] = frame.index + 2
    return frame, encoding


def file_digest(path: Path) -> str:
    """raw 파일이 실행 전후로 바뀌지 않았음을 확인하기 위한 SHA-256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 상품명 정규화 (deterministic)
# --------------------------------------------------------------------------- #

# C0/C1 제어문자와 zero-width·BOM 계열. 일반 공백류는 여기서 지우지 않고 아래에서 하나로 줄인다.
_CONTROL_CHARS_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\ufeff]"
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_product_name(value: Any) -> str | None:
    """표기 차이만 걷어내는 결정적 정규화.

    하는 일은 다음뿐이다.

    1. 유니코드 NFKC 정규화(전각/반각 통일)
    2. 의미 없는 제어문자·zero-width 문자 제거
    3. 연속 공백을 하나로, 앞뒤 공백 제거

    **하지 않는 일**: 숫자·용량·규격·등급·원산지·품종·상태 표기 제거, 유사어 병합.
    서로 다른 실제 상품을 같은 것으로 만들지 않는다. 결과가 빈 문자열이면 ``None``.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value)
    if text == "" or text.lower() == "nan":
        return None
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def split_state_suffix(normalized_name: str | None) -> tuple[str | None, str | None]:
    """검증 전용: 관측된 상태 표기가 끝에 붙어 있으면 (어종명, 상태) 로 나눈다.

    결합 키를 만드는 함수가 아니다. flow 상품명이 stock 어종명 + 상태로 이루어져 있다는
    가설을 **데이터로 확인**하기 위해서만 쓴다. 모르는 표기는 떼지 않으므로 검증이
    통과하지 못하고, 그 경우 결합은 차단된다.
    """
    if normalized_name is None or not isinstance(normalized_name, str) or not normalized_name:
        return None, None
    for label in OBSERVED_STATE_LABELS:
        suffix = f"({label})"
        if normalized_name.endswith(suffix):
            stem = normalized_name[: -len(suffix)].strip()
            return (stem or None), label
    return normalized_name, None


def species_code_from_standard_code(code: Any) -> str | None:
    """flow ``표준코드``(8자리)에서 어종코드(앞 6자리)를 뽑는다. 원본 코드는 그대로 둔다."""
    if code is None:
        return None
    if isinstance(code, float) and pd.isna(code):
        return None
    text = str(code).strip()
    if len(text) < SPECIES_CODE_LENGTH:
        return None
    return text[:SPECIES_CODE_LENGTH]


# --------------------------------------------------------------------------- #
# 표준화된 작업 프레임
# --------------------------------------------------------------------------- #

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_flow(raw_flow: pd.DataFrame) -> pd.DataFrame:
    """flow 원본을 표준 컬럼으로 옮긴다. 원본 값은 지우지 않고 함께 남긴다."""
    missing = [name for name in FLOW_COLUMNS if name not in raw_flow.columns]
    if missing:
        raise ValueError(f"입출고 원본에 필요한 컬럼이 없습니다: {missing}")
    frame = pd.DataFrame({
        "source_row": raw_flow["source_row"].astype(int),
        "coop_code": raw_flow["조합코드"],
        "warehouse_code": raw_flow["창고코드"],
        "date": raw_flow["기준일자"],
        "coop_name": raw_flow["조합명"],
        "warehouse_name": raw_flow["창고명"],
        "flow_standard_code": raw_flow["표준코드"],
        "flow_product_name_raw": raw_flow["표준코드명"],
        "direction_code": raw_flow["입출고구분"],
        "direction_name": raw_flow["입출고구분명"],
        "qty_raw": raw_flow["수량"],
    })
    frame["species_code"] = frame["flow_standard_code"].map(species_code_from_standard_code)
    frame["flow_product_name_norm"] = frame["flow_product_name_raw"].map(normalize_product_name)
    stems = frame["flow_product_name_norm"].map(split_state_suffix)
    frame["flow_species_name_stem"] = [item[0] for item in stems]
    frame["flow_state_label"] = [item[1] for item in stems]
    frame["qty"] = _numeric(frame["qty_raw"])
    frame["date_valid"] = frame["date"].fillna("").astype(str).map(lambda value: bool(_DATE_RE.match(value)))
    return frame


def prepare_stock(raw_stock: pd.DataFrame) -> pd.DataFrame:
    """stock 원본을 표준 컬럼으로 옮긴다. 어종코드는 원본을 그대로 쓴다."""
    missing = [name for name in STOCK_COLUMNS if name not in raw_stock.columns]
    if missing:
        raise ValueError(f"재고 원본에 필요한 컬럼이 없습니다: {missing}")
    frame = pd.DataFrame({
        "source_row": raw_stock["source_row"].astype(int),
        "coop_code": raw_stock["조합코드"],
        "warehouse_code": raw_stock["창고코드"],
        "date": raw_stock["기준일자"],
        "coop_name": raw_stock["조합명"],
        "warehouse_name": raw_stock["창고명"],
        "species_code": raw_stock["표준어종코드"],
        "species_name_raw": raw_stock["표준어종명"],
        "qty_raw": raw_stock["수량"],
    })
    frame["species_name_norm"] = frame["species_name_raw"].map(normalize_product_name)
    frame["qty"] = _numeric(frame["qty_raw"])
    frame["date_valid"] = frame["date"].fillna("").astype(str).map(lambda value: bool(_DATE_RE.match(value)))
    return frame


# --------------------------------------------------------------------------- #
# 코드 계층 검증
# --------------------------------------------------------------------------- #

@dataclass
class HierarchyReport:
    """flow 표준코드 앞 6자리가 stock 어종코드와 같은 체계인지에 대한 증거."""

    exact_full_code_overlap: int = 0
    flow_species_codes: int = 0
    stock_species_codes: int = 0
    shared_species_codes: int = 0
    comparable_pairs: int = 0
    name_agreements: int = 0
    name_mismatches: int = 0
    mismatch_examples: list[dict[str, Any]] = field(default_factory=list)
    flow_species_codes_absent_in_stock: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """비교 가능한 쌍이 하나라도 있고, 이름이 전부 일치할 때만 검증된 것으로 본다."""
        return self.comparable_pairs > 0 and self.name_mismatches == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_full_code_overlap": self.exact_full_code_overlap,
            "flow_species_codes": self.flow_species_codes,
            "stock_species_codes": self.stock_species_codes,
            "shared_species_codes": self.shared_species_codes,
            "comparable_pairs": self.comparable_pairs,
            "name_agreements": self.name_agreements,
            "name_mismatches": self.name_mismatches,
            "mismatch_examples": self.mismatch_examples[:20],
            "flow_species_codes_absent_in_stock": sorted(self.flow_species_codes_absent_in_stock)[:20],
            "verified": self.verified,
        }


def verify_species_hierarchy(flow: pd.DataFrame, stock: pd.DataFrame) -> HierarchyReport:
    """flow 코드 앞 6자리 == stock 어종코드 가설을 원본 이름으로 검증한다.

    같은 어종코드에 대해 flow 상품명에서 상태 표기를 뺀 부분이 stock 어종명과 글자 그대로
    같아야 한다. 하나라도 어긋나면 결합을 진행하지 않는다.
    """
    report = HierarchyReport()
    flow_codes = set(flow["flow_standard_code"].dropna().astype(str))
    stock_codes = set(stock["species_code"].dropna().astype(str))
    report.exact_full_code_overlap = len(flow_codes & stock_codes)

    species_names = (
        stock.dropna(subset=["species_code"])
        .drop_duplicates(subset=["species_code"])
        .set_index("species_code")["species_name_norm"]
        .to_dict()
    )
    flow_species = set(flow["species_code"].dropna().astype(str))
    report.flow_species_codes = len(flow_species)
    report.stock_species_codes = len(stock_codes)
    report.shared_species_codes = len(flow_species & stock_codes)
    report.flow_species_codes_absent_in_stock = sorted(flow_species - stock_codes)

    pairs = flow.dropna(subset=["species_code"])[
        ["species_code", "flow_standard_code", "flow_product_name_raw", "flow_species_name_stem"]
    ].drop_duplicates()
    for row in pairs.itertuples(index=False):
        expected = species_names.get(row.species_code)
        if expected is None or row.flow_species_name_stem is None:
            continue
        report.comparable_pairs += 1
        if row.flow_species_name_stem == expected:
            report.name_agreements += 1
        else:
            report.name_mismatches += 1
            if len(report.mismatch_examples) < 20:
                report.mismatch_examples.append({
                    "species_code": row.species_code,
                    "flow_standard_code": row.flow_standard_code,
                    "flow_product_name": row.flow_product_name_raw,
                    "flow_species_name_stem": row.flow_species_name_stem,
                    "stock_species_name": expected,
                })
    return report


# --------------------------------------------------------------------------- #
# 키 유일성 / 관계 분류
# --------------------------------------------------------------------------- #

def key_uniqueness(frame: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
    """행 수·유일키·중복키·중복행·최대 다중도를 센다."""
    if frame.empty:
        return {"rows": 0, "unique_keys": 0, "duplicate_keys": 0,
                "duplicate_rows": 0, "max_multiplicity": 0}
    sizes = frame.groupby(keys, dropna=False).size()
    duplicated = sizes[sizes > 1]
    return {
        "rows": int(len(frame)),
        "unique_keys": int(len(sizes)),
        "duplicate_keys": int(len(duplicated)),
        "duplicate_rows": int(duplicated.sum()),
        "max_multiplicity": int(sizes.max()),
    }


def classify_key_cardinality(left_sizes: dict[Any, int], right_sizes: dict[Any, int]) -> dict[str, int]:
    """공통 키를 1:1 / 1:N / N:1 / N:M 으로 나눈다(left=flow, right=stock)."""
    counts = {"one_to_one": 0, "one_to_many": 0, "many_to_one": 0, "many_to_many": 0}
    for key, left in left_sizes.items():
        right = right_sizes.get(key)
        if right is None:
            continue
        if left == 1 and right == 1:
            counts["one_to_one"] += 1
        elif left == 1 and right > 1:
            counts["one_to_many"] += 1
        elif left > 1 and right == 1:
            counts["many_to_one"] += 1
        else:
            counts["many_to_many"] += 1
    return counts


# --------------------------------------------------------------------------- #
# flow 일별 집계
# --------------------------------------------------------------------------- #

def _join_unique(values: pd.Series) -> str:
    items = sorted({str(value) for value in values.dropna()})
    return "|".join(items)


def _join_rows(values: pd.Series) -> str:
    return "|".join(str(int(value)) for value in sorted(values.dropna()))


def aggregate_flow(flow: pd.DataFrame) -> pd.DataFrame:
    """flow를 결합 키 단위로 모은다. 입고와 출고를 절대 섞지 않는다.

    같은 조합·창고·날짜·어종에 여러 event가 있을 수 있으므로 방향별로 따로 합산한다.
    실제 기록이 없는 방향은 0이 아니라 결측으로 남긴다(없는 값을 만들지 않는다).
    """
    usable = flow.dropna(subset=["species_code"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=[
            *JOIN_KEY_COLUMNS, "coop_name", "warehouse_name",
            "inbound_qty_actual", "outbound_qty_actual",
            "inbound_record_count", "outbound_record_count",
            "flow_standard_codes", "flow_standard_code_count",
            "flow_product_names", "flow_product_names_normalized",
            "flow_state_labels", "flow_source_rows", "flow_qty_invalid_count",
        ])
    keys = list(JOIN_KEY_COLUMNS)
    grouped = usable.groupby(keys, dropna=False)

    inbound = usable[usable["direction_name"] == INBOUND_LABEL].groupby(keys, dropna=False)["qty"]
    outbound = usable[usable["direction_name"] == OUTBOUND_LABEL].groupby(keys, dropna=False)["qty"]

    aggregated = grouped.agg(
        coop_name=("coop_name", "first"),
        warehouse_name=("warehouse_name", "first"),
        flow_standard_codes=("flow_standard_code", _join_unique),
        flow_standard_code_count=("flow_standard_code", "nunique"),
        flow_product_names=("flow_product_name_raw", _join_unique),
        flow_product_names_normalized=("flow_product_name_norm", _join_unique),
        flow_state_labels=("flow_state_label", _join_unique),
        flow_source_rows=("source_row", _join_rows),
        flow_qty_invalid_count=("qty", lambda values: int(values.isna().sum())),
    )
    aggregated["inbound_qty_actual"] = inbound.sum(min_count=1)
    aggregated["outbound_qty_actual"] = outbound.sum(min_count=1)
    aggregated["inbound_record_count"] = inbound.size().reindex(aggregated.index).fillna(0).astype(int)
    aggregated["outbound_record_count"] = outbound.size().reindex(aggregated.index).fillna(0).astype(int)
    return aggregated.reset_index()


# --------------------------------------------------------------------------- #
# 안전 결합
# --------------------------------------------------------------------------- #

@dataclass
class JoinResult:
    matched: pd.DataFrame
    metrics: dict[str, Any]
    hierarchy: HierarchyReport
    blockers: list[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return not self.blockers


def _match_reason(row: Any, stock_locations: set[tuple[str, str]], stock_location_dates: set[tuple[str, str, str]],
                  stock_species: set[str]) -> str:
    """flow-only 키가 왜 결합되지 않았는지 위치 → 날짜 → 상품 순으로 판정한다."""
    if (row.coop_code, row.warehouse_code) not in stock_locations:
        return MATCH_UNMATCHED_LOCATION
    if (row.coop_code, row.warehouse_code, row.date) not in stock_location_dates:
        return MATCH_UNMATCHED_DATE
    if row.species_code not in stock_species:
        return MATCH_UNMATCHED_PRODUCT
    return MATCH_AMBIGUOUS


def join_stock_flow(flow: pd.DataFrame, stock: pd.DataFrame) -> JoinResult:
    """검증을 통과한 경우에만 어종 단위로 재고와 입출고를 결합한다.

    결합 전에 코드 계층을 검증하고, stock 쪽 중복키는 자동으로 고르지 않고
    ``ambiguous_key``로 남긴다. 결과 행 수는 항상 설명 가능하다.
    """
    hierarchy = verify_species_hierarchy(flow, stock)
    keys = list(JOIN_KEY_COLUMNS)

    stock_key_stats = key_uniqueness(stock, keys)
    flow_event_stats = key_uniqueness(
        flow.dropna(subset=["species_code"]), keys + ["flow_standard_code", "direction_code"]
    )
    flow_species_stats = key_uniqueness(flow.dropna(subset=["species_code"]), keys)

    # stock 중복키: 어느 값도 임의로 고르지 않는다.
    stock_sizes = stock.groupby(keys, dropna=False).size()
    ambiguous_keys = set(stock_sizes[stock_sizes > 1].index)
    stock_index = pd.MultiIndex.from_frame(stock[keys])
    stock_ambiguous = stock[stock_index.isin(ambiguous_keys)].copy()
    stock_unique = stock[~stock_index.isin(ambiguous_keys)].copy()

    flow_aggregated = aggregate_flow(flow)
    flow_sizes = flow_aggregated.groupby(keys, dropna=False).size().to_dict()
    stock_unique_sizes = stock_unique.groupby(keys, dropna=False).size().to_dict()
    cardinality = classify_key_cardinality(flow_sizes, stock_unique_sizes)

    blockers: list[str] = []
    if not hierarchy.verified:
        blockers.append(
            "flow 표준코드 앞 6자리와 stock 어종코드가 같은 체계임을 상품명으로 확인하지 못했습니다."
        )
    if cardinality["many_to_many"]:
        blockers.append(f"통제되지 않은 N:M 키가 {cardinality['many_to_many']}건 있습니다.")
    if flow_aggregated.duplicated(subset=keys).any():
        blockers.append("집계 후에도 flow 키가 유일하지 않습니다.")

    stock_side = stock_unique.rename(columns={
        "source_row": "stock_source_row",
        "qty": "stock_qty_actual",
        "qty_raw": "stock_qty_raw",
        "date_valid": "stock_date_valid",
    })
    merged = stock_side.merge(
        flow_aggregated.rename(columns={"coop_name": "flow_coop_name",
                                        "warehouse_name": "flow_warehouse_name"}),
        on=keys, how="outer", indicator="merge_side", validate="one_to_one",
    )
    # 결합되지 않은 flow 키도 위치 이름을 잃지 않도록 flow 쪽 값으로 채운다(원본 값 그대로).
    for column, fallback in (("coop_name", "flow_coop_name"),
                             ("warehouse_name", "flow_warehouse_name")):
        merged[column] = merged[column].fillna(merged[fallback])
    merged = merged.drop(columns=["flow_coop_name", "flow_warehouse_name"])

    stock_locations = set(zip(stock["coop_code"], stock["warehouse_code"]))
    stock_location_dates = set(zip(stock["coop_code"], stock["warehouse_code"], stock["date"]))
    stock_species = set(stock["species_code"].dropna().astype(str))

    statuses: list[str] = []
    for row in merged.itertuples(index=False):
        if row.merge_side == "both":
            statuses.append(MATCH_MATCHED)
        elif row.merge_side == "left_only":
            statuses.append(MATCH_STOCK_ONLY)
        else:
            statuses.append(_match_reason(row, stock_locations, stock_location_dates, stock_species))
    merged["match_status"] = statuses
    # stock에 짝이 없는 flow 키만 출력 행을 늘린다. 그 수를 따로 세어 행 수를 설명한다.
    flow_only_rows = int(merged["merge_side"].eq("right_only").sum())
    merged = merged.drop(columns=["merge_side"])

    if not stock_ambiguous.empty:
        extra = stock_ambiguous.rename(columns={
            "source_row": "stock_source_row",
            "qty": "stock_qty_actual",
            "qty_raw": "stock_qty_raw",
            "date_valid": "stock_date_valid",
        })
        extra["match_status"] = MATCH_AMBIGUOUS
        merged = pd.concat([merged, extra], ignore_index=True)

    expected_rows = len(stock) + flow_only_rows
    if len(merged) != expected_rows:
        blockers.append(f"결합 결과 행 수({len(merged)})가 설명되지 않습니다(기대 {expected_rows}).")

    status_counts = merged["match_status"].value_counts().to_dict()
    matched_count = int(status_counts.get(MATCH_MATCHED, 0))
    flow_keys_total = int(len(flow_aggregated))
    metrics = {
        "flow_source_rows": int(len(flow)),
        "stock_source_rows": int(len(stock)),
        "flow_event_key_uniqueness": flow_event_stats,
        "flow_species_key_uniqueness": flow_species_stats,
        "stock_key_uniqueness": stock_key_stats,
        "flow_aggregated_keys": flow_keys_total,
        "stock_keys": int(len(stock_unique_sizes)),
        "common_keys": int(len(set(flow_sizes) & set(stock_unique_sizes))),
        "flow_matched_keys": matched_count,
        "flow_unmatched_keys": flow_keys_total - matched_count,
        "stock_matched_rows": matched_count,
        "stock_unmatched_rows": int(status_counts.get(MATCH_STOCK_ONLY, 0)),
        "ambiguous_rows": int(status_counts.get(MATCH_AMBIGUOUS, 0)),
        "join_rate_flow_side": round(matched_count / flow_keys_total, 6) if flow_keys_total else 0.0,
        "join_rate_stock_side": round(matched_count / len(stock), 6) if len(stock) else 0.0,
        "cardinality": cardinality,
        "row_expansion_factor": round(len(merged) / len(stock), 6) if len(stock) else 0.0,
        "output_rows": int(len(merged)),
        "match_status_counts": {str(key): int(value) for key, value in status_counts.items()},
    }
    return JoinResult(matched=merged, metrics=metrics, hierarchy=hierarchy, blockers=blockers)


# --------------------------------------------------------------------------- #
# 재고 수지 진단 · 이상치
# --------------------------------------------------------------------------- #

def add_analysis_fields(joined: pd.DataFrame) -> pd.DataFrame:
    """분석용 파생값과 flag를 붙인다. 실측값은 그대로 두고 별도 컬럼으로만 만든다."""
    frame = joined.copy()
    frame["flow_record_available"] = frame["match_status"].eq(MATCH_MATCHED)
    for direction in ("inbound", "outbound"):
        actual = f"{direction}_qty_actual"
        if actual not in frame.columns:
            frame[actual] = pd.NA
        values = pd.to_numeric(frame[actual], errors="coerce")
        # 기록이 없는 방향은 분석용 컬럼에서만 0으로 채운다. 실측 컬럼은 결측으로 남는다.
        frame[f"{direction}_qty_for_analysis"] = values.fillna(0.0)
        frame[f"{direction}_qty_zero_filled"] = values.isna()
    frame["flow_zero_fill_applied"] = (
        frame["inbound_qty_zero_filled"] | frame["outbound_qty_zero_filled"]
    )

    stock_qty = pd.to_numeric(frame.get("stock_qty_actual"), errors="coerce")
    frame["stock_qty_missing"] = stock_qty.isna()
    frame["stock_qty_zero"] = stock_qty.eq(0)
    frame["stock_qty_negative"] = stock_qty.lt(0)
    inbound = pd.to_numeric(frame.get("inbound_qty_actual"), errors="coerce")
    outbound = pd.to_numeric(frame.get("outbound_qty_actual"), errors="coerce")
    frame["inbound_qty_negative"] = inbound.lt(0)
    frame["outbound_qty_negative"] = outbound.lt(0)
    frame["flow_qty_nonpositive"] = inbound.le(0) | outbound.le(0)
    frame["product_name_missing"] = frame.get(
        "species_name_norm", pd.Series([None] * len(frame))
    ).isna() & frame.get(
        "flow_product_names_normalized", pd.Series([""] * len(frame))
    ).fillna("").eq("")
    frame["location_missing"] = frame["coop_code"].isna() | frame["warehouse_code"].isna()
    date_valid = frame.get("stock_date_valid")
    if date_valid is None:
        date_valid = pd.Series([True] * len(frame), index=frame.index)
    frame["date_invalid"] = ~date_valid.fillna(True).astype(bool)
    return frame


def add_balance_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    """재고 수지 참고 진단: ``재고 변화 ≈ 입고 − 출고``.

    hard validation이 아니다. 전날 재고가 없는 계열 첫 행이나 날짜가 끊긴 구간은
    ``no_prior_day``로 남기고 판정하지 않는다.
    """
    result = frame.copy()
    parsed = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="coerce")
    result["_date_parsed"] = parsed
    result = result.sort_values(["coop_code", "warehouse_code", "species_code", "_date_parsed"],
                                kind="mergesort")
    grouped = result.groupby(["coop_code", "warehouse_code", "species_code"], dropna=False)
    result["stock_qty_prev_day"] = grouped["stock_qty_actual"].shift(1)
    previous_date = grouped["_date_parsed"].shift(1)
    consecutive = (result["_date_parsed"] - previous_date).dt.days.eq(1)

    result["stock_delta"] = pd.to_numeric(result["stock_qty_actual"], errors="coerce") - pd.to_numeric(
        result["stock_qty_prev_day"], errors="coerce")
    result["net_flow"] = result["inbound_qty_for_analysis"] - result["outbound_qty_for_analysis"]
    result["balance_gap"] = result["stock_delta"] - result["net_flow"]

    status = pd.Series(["no_prior_day"] * len(result), index=result.index, dtype=object)
    comparable = consecutive & result["balance_gap"].notna()
    status[comparable & result["balance_gap"].abs().le(BALANCE_EPSILON)] = "consistent"
    status[comparable & result["balance_gap"].abs().gt(BALANCE_EPSILON)] = "gap"
    status[~comparable & result["match_status"].eq(MATCH_AMBIGUOUS)] = "not_applicable"
    result["balance_status"] = status
    result.loc[~comparable, ["stock_delta", "balance_gap"]] = pd.NA
    return result.drop(columns=["_date_parsed"])


def balance_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """재고 수지 진단 요약. 재고 snapshot 시점 해석의 근거로만 쓴다."""
    status = frame["balance_status"]
    comparable = status.isin(("consistent", "gap"))
    total = int(comparable.sum())
    consistent = int(status.eq("consistent").sum())
    moved = comparable & frame["flow_record_available"]
    still = comparable & ~frame["flow_record_available"]
    return {
        "comparable_day_pairs": total,
        "consistent": consistent,
        "gap": int(status.eq("gap").sum()),
        "no_prior_day": int(status.eq("no_prior_day").sum()),
        "consistent_rate": round(consistent / total, 6) if total else 0.0,
        "pairs_with_flow_record": int(moved.sum()),
        "pairs_with_flow_record_consistent": int((moved & status.eq("consistent")).sum()),
        "pairs_without_flow_record": int(still.sum()),
        "pairs_without_flow_record_consistent": int((still & status.eq("consistent")).sum()),
        "max_abs_gap": float(frame.loc[comparable, "balance_gap"].abs().max()) if total else 0.0,
    }


def anomaly_summary(frame: pd.DataFrame) -> dict[str, int]:
    flags = ("stock_qty_missing", "stock_qty_zero", "stock_qty_negative",
             "inbound_qty_negative", "outbound_qty_negative", "flow_qty_nonpositive",
             "product_name_missing", "location_missing", "date_invalid")
    return {flag: int(frame[flag].fillna(False).astype(bool).sum()) for flag in flags if flag in frame}


# --------------------------------------------------------------------------- #
# 산출 프레임 구성
# --------------------------------------------------------------------------- #

OUTPUT_COLUMNS = [
    # 결합 키
    "coop_code", "warehouse_code", "date", "species_code",
    # 위치·상품 표시명
    "coop_name", "warehouse_name", "species_name", "species_name_normalized",
    # 실제 관측값
    "stock_qty_actual", "inbound_qty_actual", "outbound_qty_actual",
    "inbound_record_count", "outbound_record_count",
    # 분석용 파생값
    "inbound_qty_for_analysis", "outbound_qty_for_analysis",
    "inbound_qty_zero_filled", "outbound_qty_zero_filled",
    "flow_record_available", "flow_zero_fill_applied",
    # 결합 metadata
    "match_status", "species_code_source", "species_name_consistency",
    "flow_standard_codes", "flow_standard_code_count", "flow_state_labels",
    "flow_product_names", "flow_product_names_normalized",
    # 재고 수지 참고 진단
    "stock_qty_prev_day", "stock_delta", "net_flow", "balance_gap", "balance_status",
    # 이상치 flag
    "stock_qty_missing", "stock_qty_zero", "stock_qty_negative",
    "inbound_qty_negative", "outbound_qty_negative", "flow_qty_nonpositive",
    "product_name_missing", "location_missing", "date_invalid",
    # 계보
    "source_dataset", "stock_source_file", "stock_source_row",
    "flow_source_file", "flow_source_rows", "stock_qty_raw",
    # provenance
    "inventory_provenance", "inbound_provenance", "outbound_provenance",
    "flow_zero_fill_provenance", "weight_kg_provenance",
    "interwarehouse_transfer_provenance", "distance_provenance",
    "transfer_cost_provenance", "vehicle_capacity_provenance",
]


def build_output(frame: pd.DataFrame, flow_file: str, stock_file: str) -> pd.DataFrame:
    """최종 processed 프레임. 실측값·파생값·결합 metadata의 의미를 컬럼으로 분리한다."""
    result = frame.copy()
    result["species_name"] = result.get("species_name_raw")
    result["species_name_normalized"] = result.get("species_name_norm")
    # 재고 행이 있으면 어종코드는 stock 원본 값이고, flow만 있는 행은 표준코드에서 유도한 값이다.
    from_stock = result["stock_source_row"].notna() if "stock_source_row" in result else pd.Series(
        [False] * len(result), index=result.index)
    result["species_code_source"] = [
        "stock_standard_species_code" if flag else "derived_from_flow_standard_code_prefix"
        for flag in from_stock
    ]
    # 이름 일치 여부는 결합된 행에서만 판정 가능하다.
    stems = result.get("flow_product_names_normalized")
    consistency: list[str] = []
    for status, names, species_name in zip(result["match_status"], stems.fillna("") if stems is not None
                                           else [""] * len(result),
                                           result["species_name_normalized"].fillna("")):
        if status != MATCH_MATCHED or not names or not species_name:
            consistency.append("not_comparable")
            continue
        stripped = {split_state_suffix(name)[0] for name in str(names).split("|") if name}
        consistency.append("verified_identical" if stripped == {species_name} else "mismatch")
    result["species_name_consistency"] = consistency

    result["source_dataset"] = DATASET_ID
    result["stock_source_file"] = [stock_file if flag else "" for flag in from_stock]
    result["flow_source_file"] = [
        flow_file if isinstance(rows, str) and rows else ""
        for rows in result.get("flow_source_rows", pd.Series([""] * len(result))).fillna("")
    ]

    # provenance: 실제 관측된 값은 actual, 채워 넣은 값은 assumed, 없는 것은 not_available.
    inbound = pd.to_numeric(result.get("inbound_qty_actual"), errors="coerce")
    outbound = pd.to_numeric(result.get("outbound_qty_actual"), errors="coerce")
    stock_qty = pd.to_numeric(result.get("stock_qty_actual"), errors="coerce")
    result["inventory_provenance"] = [
        PROVENANCE_ACTUAL if pd.notna(value) else PROVENANCE_NOT_AVAILABLE for value in stock_qty
    ]
    result["inbound_provenance"] = [
        PROVENANCE_ACTUAL if pd.notna(value) else PROVENANCE_NOT_AVAILABLE for value in inbound
    ]
    result["outbound_provenance"] = [
        PROVENANCE_ACTUAL if pd.notna(value) else PROVENANCE_NOT_AVAILABLE for value in outbound
    ]
    result["flow_zero_fill_provenance"] = [
        PROVENANCE_ASSUMED if bool(applied) else PROVENANCE_ACTUAL
        for applied in result["flow_zero_fill_applied"].fillna(False)
    ]
    # 이 원본에는 중량(kg)과 거점간 이동 이력이 없다. 만들어내지 않는다.
    result["weight_kg_provenance"] = PROVENANCE_NOT_AVAILABLE
    result["interwarehouse_transfer_provenance"] = PROVENANCE_NOT_AVAILABLE
    result["distance_provenance"] = PROVENANCE_NOT_AVAILABLE
    result["transfer_cost_provenance"] = PROVENANCE_NOT_AVAILABLE
    result["vehicle_capacity_provenance"] = PROVENANCE_NOT_AVAILABLE

    for column in OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result = result[OUTPUT_COLUMNS]
    # 결정적 정렬: 입력 행 순서가 달라도 같은 결과가 나온다.
    return result.sort_values(
        ["coop_code", "warehouse_code", "species_code", "date", "match_status"],
        kind="mergesort",
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 품질 등급 · manifest
# --------------------------------------------------------------------------- #

def grade_dataset(metrics: dict[str, Any], hierarchy: HierarchyReport,
                  anomalies: dict[str, int], blockers: list[str]) -> tuple[str, list[str]]:
    """결합률 하나가 아니라 모호성·결측·중복·계보·수량 타당성을 함께 본다."""
    notes: list[str] = []
    if blockers:
        return STATUS_UNUSABLE, list(blockers)
    if not hierarchy.verified:
        return STATUS_UNUSABLE, ["코드 계층이 검증되지 않았습니다."]

    status = STATUS_USABLE
    if metrics["ambiguous_rows"]:
        status = STATUS_REVIEW
        notes.append(f"모호한 키 {metrics['ambiguous_rows']}행은 결합하지 않고 남겼습니다.")
    if metrics["cardinality"]["many_to_many"]:
        status = STATUS_REVIEW
        notes.append("N:M 키가 있습니다.")
    if metrics["flow_unmatched_keys"]:
        notes.append(f"결합되지 않은 flow 키 {metrics['flow_unmatched_keys']}건을 보존했습니다.")
    if anomalies.get("stock_qty_negative") or anomalies.get("inbound_qty_negative") \
            or anomalies.get("outbound_qty_negative"):
        status = STATUS_REVIEW
        notes.append("음수 수량이 있습니다(원본 값은 수정하지 않고 flag만 남겼습니다).")
    if anomalies.get("date_invalid"):
        status = STATUS_REVIEW
        notes.append("날짜 형식 오류가 있습니다.")
    if abs(metrics["row_expansion_factor"] - 1.0) > 0.05:
        status = STATUS_REVIEW
        notes.append("행 확장이 관측되었습니다.")
    return status, notes


def build_manifest(sources: WarehouseSources, encodings: dict[str, str], digests: dict[str, str],
                   metrics: dict[str, Any], hierarchy: HierarchyReport, balance: dict[str, Any],
                   anomalies: dict[str, int], scope: dict[str, Any], status: str,
                   notes: list[str], output_path: Path, name_key_comparison: dict[str, Any]) -> dict[str, Any]:
    """기계가 읽을 수 있는 결합 이력. 개인정보나 원본 전량은 담지 않는다."""
    return {
        "dataset_id": DATASET_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "normalization_version": NORMALIZATION_VERSION,
        "join_version": JOIN_VERSION,
        "raw_sources": [
            {"role": "flow", "file_name": sources.flow_path.name,
             "encoding": encodings.get("flow"), "sha256": digests.get("flow"),
             "rows": metrics["flow_source_rows"]},
            {"role": "stock", "file_name": sources.stock_path.name,
             "encoding": encodings.get("stock"), "sha256": digests.get("stock"),
             "rows": metrics["stock_source_rows"]},
        ],
        "join_key": list(JOIN_KEY_COLUMNS),
        "join_key_description": (
            "조합코드 + 창고코드 + 기준일자 + 어종코드. 어종코드는 stock 원본의 표준어종코드이고, "
            "flow 쪽은 표준코드 앞 6자리에서 유도한 뒤 원본 상품명으로 동일성을 검증했다. "
            "상품코드 전체 일치 join과 상품명 join은 사용하지 않는다."
        ),
        "product_code_relation": hierarchy.to_dict(),
        "key_metrics": metrics,
        "name_key_comparison": name_key_comparison,
        "stock_flow_balance_diagnostic": balance,
        "anomalies": anomalies,
        "scope": scope,
        "provenance": {
            "stock_qty": PROVENANCE_ACTUAL,
            "inbound_qty": PROVENANCE_ACTUAL,
            "outbound_qty": PROVENANCE_ACTUAL,
            "inbound_outbound_zero_fill": PROVENANCE_ASSUMED,
            "weight_kg": PROVENANCE_NOT_AVAILABLE,
            "interwarehouse_transfer_history": PROVENANCE_NOT_AVAILABLE,
            "distance": PROVENANCE_NOT_AVAILABLE,
            "travel_time": PROVENANCE_NOT_AVAILABLE,
            "transfer_cost": PROVENANCE_NOT_AVAILABLE,
            "vehicle_capacity": PROVENANCE_NOT_AVAILABLE,
        },
        "output": {"file_name": output_path.name, "rows": metrics["output_rows"],
                   "encoding": "utf-8-sig"},
        "validation_status": status,
        "validation_notes": notes,
        "limitations": [
            "입고/출고는 창고 단위 관측값이며 출발지-도착지가 없다. 거점간 실제 이동 이력이 아니다.",
            "재고는 어종 단위 snapshot이라 냉동/냉장 등 상태별 재고로 나눌 수 없다.",
            "이 원본에는 중량(kg) 컬럼이 없다. 수량 단위만 사용한다.",
        ],
    }


def compare_name_key(flow: pd.DataFrame, stock: pd.DataFrame) -> dict[str, Any]:
    """과거 진단에서 쓰던 '정규화 상품명' 키와 어종코드 키의 차이를 수치로 설명한다."""
    flow_named = flow.dropna(subset=["species_code", "flow_species_name_stem"])
    name_keys = flow_named[["coop_code", "warehouse_code", "date", "flow_species_name_stem"]].drop_duplicates()
    stock_name_keys = stock.dropna(subset=["species_name_norm"])[
        ["coop_code", "warehouse_code", "date", "species_name_norm"]].drop_duplicates()
    code_keys = flow.dropna(subset=["species_code"])[list(JOIN_KEY_COLUMNS)].drop_duplicates()

    missing_name_keys = len(
        flow[flow["species_code"].notna() & flow["flow_species_name_stem"].isna()][
            list(JOIN_KEY_COLUMNS)].drop_duplicates()
    )
    collapsed = flow_named.groupby(
        ["coop_code", "warehouse_code", "date", "flow_species_name_stem"], dropna=False
    )["species_code"].nunique()
    collisions = int((collapsed - 1)[collapsed > 1].sum())
    stock_name_dupes = key_uniqueness(
        stock.dropna(subset=["species_name_norm"]),
        ["coop_code", "warehouse_code", "date", "species_name_norm"],
    )
    return {
        "flow_species_code_keys": int(len(code_keys)),
        "flow_normalized_name_keys": int(len(name_keys)),
        "stock_normalized_name_keys": int(len(stock_name_keys)),
        "keys_dropped_for_missing_product_name": int(missing_name_keys),
        "keys_collapsed_by_name_collision": collisions,
        "stock_name_key_duplicate_keys": stock_name_dupes["duplicate_keys"],
        "explanation": (
            "어종코드 키 수 = 상품명 키 수 + 상품명이 없는 키 + 서로 다른 어종코드가 같은 이름으로 "
            "합쳐진 키. 상품명 키는 붕장어·장어류처럼 코드가 다른 어종을 한 키로 합쳐 모호해진다."
        ),
    }


def scope_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "coops": int(frame["coop_code"].nunique()),
        "warehouses": int(frame["warehouse_code"].nunique()),
        "species": int(frame["species_code"].nunique()),
        "dates": int(frame["date"].nunique()),
        "date_min": str(frame["date"].min()),
        "date_max": str(frame["date"].max()),
    }


# --------------------------------------------------------------------------- #
# 전체 파이프라인
# --------------------------------------------------------------------------- #

@dataclass
class WarehouseJoinReport:
    output: pd.DataFrame
    manifest: dict[str, Any]
    join: JoinResult
    unmatched: pd.DataFrame
    balance_gaps: pd.DataFrame

    @property
    def status(self) -> str:
        return str(self.manifest["validation_status"])


def build_warehouse_dataset(flow_raw: pd.DataFrame, stock_raw: pd.DataFrame,
                            sources: WarehouseSources,
                            encodings: dict[str, str] | None = None,
                            digests: dict[str, str] | None = None,
                            output_name: str = "") -> WarehouseJoinReport:
    """원본 두 프레임에서 결합본과 manifest를 만든다. 파일은 쓰지 않는다."""
    flow = prepare_flow(flow_raw)
    stock = prepare_stock(stock_raw)

    join = join_stock_flow(flow, stock)
    enriched = add_balance_diagnostic(add_analysis_fields(join.matched))
    output = build_output(enriched, sources.flow_path.name, sources.stock_path.name)

    anomalies = anomaly_summary(enriched)
    balance = balance_summary(enriched)
    scope = scope_summary(output)
    status, notes = grade_dataset(join.metrics, join.hierarchy, anomalies, join.blockers)
    output_path = sources.processed_dir / (output_name or DEFAULT_OUTPUT_NAME)
    manifest = build_manifest(
        sources, encodings or {}, digests or {}, join.metrics, join.hierarchy, balance,
        anomalies, scope, status, notes, output_path, compare_name_key(flow, stock),
    )
    unmatched = output[output["match_status"].isin(
        (MATCH_UNMATCHED_PRODUCT, MATCH_UNMATCHED_LOCATION, MATCH_UNMATCHED_DATE, MATCH_AMBIGUOUS)
    )].copy()
    balance_gaps = output[output["balance_status"].eq("gap")].copy()
    return WarehouseJoinReport(output, manifest, join, unmatched, balance_gaps)


DEFAULT_OUTPUT_NAME = "suhyup_warehouse_stock_flow_species_matched_actual.csv"
DEFAULT_MANIFEST_NAME = "suhyup_warehouse_stock_flow_species_matched_actual_manifest.json"
DEFAULT_UNMATCHED_NAME = "suhyup_warehouse_stock_flow_species_unmatched_review.csv"
DEFAULT_BALANCE_NAME = "suhyup_warehouse_stock_flow_balance_gaps_review.csv"

#: 상품코드를 직접 연결해 만들어진 과거 산출물. 분석 입력으로 쓰지 않는다.
SUPERSEDED_OUTPUTS = {
    "suhyup_warehouse_inventory_flow_actual.csv": {
        "status": "invalid_join_key",
        "reason": "8자리 표준코드와 6자리 표준어종코드를 같은 코드체계로 보고 직접 연결해 결합률 0%가 되었다.",
    },
    "suhyup_warehouse_inventory_flow_actual_final.csv": {
        "status": "superseded_ambiguous_name_key",
        "reason": (
            "정규화 상품명을 결합 키로 사용했다. 서로 다른 어종코드가 같은 이름을 갖는 경우"
            "(붕장어·장어류)를 한 키로 합쳐 모호한 결합이 남는다."
        ),
    },
}


def write_outputs(report: WarehouseJoinReport, processed_dir: Path,
                  overwrite: bool = False) -> dict[str, Path]:
    """processed 폴더에 결과를 쓴다. 기존 파일은 기본적으로 덮어쓰지 않는다."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "dataset": processed_dir / DEFAULT_OUTPUT_NAME,
        "manifest": processed_dir / DEFAULT_MANIFEST_NAME,
        "unmatched": processed_dir / DEFAULT_UNMATCHED_NAME,
        "balance_gaps": processed_dir / DEFAULT_BALANCE_NAME,
    }
    if not overwrite:
        existing = [path.name for path in targets.values() if path.exists()]
        if existing:
            raise FileExistsError(f"이미 있는 산출물을 덮어쓰지 않습니다: {existing}")
    # 기존 processed 파일은 그대로 두고 상태만 별도 파일로 기록한다.
    status_path = processed_dir / "suhyup_warehouse_processed_status.json"
    status_path.write_text(json.dumps({
        "generated_at": report.manifest["generated_at"],
        "valid_outputs": [DEFAULT_OUTPUT_NAME],
        "superseded_outputs": SUPERSEDED_OUTPUTS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    targets["status"] = status_path

    report.output.to_csv(targets["dataset"], index=False, encoding="utf-8-sig")
    report.unmatched.to_csv(targets["unmatched"], index=False, encoding="utf-8-sig")
    report.balance_gaps.to_csv(targets["balance_gaps"], index=False, encoding="utf-8-sig")
    targets["manifest"].write_text(
        json.dumps(report.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return targets


def run_from_sources(sources: WarehouseSources) -> tuple[WarehouseJoinReport, dict[str, str]]:
    """raw 두 파일을 읽어 결합본을 만들고, 실행 전후 raw 해시가 같은지 확인한다."""
    before = {"flow": file_digest(sources.flow_path), "stock": file_digest(sources.stock_path)}
    flow_raw, flow_encoding = read_raw_csv(sources.flow_path)
    stock_raw, stock_encoding = read_raw_csv(sources.stock_path)
    report = build_warehouse_dataset(
        flow_raw, stock_raw, sources,
        encodings={"flow": flow_encoding, "stock": stock_encoding},
        digests=before,
    )
    after = {"flow": file_digest(sources.flow_path), "stock": file_digest(sources.stock_path)}
    if before != after:
        raise RuntimeError("실행 중 raw 파일이 변경되었습니다.")
    report.manifest["raw_unchanged"] = True
    return report, before
