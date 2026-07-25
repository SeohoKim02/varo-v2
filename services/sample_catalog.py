"""Catalog of workbooks used to review the dynamic simulation and DQN training.

Two catalogs live here:

* ``SAMPLE_WORKBOOKS`` — the in-project simulation review samples under ``samples/``.
* ``discover_dqn_samples`` — the user's pre-built DQN training workbooks, found by
  scanning known project folders. Originals are never modified, moved, or copied;
  the app references them in place through absolute paths.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SampleWorkbook:
    key: str
    label: str
    filename: str
    store_count: int
    dc_count: int


SAMPLE_WORKBOOKS = (
    SampleWorkbook("small_4stores_1dc", "4점포 1DC 샘플", "Varo_V2_sample_small_4stores_1dc.xlsx", 4, 1),
    SampleWorkbook("normal_6stores_1dc", "6점포 1DC 샘플", "Varo_V2_sample_normal_6stores_1dc.xlsx", 6, 1),
    SampleWorkbook("standard_8stores_1dc", "8점포 1DC 샘플", "Varo_V2_sample_standard_8stores_1dc.xlsx", 8, 1),
    SampleWorkbook("dual_dc_10stores_2dc", "10점포 2DC 샘플", "Varo_V2_sample_dual_dc_10stores_2dc.xlsx", 10, 2),
    SampleWorkbook("edge_3stores_1dc", "3점포 1DC 샘플", "Varo_V2_sample_edge_3stores_1dc.xlsx", 3, 1),
)


def samples_dir(base_dir: Path | None = None) -> Path:
    root = base_dir or Path(__file__).resolve().parents[1]
    return root / "samples"


def sample_options() -> dict[str, SampleWorkbook]:
    return {sample.label: sample for sample in SAMPLE_WORKBOOKS}


def sample_path(sample: SampleWorkbook, base_dir: Path | None = None) -> Path:
    """Return a catalog path only when it remains inside the V2 samples folder."""
    root = samples_dir(base_dir).resolve()
    path = (root / sample.filename).resolve()
    if path.parent != root:
        raise ValueError("샘플 파일 경로가 V2 samples 폴더를 벗어났습니다.")
    return path


# --------------------------------------------------------------------------- #
# DQN training sample discovery
# --------------------------------------------------------------------------- #

# Filename keywords that flag a workbook as a training-sample candidate.
_SAMPLE_KEYWORDS = ("dqn", "sample", "샘플", "varo", "재고", "inventory")
# The specific DQN pack the user prepared (Varo_DQN_sample_01_..., etc.).
_DQN_NAME_RE = re.compile(r"varo[_\- ]?dqn[_\- ]?sample[_\- ]?(\d+)", re.IGNORECASE)
# Faithful Korean labels for the category token carried in each real filename.
_CATEGORY_LABELS = {
    "fresh_meal": "냉장 도시락",
    "frozen": "냉동식품",
    "dairy_bakery": "유제품·베이커리",
    "produce": "신선채소",
    "bakery": "베이커리",
    "beverage_dry": "음료·건식",
    "meat_egg": "정육·계란",
    "seafood": "수산",
    "meal_kit": "밀키트",
    "mixed": "혼합 재고",
}


@dataclass(frozen=True)
class DqnSampleInfo:
    sample_id: str
    file_name: str
    file_path: str
    store_count: int
    dc_count: int
    product_count: int
    inventory_count: int
    route_count: int
    recommendation_count: int
    has_required_sheets: bool
    validation_status: str
    note: str
    label: str = ""
    category: str = ""
    sort_key: int = 999
    modified_at: str = ""
    file_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _candidate_dqn_dirs(base_dir: Path) -> list[Path]:
    """Ordered, de-duplicated list of existing folders that may hold DQN samples.

    Uses an optional ``VARO_DQN_SAMPLES_DIR`` path override (never a secret), the
    in-project normalized-copy folder, and well-known project roots derived from
    the current user's home directory — so it works without hardcoding usernames.
    """
    # An explicit path override is authoritative: search only it (used by tests
    # and by users who keep the pack in a custom location). It is a path, not a secret.
    override = os.environ.get("VARO_DQN_SAMPLES_DIR")
    if override:
        path = Path(override)
        return [path.resolve()] if path.is_dir() else []

    candidates: list[Path] = []
    candidates.append(base_dir / "data" / "normalized_samples")
    candidates.append(base_dir / "samples" / "dqn")

    home = Path.home()
    # Windows can silently redirect the Desktop into OneDrive (Known Folder
    # Move), so probe both the classic and the OneDrive Desktop locations.
    desktop_roots = [home / "Desktop", home / "OneDrive" / "Desktop"]
    onedrive = os.environ.get("OneDrive") or os.environ.get("ONEDRIVE")
    if onedrive:
        desktop_roots.append(Path(onedrive) / "Desktop")

    project_roots: list[Path] = []
    for desktop in desktop_roots:
        project_roots.extend([
            desktop / "Projects" / "Varo",
            desktop / "projects" / "Varo",
            desktop / "Projects",
            desktop / "projects",
        ])
    project_roots.extend([base_dir.parent, base_dir.parent.parent])
    for root in project_roots:
        candidates.append(root / "Varo_DQN_training_samples_10pack")
        candidates.append(root)
    # Known machine-specific fallbacks the user listed (existence-checked below).
    candidates.append(Path(r"C:\Users\user\Desktop\Projects\Varo\Varo_DQN_training_samples_10pack"))
    candidates.append(Path(r"C:\Users\82102\Desktop\Projects\Varo\Varo_DQN_training_samples_10pack"))
    candidates.append(Path(r"C:\Projects\Varo\Varo_DQN_training_samples_10pack"))
    # Shallow recursive fallback: look one/two levels under the project parents
    # for a folder with the pack's exact name (cheap glob, never a full rglob).
    for root in (base_dir.parent, base_dir.parent.parent, *desktop_roots):
        try:
            candidates.extend(root.glob("*/Varo_DQN_training_samples_10pack"))
            candidates.extend(root.glob("*/*/Varo_DQN_training_samples_10pack"))
        except OSError:
            continue

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


def _has_keyword(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in _SAMPLE_KEYWORDS)


def _sample_id_and_sort(name: str) -> tuple[str, int]:
    match = _DQN_NAME_RE.search(name)
    if match:
        return f"{int(match.group(1)):02d}", int(match.group(1))
    slug = re.sub(r"\.xlsx$", "", name, flags=re.IGNORECASE)
    slug = re.sub(r"[^0-9A-Za-z가-힣]+", "_", slug).strip("_")
    return slug or name, 999


def sample_id_from_filename(name: str) -> str:
    """Public helper: short, filesystem-safe sample id parsed from a filename."""
    return _sample_id_and_sort(name)[0]


def _category_from_name(name: str) -> str:
    stem = re.sub(r"\.xlsx$", "", name, flags=re.IGNORECASE).lower()
    for token, label in _CATEGORY_LABELS.items():
        if token in stem:
            return label
    tail = re.sub(r"^varo[_\- ]?dqn[_\- ]?sample[_\- ]?\d+[_\- ]?", "", stem, flags=re.IGNORECASE)
    tail = re.sub(r"\d+stores?", "", tail)
    tail = re.sub(r"\d+dc", "", tail)
    return tail.strip("_ ").replace("_", " ") or "재고"


def _build_note(store_count: int, dc_count: int, status: str) -> str:
    parts: list[str] = []
    if store_count <= 2:
        parts.append("극단 테스트용 소규모")
    elif store_count >= 10:
        parts.append("대규모 샘플")
    else:
        parts.append("표준 규모")
    if dc_count >= 2:
        parts.append("다중 DC")
    if not (2 <= store_count <= 10):
        parts.append("점포 수 확인 필요")
    if status not in ("통과", "주의"):
        parts.append(status)
    return " · ".join(parts)


def _inspect_workbook(path: Path) -> DqnSampleInfo | None:
    """Load + validate one workbook (no heavy pipeline) into catalog metadata."""
    from services.data_loader import DataLoadError, load_excel_data
    from services.data_validator import validate_workbook_data

    sample_id, sort_key = _sample_id_and_sort(path.name)
    category = _category_from_name(path.name)
    try:
        data = load_excel_data(path)
    except DataLoadError:
        return DqnSampleInfo(
            sample_id=sample_id, file_name=path.name, file_path=str(path),
            store_count=0, dc_count=0, product_count=0, inventory_count=0,
            route_count=0, recommendation_count=0, has_required_sheets=False,
            validation_status="구조 확인 필요", note="필수 시트를 읽지 못했습니다.",
            label=path.name, category=category, sort_key=sort_key,
        )
    except Exception:  # pragma: no cover - defensive: never break the catalog
        return None

    validation = validate_workbook_data(data)
    summary = validation.summary
    store_count = int(summary.get("store_count", 0))
    dc_count = int(summary.get("dc_count", 0))
    label = f"DQN 샘플 {sample_id} · {store_count}점포 {dc_count}DC · {category}"
    return DqnSampleInfo(
        sample_id=sample_id,
        file_name=path.name,
        file_path=str(path),
        store_count=store_count,
        dc_count=dc_count,
        product_count=int(summary.get("product_count", 0)),
        inventory_count=int(summary.get("inventory_count", 0)),
        route_count=int(summary.get("route_count", 0)),
        recommendation_count=int(summary.get("recommendation_count", 0)),
        has_required_sheets=True,
        validation_status=validation.status,
        note=_build_note(store_count, dc_count, validation.status),
        label=label,
        category=category,
        sort_key=sort_key,
    )


# path -> (mtime, size, info). Persists across Streamlit reruns in one process;
# a file whose mtime or size changes is re-inspected so the catalog stays fresh.
_INSPECT_CACHE: dict[str, tuple[float, int, DqnSampleInfo | None]] = {}


def _inspect_cached(path: Path) -> DqnSampleInfo | None:
    from dataclasses import replace
    from datetime import datetime

    try:
        stat = path.stat()
    except OSError:
        return None
    key = str(path)
    signature = (stat.st_mtime, stat.st_size)
    cached = _INSPECT_CACHE.get(key)
    if cached is not None and cached[:2] == signature:
        return cached[2]
    info = _inspect_workbook(path)
    if info is not None:
        info = replace(
            info,
            modified_at=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            file_size=int(stat.st_size),
        )
    _INSPECT_CACHE[key] = (signature, stat.st_size, info)
    return info


def discover_dqn_samples(base_dir: Path | None = None, broaden_if_fewer_than: int = 10) -> list[DqnSampleInfo]:
    """Find the user's DQN training samples across known project folders.

    Returns Varo-structured workbooks whose filename matches the DQN pack first;
    if fewer than ``broaden_if_fewer_than`` are found, keyword-matching
    Varo-structured workbooks are added as a fallback. Never creates files.
    """
    base = base_dir or Path(__file__).resolve().parents[1]
    dqn: dict[str, DqnSampleInfo] = {}
    other: dict[str, DqnSampleInfo] = {}
    for directory in _candidate_dqn_dirs(base):
        for path in sorted(directory.glob("*.xlsx")):
            name = path.name
            if name.startswith("~$"):
                continue
            is_dqn = bool(_DQN_NAME_RE.search(name))
            if not is_dqn and not _has_keyword(name):
                continue
            info = _inspect_cached(path)
            if info is None or not info.has_required_sheets:
                continue
            bucket = dqn if is_dqn else other
            bucket.setdefault(info.sample_id, info)

    ordered = sorted(dqn.values(), key=lambda item: (item.sort_key, item.file_name))
    if len(ordered) < broaden_if_fewer_than:
        seen = {item.sample_id for item in ordered}
        ordered += sorted(
            (item for item in other.values() if item.sample_id not in seen),
            key=lambda item: (item.sort_key, item.file_name),
        )
    return ordered


def dqn_sample_options(base_dir: Path | None = None) -> dict[str, DqnSampleInfo]:
    """Label -> DqnSampleInfo for a selection UI, preserving catalog order."""
    return {info.label: info for info in discover_dqn_samples(base_dir)}
