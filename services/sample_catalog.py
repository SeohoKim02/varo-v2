"""Catalog of in-project workbooks used to review the dynamic simulation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
