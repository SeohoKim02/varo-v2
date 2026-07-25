"""Safe upload entry layer: extension guard + encoding-robust reading.

Wraps the existing Excel loader with a thin, user-facing safety layer so that a
real operator upload never crashes the app:

* Only ``.xlsx`` / ``.xls`` / ``.csv`` are accepted; anything else is refused
  immediately with a plain message (no read attempt).
* Empty / corrupt / encrypted files raise a clean ``DataLoadError`` instead of a
  library traceback.
* CSV files are decoded with a Korean-Windows-aware encoding fallback so they are
  *read* safely; because a single flat table cannot supply Varo's four related
  sheets (stores/products/inventory/routes), the user is guided to a multi-sheet
  Excel workbook rather than silently producing a broken analysis.

The Excel path delegates unchanged to ``data_loader.load_excel_data`` so existing
behavior and tests are untouched.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from services.data_loader import DataLoadError, load_excel_data

SUPPORTED_EXTENSIONS = (".xlsx", ".xls", ".csv")
_EXCEL_EXTENSIONS = (".xlsx", ".xls")
# Tried in order; utf-8-sig first so a UTF-8 BOM is stripped, then Korean Windows.
_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")

_CSV_NEEDS_WORKBOOK = (
    "CSV 파일은 한 개의 표만 담을 수 있어 분석에 필요한 여러 시트"
    "(점포·상품·재고·경로)를 만들 수 없습니다. 여러 시트가 있는 Excel 파일을 올려주세요."
)


def file_extension(filename: Any) -> str:
    return Path(str(filename or "")).suffix.lower()


def is_supported_extension(filename: Any) -> bool:
    return file_extension(filename) in SUPPORTED_EXTENSIONS


def _read_all_bytes(source: Any) -> bytes | None:
    """Best-effort raw bytes for empty/size checks without consuming the stream."""
    try:
        if hasattr(source, "getvalue"):
            value = source.getvalue()
            return value if isinstance(value, bytes) else str(value).encode("utf-8")
        if isinstance(source, (str, Path)):
            return Path(source).read_bytes()
        if hasattr(source, "read"):
            position = source.tell() if hasattr(source, "tell") else None
            if hasattr(source, "seek"):
                source.seek(0)
            content = source.read()
            if position is not None and hasattr(source, "seek"):
                source.seek(position)
            return content if isinstance(content, bytes) else str(content).encode("utf-8")
    except Exception:
        return None
    return None


def read_csv_frame(source: Any) -> pd.DataFrame:
    """Decode a CSV with an encoding fallback and return a single DataFrame.

    Raises ``DataLoadError`` (clean message) when the file is empty or cannot be
    decoded/parsed in any supported encoding.
    """
    raw = _read_all_bytes(source)
    if raw is None:
        raise DataLoadError("파일을 읽을 수 없습니다. 파일이 손상되었는지 확인하세요.")
    if len(raw.strip()) == 0:
        raise DataLoadError("파일이 비어 있습니다. 데이터가 있는 파일을 올려주세요.")
    last_error: Exception | None = None
    for encoding in _CSV_ENCODINGS:
        try:
            frame = pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError("데이터 행이 없습니다. 헤더 아래에 데이터를 추가하세요.") from exc
        except Exception as exc:  # malformed CSV structure
            last_error = exc
            continue
        if frame.shape[1] == 0:
            raise DataLoadError("CSV에서 열을 찾을 수 없습니다. 구분자와 형식을 확인하세요.")
        return frame
    raise DataLoadError(
        "CSV 인코딩을 인식할 수 없습니다. UTF-8 또는 Windows(EUC-KR) 형식으로 저장한 뒤 다시 올려주세요."
    ) from last_error


def read_uploaded_data(source: str | Path | BinaryIO, filename: str = "", *, return_report: bool = False):
    """Read an uploaded workbook by extension, guarding unsupported/empty/corrupt files.

    Excel (.xlsx/.xls) delegates to the existing multi-sheet loader. CSV is decoded
    safely then refused with guidance (a single table cannot form the four sheets).
    """
    extension = file_extension(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        raise DataLoadError("지원하지 않는 파일 형식입니다. .xlsx, .xls, .csv 파일만 올릴 수 있습니다.")

    raw = _read_all_bytes(source)
    if raw is not None and len(raw.strip()) == 0:
        raise DataLoadError("파일이 비어 있습니다. 데이터가 있는 파일을 올려주세요.")

    if extension in _EXCEL_EXTENSIONS:
        # load_excel_data already converts library errors into a clean DataLoadError.
        return load_excel_data(source, return_report=return_report)

    # CSV: decode safely (raises a clean message on failure), then guide to Excel.
    read_csv_frame(source)
    raise DataLoadError(_CSV_NEEDS_WORKBOOK)
