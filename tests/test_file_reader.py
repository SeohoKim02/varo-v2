"""Upload safety-layer tests: extension guard, empty/corrupt, CSV encodings, Excel."""
from __future__ import annotations

import io
import unittest

from services.data_loader import DataLoadError
from services.file_reader import (
    file_extension,
    is_supported_extension,
    read_csv_frame,
    read_uploaded_data,
)
from tests.fixtures import workbook_excel_bytes


class ExtensionGuardTests(unittest.TestCase):
    def test_supported_and_unsupported_extensions(self):
        for name in ("a.xlsx", "b.XLS", "c.csv", "폴더/한글 파일 이름.xlsx"):
            self.assertTrue(is_supported_extension(name), name)
        for name in ("a.pdf", "b.txt", "c.json", "noext"):
            self.assertFalse(is_supported_extension(name), name)

    def test_extension_lowercased(self):
        self.assertEqual(file_extension("REPORT.XLSX"), ".xlsx")

    def test_unsupported_extension_refused_without_read(self):
        with self.assertRaises(DataLoadError):
            read_uploaded_data(io.BytesIO(b"whatever"), "data.pdf")


class EmptyAndCorruptTests(unittest.TestCase):
    def test_empty_file_message(self):
        for name in ("x.xlsx", "x.csv"):
            with self.assertRaises(DataLoadError) as ctx:
                read_uploaded_data(io.BytesIO(b""), name)
            self.assertIn("비어", str(ctx.exception))

    def test_corrupt_excel_is_clean_error(self):
        with self.assertRaises(DataLoadError):
            read_uploaded_data(io.BytesIO(b"not a real excel file"), "x.xlsx")


class CsvEncodingTests(unittest.TestCase):
    def test_utf8_bom_decodes(self):
        raw = "﻿점포,재고\nS1,10\n".encode("utf-8-sig")
        frame = read_csv_frame(io.BytesIO(raw))
        self.assertEqual(list(frame.columns), ["점포", "재고"])

    def test_cp949_decodes(self):
        raw = "점포,재고\nS1,10\n".encode("cp949")
        frame = read_csv_frame(io.BytesIO(raw))
        self.assertEqual(list(frame.columns), ["점포", "재고"])

    def test_csv_read_but_guided_to_excel(self):
        raw = "점포,재고\nS1,10\n".encode("utf-8")
        with self.assertRaises(DataLoadError) as ctx:
            read_uploaded_data(io.BytesIO(raw), "one_table.csv")
        self.assertIn("여러 시트", str(ctx.exception))


class ExcelPathTests(unittest.TestCase):
    def test_valid_xlsx_reads_all_required_sheets(self):
        data = read_uploaded_data(workbook_excel_bytes(), "정상 업로드.xlsx")
        for sheet in ("stores", "products", "inventory", "routes"):
            self.assertIn(sheet, data)

    def test_header_only_excel_has_zero_data_rows(self):
        import pandas as pd
        empty_sheets = {
            "stores": pd.DataFrame(columns=["node_id", "node_name", "node_type"]),
            "products": pd.DataFrame(columns=["product_id", "product_name"]),
            "inventory": pd.DataFrame(columns=["store_id", "product_id", "stock_qty"]),
            "routes": pd.DataFrame(columns=["source_id", "target_id", "distance_km", "estimated_cost", "travel_time_min"]),
        }
        data = read_uploaded_data(workbook_excel_bytes(empty_sheets), "헤더만.xlsx")
        self.assertEqual(len(data["stores"]), 0)


if __name__ == "__main__":
    unittest.main()
