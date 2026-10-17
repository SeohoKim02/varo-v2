"""UCI Online Retail transaction adapter for a separate partial-field benchmark.

This is one documented online retailer, not Suhyup and not a store network.
Country means customer residence; it is never used as an inventory location.
"""
from __future__ import annotations
import pandas as pd

REQUIRED=("InvoiceNo","StockCode","Quantity","InvoiceDate","UnitPrice")


def adapt_uci_transactions(frame: pd.DataFrame, *, source_hash: str, first_excel_row=2):
    if not set(REQUIRED)<=set(frame):raise ValueError("공식 거래 자료의 필수 열이 없습니다.")
    if len(source_hash)!=64 or any(c not in "0123456789abcdefABCDEF" for c in source_hash):raise ValueError("원본 파일 확인 정보가 필요합니다.")
    records=[]
    for offset,(_,row) in enumerate(frame.iterrows()):
        qty=pd.to_numeric(row["Quantity"],errors="coerce")
        price=pd.to_numeric(row["UnitPrice"],errors="coerce")
        when=pd.to_datetime(row["InvoiceDate"],errors="coerce")
        invoice=str(row["InvoiceNo"]).strip()
        flags=[]
        if pd.isna(qty):flags.append("INVALID_QUANTITY")
        if pd.isna(price) or price<0:flags.append("INVALID_PRICE")
        if pd.isna(when):flags.append("INVALID_DATE")
        cancelled=invoice.upper().startswith("C")
        if cancelled:flags.append("CANCELLATION")
        if not pd.isna(qty) and qty<0:flags.append("NEGATIVE_ADJUSTMENT_OR_RETURN")
        if not pd.isna(price) and price==0:flags.append("ZERO_PRICE_REQUIRES_CONTEXT")
        product=str(row["StockCode"]).strip()
        if product in {"","None","nan"}:flags.append("MISSING_PRODUCT")
        records.append({"dataset_id":"UCI_ONLINE_RETAIL_352","network_id":"UCI_ONLINE_RETAIL_352",
            "source_file":"Online Retail.xlsx","source_row_id":first_excel_row+offset,"source_sha256":source_hash,
            "product_id":product,"product_key":"UCI_ONLINE_RETAIL_352:StockCode:"+product,
            "invoice_id":invoice,"transaction_time":when.isoformat() if not pd.isna(when) else None,
            "quantity":float(qty) if not pd.isna(qty) else None,"quantity_unit":"EA","unit_status":"SOURCE_METADATA",
            "unit_price":float(price) if not pd.isna(price) else None,"currency":"GBP","currency_status":"SOURCE_METADATA",
            "signed_line_value":float(qty*price) if not pd.isna(qty) and not pd.isna(price) else None,
            "record_kind":"CANCELLATION" if cancelled else "RETAIL_TRANSACTION",
            "quality_flags":flags,"ordinary_positive_sale_eligible":not flags and qty>0 and price>0,
            "scope":"SINGLE_ONLINE_BUSINESS_TRANSACTION","location_id":None,"inventory_qty":None,
            "transport_cost":None,"unit_cost":None,"expiry_date":None,"true_demand_observed":False,
            "customer_country":row.get("Country"),"customer_identifiers_retained":False})
    return records
