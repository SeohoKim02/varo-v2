import pandas as pd
from services.public_retail_candidate import adapt_uci_transactions

def test_sales_price_join_and_return_semantics_are_preserved():
    data=pd.DataFrame({"InvoiceNo":["1","C2"],"StockCode":["001","001"],"Quantity":[3,-2],
        "InvoiceDate":["2011-01-01","2011-01-02"],"UnitPrice":[2,2],"Country":["United Kingdom"]*2,"CustomerID":[12345]*2})
    records=adapt_uci_transactions(data,source_hash="a"*64)
    assert records[0]["signed_line_value"]==6 and records[0]["ordinary_positive_sale_eligible"]
    assert records[1]["quantity"]==-2 and records[1]["signed_line_value"]==-4
    assert not records[1]["ordinary_positive_sale_eligible"]
    assert records[0]["product_id"]=="001"
    assert all(r["location_id"] is None and r["inventory_qty"] is None and r["transport_cost"] is None for r in records)
    assert all("CustomerID" not in r for r in records)

def test_missing_and_zero_price_not_conflated():
    data=pd.DataFrame({"InvoiceNo":["1","2"],"StockCode":["a","b"],"Quantity":[1,1],"InvoiceDate":["2011-01-01"]*2,"UnitPrice":[None,0]})
    rows=adapt_uci_transactions(data,source_hash="b"*64)
    assert rows[0]["signed_line_value"] is None and rows[1]["signed_line_value"]==0
    assert all(not row["ordinary_positive_sale_eligible"] for row in rows)
