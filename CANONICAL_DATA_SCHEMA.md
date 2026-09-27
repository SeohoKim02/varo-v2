# Varo actual-data interchange, version 1.0.0

The real-data root is `C:\VARO_V2_REAL_DATA`. The adapter is an offline,
nullable interchange layer. It does not replace `analysis_pipeline`, modify
algorithm formulas, impute missing inputs, or automatically run algorithms.
The existing `services/legacy_adapters/data_adapter.py` continues serving the
application workbook contract. `suhyup_to_existing` is the strict reference
bridge for the existing actual-data benchmark.

## Tables and meaning

| Table | Grain / quantities | Required non-null observations |
|---|---|---|
| inventory_snapshot | date/period, location, product, processing state; stock | snapshot_date, inventory_qty |
| inventory_flow | date/period, location, product, state, flow_kind | date; only observed quantity columns populated |
| transfer_network | date, source, target, product; observed distribution OR benchmark candidate | date, source_id, target_id, product_id |
| product_master | observed dataset-local product key, optional processing state | product_id |
| location_master | observed dataset-local location key | location_id |
| demand_series | date/period, location/product or explicit aggregate; sales and demand separate | date |
| location_product_observation | warehouse/SKU occurrence with no fabricated movement | date, location_id, product_id |

Every row requires source_dataset, source_file, source_sheet_or_table,
source_row_id, transform_version, date_grain and scope. A source identifier can
be NULL when the source is explicitly aggregated or unidentified. This does
not authorize inventing a location or product. `schema_document()` contains the
complete machine-readable required/optional contract and logical key candidates.

Numeric values are nullable floating point, including integral quantities.
Identifiers are strings and preserve leading zeroes. NULL differs from zero.
`unit`, `weight_unit`, `capacity_unit`, and `volume_unit` are independent.
Unknown units stay NULL and block unqualified comparisons. Daily, monthly,
quarterly and static grains are explicit; period_start/end are calendar bounds,
not generated daily observations. A monthly snapshot_date is its period anchor.

## Source scope and semantic rules

- Suhyup: both raw July 2026 files; count and observed kg preserved independently.
  State-processing codes are part of stock/flow keys. The benchmark export is
  a separate `existing_benchmark_candidate` network source. Existing source
  surplus / target need formulas are documented as benchmark proxies, never
  promoted to observed retail demand. Verified one-to-one code crosswalks bridge
  raw `000156` to historical `156`; canonical IDs keep the raw spelling.
- LogisAll: all received national monthly stock, regional daily sales and
  region-to-region distribution files. Production tables and public samples
  are excluded. ZIP denotes a region, not a warehouse. Missing regional stock
  and inbound/outbound full products remain unavailable. `PDTN_QY` is not stock.
- Jangbogo: small warehouse/category monthly purchases and sales, two selected
  warehouse/SKU observation releases and the latest full SKU purchase-request
  partition. This is an explicit first-adapter subset, not all 93.8 GB of raw.
  Purchase requests are order_qty, not inbound/demand. Purchase count is not
  quantity. Category IDs have a `category:` namespace and product_grain.
- NFQS: raw JSON for all 24 quarters. National total and regional values carry
  distinct scope. Product code 999 is an all-product total, flagged separately.
  Never sum total rows with their components. Ton precision is preserved.
- AI Hub: 61 workbook observation dates, two independently named warehouse
  metadata rows, and 20,480 existing verified numeric item measurements traced
  to their raw ZIP member. The daily workbook has no site or SKU key, so it is
  not joined to the two warehouses or product measurements. Occupied volume and
  utilization do not become an inferred capacity. Image/COCO archives are unused.

## Quality and lineage

CSV source_row_id is the 1-based data-record number excluding the header.
Excel row IDs are actual worksheet row numbers; JSON IDs identify array position
and unpivoted field. Extracted measurements identify the original ZIP member.
Master rows retain first observed source lineage; conflicts are counted.

Invalid dates stay NULL while raw_date remains unchanged. Negative quantities,
nonfinite values, missing required inputs, unresolved identity and unspecified
units are flagged. No raw row is corrected or silently dropped. Bad rows remain
in canonical output with analysis_eligible=false (logical quarantine).
Repeated logical keys remain source observations; repeats are flagged across
chunks and reported. A table with duplicate keys needs a release/aggregation
policy before use. The first occurrence is not proof of an unambiguous fact.
Do not auto-sum overlapping releases. No quantity aggregation is currently done.

Each selected field is classified DIRECT, RENAMED, DERIVED, AGGREGATED,
UNAVAILABLE or NOT_APPLICABLE when warranted. Derived fields document source,
formula, unit and assumptions. Quantity checks compare source/canonical row
counts, nonnull counts and sums, including negative flagged values; they do not
claim that a raw sum across mixed units or national/regional totals is meaningful.

Reports under each dataset `results/`:

- canonical_source_inventory.json: existing verified manifest plus scope evidence
- canonical_mapping.json: full schema and per-file mappings
- data_quality_report.json: counts, nulls, duplicates, flags, units, date order,
  master integrity, semantic checks and quantity conservation
- algorithm_coverage.csv: all 15 requested algorithms; conservative semantic gates
- Suhyup only: canonical_reference_regression.json, exact 31-day comparison

FULL/PARTIAL/MISSING in field reports describes completeness, not correctness.
Algorithm coverage FULL/PARTIAL/BENCHMARK_ONLY/UNSUPPORTED is a separate decision.
No new dataset is declared operationally FULL merely because column names exist.

## Running

Install `requirements-data.txt` in the offline data environment, then:

```powershell
python -m services.canonical_data_pipeline --data-root C:\VARO_V2_REAL_DATA
python -m services.canonical_reference_validation --data-root C:\VARO_V2_REAL_DATA
```

Use `--datasets suhyup logisall jangbogo nfqs aihub` to select adapters. CSV inputs
are streamed in 50,000-row chunks; Parquet uses Zstandard compression. Outputs
are `processed/canonical_<table>.parquet`. Tables without observations are absent.
Reruns replace these named generated outputs; raw and existing benchmark files
are read-only. Full tests should set `VARO_OUTPUT_ROOT` to a temporary directory
to protect production DQN latest pointers.
