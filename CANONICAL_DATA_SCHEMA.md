# Varo actual-data interchange, version 1.1.0

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
| inventory_flow | date/period, location, product, variant, state, flow_kind | date; only observed quantity columns populated |
| transfer_network | date, source, target, product; observed distribution OR benchmark candidate | date, source_id, target_id, product_id |
| product_master | observed dataset-local product key, optional processing state | product_id |
| location_master | observed dataset-local location key | location_id |
| demand_series | date/period, location/product/variant or explicit aggregate; sales and demand separate | date |
| location_product_observation | warehouse/SKU occurrence with no fabricated movement | date, location_id, product_id |

Auxiliary evidence tables (`schema_document()["auxiliary_tables"]`):

| Table | Grain | Required |
|---|---|---|
| product_measurement | one physical measurement (member file) of an item in a handling context | product_id, measurement_context |
| product_identity_version | one observed (product_id, product_name) with its valid_from/valid_to | product_id, version_seq, valid_from |

Every row requires source_dataset, source_file, source_sheet_or_table,
source_row_id, transform_version, date_grain and scope. A source identifier can
be NULL when the source is explicitly aggregated or unidentified. This does
not authorize inventing a location or product. `schema_document()` contains the
complete machine-readable required/optional contract and logical key candidates.

Numeric values are nullable floating point, including integral quantities.
Identifiers are strings and preserve leading zeroes. NULL differs from zero.
Daily, monthly, quarterly and static grains are explicit; period_start/end are
calendar bounds, not generated daily observations. A monthly snapshot_date is
its period anchor.

## Units and evidence

`unit` is the quantity unit of the row's counts. `weight_unit`, `volume_unit`,
`dimension_unit`, `currency`, `distance_unit`, `time_unit` and `capacity_unit`
are independent. Each has a `<name>_status`:

- DIRECT: printed in the source header or value (Suhyup `재고량(킬로그램)`, AI Hub `보관공간(㎥)`).
- SOURCE_METADATA: an official page, specification or label states it (NFQS page
  "재고량의 단위는 톤(t) 입니다."; AI Hub label caption "가로 …cm … 무게 …kg").
- DERIVED: produced by a documented transformation or reference engine (Suhyup benchmark KRW/km/min).
- UNKNOWN: no evidence; the unit stays NULL (Suhyup counts, all LogisAll and Jangbogo quantities).

A NULL unit is never replaced by kg, EA or ton. A unit without a status, or a
status that contradicts the unit, is flagged. Pack text inside a product name
(`5KG`, `1EA`) describes the item, not the quantity column. Per-dataset evidence
is in `canonical_mapping.json` → `unit_evidence`.

## Quality flags

`validation_flags` keeps detailed reasons (`negative:order_qty`,
`missing:target_id`, ...). `quality_flags` maps them to standard codes, most
severe first, and `quality_status` is the first code or VALID:

MISSING_KEY, MISSING_VALUE, SOURCE_ANOMALY, NEGATIVE_QUANTITY,
DUPLICATE_OBSERVATION, UNKNOWN_UNIT, UNKNOWN_LOCATION, AMBIGUOUS_PRODUCT,
BENCHMARK_PROXY, OUT_OF_SCOPE.

Flags state analysis scope; they never delete, clip or repair a row.
`analysis_eligible=false` when any blocking flag exists. Informational flags
(`float_residue_negative`, `source_flow_identity_mismatch`,
`product_identity_versioned`, `product_label_variants`) document semantics
without blocking.

## Repeated logical keys

Every source record stays traceable. Records sharing a canonical key collapse
only under a documented dataset policy, into one row with `source_record_count`
and `source_row_id` listing every 1-based record id. A record with a NULL key
part or a NULL/non-numeric/negative measure is never merged. The keys must
cover every non-measure column, so no differing attribute is hidden.

Other repeated keys keep every row. After export, every member is classified:
`cross_release_conflict` (different files, different values),
`cross_release_repeat` (different files, same values),
`repeated_key_missing_subkey` (the distinguishing sub-key, e.g. barcode, is
missing) or `repeated_key_unexplained`. `duplicate_key` still marks each
occurrence after the first. Nothing is summed or picked.

## Source scope and semantic rules

- Suhyup: both raw July 2026 files; count unit UNKNOWN and observed kg DIRECT.
  State-processing codes are part of stock/flow keys. The benchmark export is
  a separate `existing_benchmark_candidate` network source. Existing source
  surplus / target need formulas are documented as benchmark proxies, never
  promoted to observed retail demand. Verified one-to-one code crosswalks bridge
  raw `000156` to historical `156`; canonical IDs keep the raw spelling.
- LogisAll: all received national monthly stock, regional daily sales and
  region-to-region distribution files. ZIPs are masked to first-digit zones
  (`00000`…`60000`), not warehouses. Repeated (date, product, zone) keys are
  distinct sub-zone records: in every release the sales multiplicity equals the
  distribution sender-zone multiplicity and releases never overlap in date.
  They are summed with full lineage. A distribution record with an empty
  arrival field keeps target_id NULL (MISSING_KEY) and is excluded from network
  use; nothing restores it. `PDTN_QY` is not stock.
- Jangbogo: small warehouse/category monthly purchases and sales, two selected
  warehouse/SKU observation releases and the latest full SKU purchase-request
  partition. Purchase requests are order_qty, not inbound/demand, at the
  official month × product × barcode grain (`BRCD_INFO` → product_variant_id;
  literal `NULL` is a missing marker); barcode variants are never summed.
  Negative purchase requests are undocumented signed values: kept, flagged
  NEGATIVE_QUANTITY, never abs() or zeroed. Fully identical occurrence rows
  collapse with a record count. A product_id whose name changes over time is
  versioned in product_identity_version (never renumbered).
- NFQS: raw JSON for all 24 quarters. National total and regional values carry
  distinct scope. Product code 999 is an all-product total, flagged separately.
  Never sum total rows with their components. Ton precision is preserved.
  Negative residues with |x| < 1e-9 t (1/1000 of the finest reported decimal)
  are kept exactly and flagged `float_residue_negative`. The quarterly check is
  the same-record identity ICEALIST + ICEAIN − ICEAOUT = INVENTOTAL (data.go.kr
  describes 이월량/입출하량/재고량); carry-over is re-reported each quarter, so no
  cross-quarter flow identity is assumed. An absent product is NOT_REPORTED in
  `results/nfqs_panel_coverage.csv`, never zero stock.
- AI Hub: 61 workbook observation dates, two independently named warehouse
  metadata rows, and 20,480 item measurements traced to their raw ZIP member.
  A barcode measured as an inbound item and as an outbound item has two
  product_measurement rows and one product_master row (no weight is chosen).
  The daily workbook, measurements and label JSON have no site key, so flow
  rows keep location_id NULL (UNKNOWN_LOCATION) and no placeholder location is
  created. Occupied volume and utilization do not become an inferred capacity.

## Quality and lineage

CSV source_row_id is the 1-based data-record number excluding the header.
Excel row IDs are actual worksheet row numbers; JSON IDs identify array position
and unpivoted field. Extracted measurements identify the original ZIP member.
Master rows retain first observed source lineage; name conflicts are counted
exactly over every canonical fact row.

Invalid dates stay NULL while raw_date remains unchanged. Negative quantities,
nonfinite values, missing required inputs, unresolved identity and unspecified
units are flagged. No raw row is corrected or silently dropped.

Each selected field is classified DIRECT, RENAMED, DERIVED, AGGREGATED,
UNAVAILABLE or NOT_APPLICABLE when warranted. Derived fields document source,
formula, unit and assumptions. Quantity checks compare source sums, non-null
records and represented record counts with canonical output; collapsed tables
also prove every source row id appears exactly once.

Reports under each dataset `results/`:

- canonical_source_inventory.json: existing verified manifest plus scope evidence
- canonical_mapping.json: full schema, unit evidence and per-file mappings
- data_quality_report.json: counts, quality codes, flags, units, date order,
  master integrity, dataset investigations and quantity conservation
- algorithm_coverage.csv: all 15 algorithms, data grain, unit status and FULL blockers
- Suhyup only: canonical_reference_regression.json, exact 31-day comparison
- NFQS only: nfqs_panel_coverage.csv

FULL/PARTIAL/MISSING in field reports describes completeness, not correctness.
Algorithm coverage FULL/PARTIAL/BENCHMARK_ONLY/UNSUPPORTED is a separate decision.
FULL additionally requires known units, observed constraints and a recorded
semantic review; column presence never proves FULL.

## Running

Install `requirements-data.txt` in the offline data environment, then:

```powershell
python -m services.canonical_data_pipeline --data-root C:\VARO_V2_REAL_DATA
python -m services.canonical_reference_validation --data-root C:\VARO_V2_REAL_DATA
```

Use `--datasets suhyup logisall jangbogo nfqs aihub` to select adapters. CSV inputs
are streamed in 50,000-row chunks, except files under a collapse policy, which
are read whole (bounded LogisAll/Jangbogo files). Parquet uses Zstandard
compression. Outputs are `processed/canonical_<table>.parquet`. Tables without
observations are absent. Reruns replace these named generated outputs; raw and
existing benchmark files are read-only. Full tests should set `VARO_OUTPUT_ROOT`
to a temporary directory to protect production DQN latest pointers.
