# Varo actual-data interchange, schema version 1.2.0

Schema 1.2.0 extends 1.1.0 without changing any 1.1.0 transform. The domestic
adapters (Suhyup, LogisAll, Jangbogo, NFQS, AI Hub) keep `transform_version`
1.1.0. The external generalisation adapters (M5, Favorita, FreshRetailNet,
KAMP) write 1.2.0.

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

Context tables (1.2.0, `schema_document()["context_tables"]`) hold dated context,
never fact quantities:

| Table | Grain | Required |
|---|---|---|
| calendar_event | one named event/holiday on a date (event_type, event_locale, event_locale_name, event_transferred) | date, event_name |
| covariate_series | one dated covariate value for a location or explicit scope (store traffic, oil price, weather, SNAP/holiday indicator) | date, covariate_name |

Fields added in 1.2.0 (NULL for 1.1.0 outputs):

- Identity: `product_id_namespace`/`product_key` and `location_id_namespace`/`location_key` on
  inventory_snapshot, inventory_flow, demand_series, location_product_observation, the masters and covariate_series.
- demand_series: `discount_rate` (1.0 = no discount), `stockout_hours`, `stockout_window_hours`,
  `hourly_sales` (24 values joined by `|`, shortest round-trip text) and `hourly_stockout_status`
  (24 characters `0`/`1`, hour 0..23).
- inventory_flow: `project_id` and `project_part`, the ordering project and building part of an
  order-based shipment. They are not locations, and they are part of the inventory_flow key.
- product_master: `category_path` (`level=value|level=value`, lossless hierarchy).
- location_master: `city`, `location_subtype` (e.g. store type), `location_cluster`.

## Identity namespaces

`product_id`/`location_id` keep the raw dataset-local spelling. External rows also
carry `<entity>_id_namespace` = `<Dataset>:<source field>` (`M5:item_id`,
`M5:store_id`, `M5:state_id`, `Favorita:item_nbr`, `Favorita:store_nbr`,
`FreshRetailNet:product_id`, `FreshRetailNet:store_id`, `KAMP:rebar_grade`).
`canonicalize` derives `<entity>_key` = namespace + `:` + raw id centrally. A key
supplied by an adapter is overwritten, and a NULL part gives a NULL key. Unions
across datasets must key on `product_key`/`location_key`: the same raw id in two
namespaces stays two entities. 1.1.0 domestic outputs carry no namespace, so their
key is NULL and `(source_dataset, id)` identifies them.

## Wide-to-long lineage

An unpivoted cell keeps `source_row_id` = `<record>:<source column>`, as NFQS
region fields already did. Examples are M5 `17:d_123` (CSV data record 17, column
d_123) and KAMP `2:HD10` (worksheet row 2, grade column HD10). A source NULL cell
stays NULL and is flagged `missing:<field>`. A source zero stays zero. No cell is
generated for an absent record. Streamed conservation compares exact (fsum) sums
and non-null counts, and it also compares zero counts, so NULL and zero cannot
trade places.

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

Flags added in 1.2.0 (no new quality code):

| Flag | Code | Blocking | Meaning |
|---|---|---|---|
| `documented_return:<col>` | NEGATIVE_QUANTITY | no | The official description states negatives are returns (Favorita). The value also carries the blocking `negative:<col>`. |
| `weekly_price_absent` | MISSING_VALUE | no | No weekly price record and zero sales. M5 guide: a missing price means not sold that week. The price stays NULL. |
| `sales_without_weekly_price` | SOURCE_ANOMALY | yes | Positive sales in a week without a price, which contradicts the M5 guide (0 rows observed). |
| `promotion_not_reported` | MISSING_VALUE | no | The source promotion field is NULL (Favorita onpromotion NaN). It is never read as False. |
| `discount_rate_zero`, `discount_rate_above_one` | SOURCE_ANOMALY | no | The discount rate is outside the documented (0, 1] range. The value is kept, and the sales observation stays usable. |
| `stockout_count_mismatch` | SOURCE_ANOMALY | yes | The hourly out-of-stock slots in 6..21 do not sum to stock_hour6_22_cnt (0 rows observed). |
| `hourly_sales_sum_mismatch` | SOURCE_ANOMALY | yes | The hourly sales differ from the daily amount by more than 1e-9 (0 rows observed). |

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

- M5: `sales_train_evaluation.csv` is converted to demand_series by unpivoting d_1..d_1941 via
  `calendar.csv`. Unit `item` is SOURCE_METADATA ("The number of units sold at day i").
  `sales_train_validation.csv` repeats d_1..d_1913 cell for cell (verified), so it is not converted
  twice. `sample_submission.csv` is a scoring template. The weekly `sell_price` (USD,
  SOURCE_METADATA) repeats on each sales day of its week. A missing weekly price stays NULL
  (`weekly_price_absent`), and price records for weeks after d_1941 are not attached. Events go to
  calendar_event and state SNAP indicators to covariate_series (location = `M5:state_id`). M5 has
  no promotion, inventory, cost or network.
- Favorita: `train.csv` (125,497,040 records) is streamed in 64 MB blocks. unit_sales is signed:
  negatives are documented returns, kept and flagged, never abs()/0, and no returns_qty is
  derived. The unit is item-dependent (count or kg) and unlabelled, so it is UNKNOWN. Zero-sales
  rows are absent from the source: an absent key is neither zero nor a stock-out, and nothing is
  generated. `onpromotion` NULL stays NULL. `stores.csv` and `items.csv` are the masters
  (city/state/type/cluster, family/class/perishable). transactions (unit `transaction`) and oil
  price (unit UNKNOWN; NULL stays NULL) go to covariate_series. holidays_events goes to
  calendar_event: `locale_name` is a place name, not a store, and `transferred` is kept.
  `test.csv` (no ground truth) and `sample_submission.csv` are not converted.
- FreshRetailNet: both parquet splits (train 90 days, eval 7 days, disjoint) go to demand_series.
  `sale_amount` is a globally normalised amount (unit `normalized_sales_amount`, SOURCE_METADATA,
  coefficient undisclosed). `hours_stock_status` 1 = out of stock, verified against
  stock_hour6_22_cnt on every row. Availability is not inventory: no inventory_snapshot exists.
  `activity_flag` maps to promotion, and `discount` to discount_rate. Store-day weather and the
  date-level holiday flag collapse from identical member rows into covariate_series, listing
  every member record id; differing members fail the run. Per-product perishability is not
  asserted: the card says 865 SKUs and the report says 863.
- KAMP: the `Export` sheet goes to inventory_flow with one row per (worksheet row, grade). 합계
  is kept as an all-products row (`aggregate_product_total`), never summed with grades. 공사/부위
  (guidebook: 발주공사명/발주공사부위) are project_id/project_part, not locations. The shipping
  plant is not keyed, so location_id is NULL (UNKNOWN_LOCATION). The quantity unit is not stated
  (UNKNOWN). The guidebook names a stock collection that the workbook does not contain, so no
  inventory_snapshot exists and shortage/surplus is never derived. The date grain is `event`
  (shipment dates, not a daily panel).

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

`C:\VARO_V2_REAL_DATA\_COLLECTION_STATUS\EXTERNAL_VALIDATION_MATRIX.csv` holds one row
per canonical dataset, derived only from the dataset's canonical outputs, algorithm
coverage and the collection catalog. It records realness, date range, fact rows,
products and locations, and the has_* flags from observed non-null fields. Its
forecast/inventory/network/dqn readiness is the best coverage level of the
respective algorithm group. Rebuild it with
`python -m services.external_canonical_pipeline --data-root C:\VARO_V2_REAL_DATA --matrix-only`.

FULL/PARTIAL/MISSING in field reports describes completeness, not correctness.
Algorithm coverage FULL/PARTIAL/BENCHMARK_ONLY/UNSUPPORTED is a separate decision.
FULL additionally requires known units, observed constraints and a recorded
semantic review; column presence never proves FULL. Pseudo-fields such as
`consistent_unit` count as present only when the runner verified them. External
datasets use explicit decisions (`EXTERNAL_COVERAGE`). The only FULL is M5 Demand
Forecast, by recorded review: observed unit sales, a documented unit, verified key
uniqueness, and holdout actuals in the data. Its limitation is that sales stay
censored by unobserved stock-outs. Every routing, network and DQN algorithm is
UNSUPPORTED for all four external datasets, because none has inventory, cost,
capacity or a source/target network.

## Running

Install `requirements-data.txt` in the offline data environment, then:

```powershell
python -m services.canonical_data_pipeline --data-root C:\VARO_V2_REAL_DATA
python -m services.canonical_reference_validation --data-root C:\VARO_V2_REAL_DATA
```

Use `--datasets suhyup logisall jangbogo nfqs aihub m5 favorita freshretailnet kamp`
to select adapters; every run ends by rebuilding the validation matrix. CSV inputs
are streamed in 50,000-row chunks, except files under a collapse policy, which
are read whole (bounded LogisAll/Jangbogo files). External sources are streamed as
M5 500 series x 1,941 days, Favorita 64 MB blocks and FreshRetailNet 250,000-row
batches. Their key uniqueness is exact: one 64-bit hash per row is checked at
finish (`Exporter.bulk_key_tables`, reported as `semantic_checks.key_uniqueness`),
and any repeat is flagged and classified exactly like the per-row counter. Parquet uses Zstandard
compression. Outputs are `processed/canonical_<table>.parquet`. Tables without
observations are absent. Reruns replace these named generated outputs; raw and
existing benchmark files are read-only. Full tests should set `VARO_OUTPUT_ROOT`
to a temporary directory to protect production DQN latest pointers.
