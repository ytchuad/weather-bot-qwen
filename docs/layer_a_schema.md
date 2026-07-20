# `layer_a.v1` schema

## Boundary

Layer A 是一個 canonical sampling cycle 的 immutable decision-time record。
record 以 `decision_cycle_id` 識別，與 paper account、strategy assignment、cash、position、order、fill 或 PnL 無關。Layer B／Layer C persistence 不在本階段。

同一個五分鐘 canonical slot 使用 deterministic ID：

```text
sha256(event_date | location | market_kind | event_slug | slot_start)[:32]
```

`decision_timestamp` 保存實際 decision time；slot 只用於跨 context entry point deduplication。

## Record envelope

每個 Parquet row 是一個完整 JSON-safe record，並同時以 nested JSON columns 保存：

```text
decision_cycle_id
schema_version = layer_a.v1
decision_timestamp
capture_timestamp
event_date
location
market_kind
event_slug
weather_state
models
market_identity
clob_books
gamma_reference_prices
source_status
completeness
```

`gamma_reference_prices` 只屬於 labelled reference data，不可作為 missing CLOB identity、timestamp 或 depth 的替代品。

## Weather state

`weather_state` 保存 observations、`max_so_far`、`min_so_far`、lags、trends、humidity、pressure、dew point、wind、rain、rain nowcast、forecast、UV（若 source 提供）以及 `status`。status 直接保留 Phase 2A fields：

```text
source_timestamp
age_seconds / age_minutes
is_missing
is_stale
is_fallback
fallback_method
source_name
quality_flags
raw_status
```

沒有 source timestamp 時 age 保持 `null`；capture 不會製造 historical metadata。

## Model state

`models` 對 cycle 內每個 generated model 保存：

```text
model_name
model_version
artifact_identity
feature_spec
numeric_features
diagnostic_features
q10 / q25 / q50 / q75 / q90（若有）
point_prediction
full_bucket_probabilities
classifier_probability（若有）
model_input_status_summary
```

缺少欄位的 model／cycle 仍會保存，並在 `completeness.missing_fields` 標記原因。

## Market identity and books

每個 bucket 的 `market_identity` 保存：

```text
market_id
condition_id
bucket
explicit_outcomes = ["Yes", "No"]
yes_token_id / no_token_id
yes_asset_id / no_asset_id
tick_size
minimum_order_size
market_schema_version
```

每個 YES／NO token 的 `clob_books` 保存完整 normalized depth：

```text
token_side
token_id
asset_id
book_timestamp
decision_timestamp
book_age_seconds
source_name
fetch_cycle_id
tick_size
minimum_order_size
bids = [{price, available_shares}, ...]
asks = [{price, available_shares}, ...]
validation_status
validation_errors
raw_book
```

`bids`／`asks` 是 canonical CLOB normalized depth 的完整內容，不是 top-of-book 摘要。Layer A capture 使用 canonical cycle 已持有的 object，不會重新 fetch 或從 Gamma price 重建。

## Completeness

每個 cycle 固定計算：

```text
weather_complete
model_state_complete
market_identity_complete
token_identity_complete
depth_pair_complete
book_timestamp_complete
fetch_cycle_coherent
replay_eligible_for_model_analysis
replay_eligible_for_clob_strategy
missing_fields
rejection_reasons
```

Incomplete records 不會被丟棄；它們是 pipeline diagnostics 的一部分。

## Prohibited fields

以下欄位不可出現在 Layer A envelope：

```text
account_id, strategy_key, capital, cash_balance,
current_paper_positions, paper_positions, target_orders,
simulated_fills, realized_pnl, unrealized_pnl,
legacy_would_trade, clob_would_trade
```

## `layer_a.market.v1`：one-minute market-only snapshot

`layer_a.market.v1` 是與 account、strategy、paper execution 完全分離的
market sampling record。`market_snapshot_id` 以
`event_date | location | market_kind | event_slug | one-minute-slot` 做
deterministic deduplication；同一分鐘的多個 caller 不會產生第二筆 snapshot。

```text
market_snapshot_id
schema_version = layer_a.market.v1
decision_timestamp
capture_timestamp
event_date / location / market_kind / event_slug
market_identity[]
clob_books[]
fetch_cycle_id
latest_model_cycle_id
latest_model_cycle_timestamp
model_age_seconds
gamma_reference_prices / gamma_reference_data
source_status
completeness
```

每個 market identity 明確保存 `market_id`、`condition_id`、`bucket`、
`explicit_outcomes`、YES/NO token ID、YES/NO asset ID、tick size、minimum
order size 與 market schema version。`clob_books` 保存 exact normalized YES
及 NO full bids/asks、book timestamp、source、fetch cycle、validation status
與 validation errors；Gamma price 只放在 reference 欄位，不能替代 CLOB depth。

每一分鐘 snapshot 只連結到該時間以前最新的 completed five-minute
`latest_model_cycle_id`，並計算 `model_age_seconds`。沒有可用 model cycle
時仍可保存 incomplete market snapshot，replay eligibility 會明確標示原因。

## `layer_a.weather.v1`: one-minute weather observation snapshot

`layer_a.weather.v1` is an account-independent observation stream. It is
captured every minute and does not run model inference or fetch CLOB data.

```text
weather_snapshot_id          deterministic event_date | location | minute slot ID
schema_version = layer_a.weather.v1
snapshot_timestamp           capture decision minute
capture_timestamp            actual local capture time
event_date / location
latest_model_cycle_id        latest already-completed five-minute cycle, or null
model_cycle_timestamp        linked cycle timestamp, or null
model_age_seconds            backward as-of age, or null
temperature_current
max_so_far / min_so_far
relative_humidity
pressure / dew_point
rain_current
observations                 scalar convenience values
observation_status           Phase 2A status for every observation field
source_status
```

`observation_status[field]` preserves `value`, `source_timestamp`,
`decision_timestamp`, `age_seconds`, `age_minutes`, `is_missing`, `is_stale`,
`is_fallback`, `fallback_method`, `source_name`, `quality_flags` and
`raw_status`. Age is recalculated only from supplied timestamps; a missing
source timestamp always produces `null` age. Numeric zero is a valid value and
is never converted to missing. A repeated source observation keeps its old
source timestamp, so its age increases truthfully on later snapshots.

Weather local chunks use `date=YYYY-MM-DD/hour=HH/minute=MM/` and are closed by
the shared `LAYER_A_MINUTE_PARTITION_MINUTES` setting (default 10 minutes).
The existing `layer_a.v1` model-cycle schema and `layer_a.market.v1` market
schema remain compatible and are not rewritten.
