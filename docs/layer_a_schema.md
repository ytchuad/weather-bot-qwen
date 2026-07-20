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
