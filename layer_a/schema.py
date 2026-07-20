"""The strategy-independent ``layer_a.v1`` record contract.

This module deliberately does not import execution/account code.  A record is
made from the immutable weather/model/market/book payload produced by the
canonical-cycle builder.  Paper-account state is therefore not an input to
the schema and cannot accidentally become part of a Layer A partition.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "layer_a.v1"
HKT = timezone(timedelta(hours=8))
_QUANTILE_KEYS = ("q10", "q25", "q50", "q75", "q90")
_MODEL_QUANTILE_ALIASES = {
    "q10": ("q10", "upside_q10", "remaining_upside_p10", "pred_tmax_q10", "pred_tmax_p10"),
    "q25": ("q25", "upside_q25", "remaining_upside_p25", "pred_tmax_q25", "pred_tmax_p25"),
    "q50": ("q50", "upside_q50", "remaining_upside_p50", "pred_tmax_q50", "pred_tmax_p50"),
    "q75": ("q75", "upside_q75", "remaining_upside_p75", "pred_tmax_q75", "pred_tmax_p75"),
    "q90": ("q90", "upside_q90", "remaining_upside_p90", "pred_tmax_q90", "pred_tmax_p90"),
}
_MODEL_PROHIBITED_FIELDS = {
    "account_id",
    "account",
    "strategy",
    "strategy_id",
    "strategy_key",
    "capital",
    "cash",
    "cash_balance",
    "current_paper_positions",
    "paper_positions",
    "paper_position",
    "target_orders",
    "target_order",
    "fills",
    "fill",
    "simulated_fills",
    "realized_pnl",
    "unrealized_pnl",
    "pnl",
    "legacy_would_trade",
    "clob_would_trade",
}


class LayerASchemaError(ValueError):
    """Raised when a record cannot satisfy the immutable Layer A envelope."""


def _jsonable(value: Any) -> Any:
    """Convert runtime objects to JSON-safe values without inventing data."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # numpy/pandas scalar values expose ``item``.  Keep this optional so the
    # capture package remains usable in small CLI/test environments.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def jsonable(value: Any) -> Any:
    """Public JSON-safe copy helper used by storage and replay code."""
    return deepcopy(_jsonable(value))


def _parse_datetime(value: Any, *, naive_timezone: timezone = HKT) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            return None
        if numeric > 100_000_000_000:
            numeric /= 1000.0
        try:
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            # Polymarket CLOB responses commonly encode epoch milliseconds
            # as strings (for example ``"1784506420427"``).  Treat them the
            # same way as numeric timestamps; otherwise a valid full book is
            # incorrectly marked as missing its book timestamp.
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(numeric) or numeric <= 0:
                return None
            if numeric > 100_000_000_000:
                numeric /= 1000.0
            try:
                parsed = datetime.fromtimestamp(numeric, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=naive_timezone)
    return parsed.astimezone(timezone.utc)


def _iso_datetime(value: Any, *, naive_timezone: timezone = HKT) -> str | None:
    parsed = _parse_datetime(value, naive_timezone=naive_timezone)
    return parsed.isoformat() if parsed is not None else None


def _number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer() and isinstance(value, int):
        return int(number)
    return number


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _parse_outcomes(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def make_decision_cycle_id(
    decision_timestamp: Any,
    *,
    event_date: str | date | None,
    location: str,
    event_slug: str,
    market_kind: str,
    cadence_minutes: int = 5,
) -> str:
    """Return a deterministic ID for one canonical sampling slot.

    The ID is independent of account/strategy values.  A five-minute slot is
    used because the existing scheduler and model snapshot cadence are
    sub-hour; the exact unrounded decision timestamp is still persisted.
    """
    parsed = _parse_datetime(decision_timestamp)
    if parsed is None:
        raise LayerASchemaError("decision_timestamp is required for cycle identity")
    cadence = max(1, int(cadence_minutes))
    local = parsed.astimezone(HKT)
    slot_minute = (local.minute // cadence) * cadence
    slot = local.replace(minute=slot_minute, second=0, microsecond=0)
    date_value = event_date.isoformat() if isinstance(event_date, date) else str(event_date or local.date())
    material = "|".join(
        (
            date_value,
            str(location),
            str(market_kind),
            str(event_slug),
            slot.isoformat(),
        )
    )
    return f"la-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _status_bundle(context: Mapping[str, Any], context_json: Mapping[str, Any]) -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    direct = context.get("input_status")
    if isinstance(direct, Mapping):
        bundle.update(jsonable(direct))
    for key in (
        "status_contract_version",
        "numeric_policy",
        "status_policy",
        "decision_timestamp",
        "weather_input_status",
        "wind_input_status",
        "pressure_input_status",
        "forecast_input_status",
        "observation_buffer_status",
        "rain_input_status",
        "nowcast_input_status",
    ):
        if key in context_json and key not in bundle:
            bundle[key] = jsonable(context_json[key])
    return bundle


def _status_value(status_map: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        candidate = status_map.get(key)
        if isinstance(candidate, Mapping) and candidate.get("value") is not None:
            return candidate.get("value")
    return None


def _build_weather_state(context: Mapping[str, Any], context_json: Mapping[str, Any]) -> dict[str, Any]:
    direct = context.get("weather_state") or context.get("weather")
    if isinstance(direct, Mapping):
        weather = jsonable(direct)
        weather.setdefault("status", _status_bundle(context, context_json))
        return weather

    status = _status_bundle(context, context_json)
    weather_status = status.get("weather_input_status")
    if not isinstance(weather_status, Mapping):
        weather_status = {}
    forecast_status = status.get("forecast_input_status")
    if not isinstance(forecast_status, Mapping):
        forecast_status = {}
    wind_status = status.get("wind_input_status")
    if not isinstance(wind_status, Mapping):
        wind_status = {}
    pressure_status = status.get("pressure_input_status")
    if not isinstance(pressure_status, Mapping):
        pressure_status = {}
    observation_status = status.get("observation_buffer_status")
    if not isinstance(observation_status, Mapping):
        observation_status = {}

    # Context JSON contains the flattened canonical values.  Feature metadata
    # carries the phase-2A kwargs when a value was not flattened by the legacy
    # snapshot adapter.
    feature_metadata = context_json.get("feature_metadata")
    if not isinstance(feature_metadata, Mapping):
        feature_metadata = {}
    numeric_sources = [context_json, context, feature_metadata]
    for key in ("wind_kwargs", "pressure_kwargs"):
        candidate = feature_metadata.get(key)
        if isinstance(candidate, Mapping):
            numeric_sources.append(candidate)

    def value(*keys: str) -> Any:
        for source in numeric_sources:
            if isinstance(source, Mapping):
                found = _first(source, *keys)
                if found is not None:
                    return jsonable(found)
        found = _status_value(weather_status, *keys)
        if found is not None:
            return jsonable(found)
        found = _status_value(observation_status, *keys)
        return jsonable(found) if found is not None else None

    observations = {
        "temperature": value("temp_now", "temp_current", "temperature"),
        "humidity": value("rh_now", "rh_current", "humidity"),
        "pressure": value("pressure_current", "pressure"),
        "dew_point": value("dew_point_current", "dew_point"),
    }
    lags = {
        key: value(key)
        for key in ("temp_30m_ago", "temp_60m_ago", "temp_120m_ago")
    }
    trends = {
        key: value(key)
        for key in (
            "temp_change_30m",
            "temp_change_60m",
            "temp_volatility_60m",
            "temp_acceleration_60m",
            "rh_change_60m",
            "dew_point_change_60m",
            "dew_point_spread_change_60m",
            "time_since_max",
            "time_since_min",
        )
    }
    wind = {
        key: value(key)
        for key in (
            "wind_ref_mean",
            "wind_ref_max",
            "wind_victoria_harbour_mean",
            "wind_victoria_harbour_max",
            "wind_offshore_highland_mean",
            "wind_offshore_highland_max",
            "wind_highland_mean",
            "wind_highland_max",
            "wind_kings_park_current",
            "wind_all_change_60m",
        )
    }
    rain = {
        key: value(key)
        for key in (
            "rain_60m",
            "rain_120m",
            "rainfall_30m_filled",
            "rainfall_60m_filled",
            "rainfall_120m_filled",
            "rain_data_ok",
            "rain_regime",
            "rain_data_gap_flag",
        )
    }
    nowcast = {
        key: value(key)
        for key in ("rain_nc_sum_0_60m", "rain_nc_sum_0_120m", "rain_nowcast")
    }
    forecast = {
        "forecast_max": value("forecast_max", "forecast_tmax"),
        "forecast_min": value("forecast_min", "forecast_tmin"),
        "forecast_source": _first(context_json, "forecast_source") or status.get("forecast_source"),
        "forecast_issue_time": _first(context_json, "forecast_issue_time") or status.get("forecast_issue_time"),
        "forecast_target_date": _first(context_json, "forecast_target_date") or status.get("forecast_target_date"),
        "revision_history": jsonable(
            forecast_status.get("revision_history", forecast_status.get("forecast_revision_history", []))
        ),
    }
    uv = {
        key: value(key)
        for key in ("uv", "uv_index", "uv_max", "uv_now")
        if value(key) is not None
    }
    # Remove only absent convenience values; status maps remain complete and
    # truthful even when all corresponding numeric values are unavailable.
    weather = {
        "observations": {key: item for key, item in observations.items() if item is not None},
        "max_so_far": value("max_so_far", "max_since_midnight"),
        "min_so_far": value("min_so_far", "min_since_midnight"),
        "lags": {key: item for key, item in lags.items() if item is not None},
        "trends": {key: item for key, item in trends.items() if item is not None},
        "humidity": value("rh_now", "rh_current"),
        "pressure": value("pressure_current", "pressure"),
        "dew_point": value("dew_point_current", "dew_point"),
        "wind": {key: item for key, item in wind.items() if item is not None},
        "rain": {key: item for key, item in rain.items() if item is not None},
        "rain_nowcast": {key: item for key, item in nowcast.items() if item is not None},
        "forecast": forecast,
        "uv": uv,
        "status": jsonable(status),
    }
    return weather


def _status_summary(model: Mapping[str, Any], status: Mapping[str, Any]) -> dict[str, Any]:
    direct = model.get("model_input_status_summary") or model.get("input_status_summary")
    if isinstance(direct, Mapping):
        return jsonable(direct)
    missing = stale = fallback = 0
    flags: list[str] = []

    def walk(value: Any) -> None:
        nonlocal missing, stale, fallback
        if isinstance(value, Mapping):
            if value.get("is_missing") is True:
                missing += 1
            if value.get("is_stale") is True:
                stale += 1
            if value.get("is_fallback") is True:
                fallback += 1
            raw_flags = value.get("quality_flags")
            if isinstance(raw_flags, (list, tuple)):
                flags.extend(str(item) for item in raw_flags)
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(status)
    return {
        "status_contract_version": status.get("status_contract_version"),
        "missing_count": missing,
        "stale_count": stale,
        "fallback_count": fallback,
        "quality_flags": _unique(flags),
    }


def _normalise_model(
    model_name: str,
    model: Mapping[str, Any],
    *,
    lineage: Mapping[str, Any],
    context_json: Mapping[str, Any],
    status: Mapping[str, Any],
) -> dict[str, Any]:
    raw = model.get("raw") if isinstance(model.get("raw"), Mapping) else model
    model_lineage = lineage.get(model_name) if isinstance(lineage.get(model_name), Mapping) else {}
    feature_metadata = context_json.get("feature_metadata")
    if not isinstance(feature_metadata, Mapping):
        feature_metadata = {}
    numeric_default = context_json.get(f"{model_name}_numeric_features")
    diagnostic_default = context_json.get(f"{model_name}_features")
    numeric = model.get("numeric_features", raw.get("_numeric_features", numeric_default))
    diagnostic = model.get("diagnostic_features", raw.get("_features", diagnostic_default))
    quantiles = dict(model.get("quantiles")) if isinstance(model.get("quantiles"), Mapping) else {}
    for key, aliases in _MODEL_QUANTILE_ALIASES.items():
        if key not in quantiles:
            quantiles[key] = _first(model, *aliases) or _first(raw, *aliases)
    quantiles = {key: jsonable(value) for key, value in quantiles.items() if value is not None}
    classifier_probability = _first(
        model,
        "classifier_probability",
        "prob_max_reached",
        "prob_min_reached",
        "probability",
    )
    if classifier_probability is None:
        classifier_probability = _first(
            raw,
            "classifier_probability",
            "prob_max_reached",
            "prob_min_reached",
            "probability",
        )
    point_prediction = _first(model, "point_prediction", "mean", "post_mean")
    if point_prediction is None:
        point_prediction = _first(raw, "point_prediction", "mean", "post_mean", "pred_tmax_p50", "pred_morning_min_p50")
    probabilities = _first(model, "full_bucket_probabilities", "bucket_probabilities", "probs")
    if not isinstance(probabilities, Mapping):
        probabilities = {}
    model_version = _first(model, "model_version", "feature_version") or model_lineage.get("model_version")
    artifact_identity = _first(model, "artifact_identity") or model_lineage.get("artifact_identity")
    feature_spec = _first(model, "feature_spec", "feature_spec_path") or model_lineage.get("feature_spec_path")
    input_status = model.get("model_input_status")
    if not isinstance(input_status, Mapping):
        input_status = status
    result = {
        "model_name": str(model.get("model_name") or model_name),
        "model_version": jsonable(model_version),
        "artifact_identity": jsonable(artifact_identity),
        "feature_spec": jsonable(feature_spec),
        "numeric_features": jsonable(numeric if isinstance(numeric, Mapping) else {}),
        "diagnostic_features": jsonable(diagnostic if isinstance(diagnostic, Mapping) else {}),
        "quantiles": quantiles,
        "point_prediction": jsonable(point_prediction),
        "full_bucket_probabilities": jsonable(probabilities),
        "classifier_probability": jsonable(classifier_probability),
        "model_input_status_summary": _status_summary(model, input_status),
    }
    # Keep direct q10/q25/... fields because they are the stable schema fields;
    # ``quantiles`` is also retained to make future extensions non-breaking.
    result.update({key: result["quantiles"].get(key) for key in _QUANTILE_KEYS})
    return result


def _normalise_market(
    market: Mapping[str, Any],
    *,
    yes_book: Mapping[str, Any] | None,
    no_book: Mapping[str, Any] | None,
) -> dict[str, Any]:
    bucket = str(market.get("bucket") or "")
    outcomes = _parse_outcomes(market.get("explicit_outcomes", market.get("outcomes")))
    yes_asset = market.get("yes_asset_id")
    no_asset = market.get("no_asset_id")
    if yes_asset is None and isinstance(yes_book, Mapping):
        yes_asset = yes_book.get("asset_id")
    if no_asset is None and isinstance(no_book, Mapping):
        no_asset = no_book.get("asset_id")
    return {
        "market_id": jsonable(_first(market, "market_id", "id", "slug")),
        "condition_id": jsonable(_first(market, "condition_id", "conditionId")),
        "bucket": bucket,
        "explicit_outcomes": outcomes,
        "yes_token_id": jsonable(_first(market, "yes_token_id", "token_id")),
        "no_token_id": jsonable(_first(market, "no_token_id", "no_token_id")),
        "yes_asset_id": jsonable(yes_asset),
        "no_asset_id": jsonable(no_asset),
        "tick_size": jsonable(_first(market, "tick_size", "orderPriceMinTickSize")),
        "minimum_order_size": jsonable(
            _first(market, "minimum_order_size", "orderMinSize", "minimumOrderSize")
        ),
        "market_schema_version": jsonable(
            _first(market, "market_schema_version", "schema_version")
        ),
    }


def _normalise_levels(levels: Any) -> list[dict[str, float | None]]:
    if not isinstance(levels, (list, tuple)):
        return []
    result: list[dict[str, float | None]] = []
    for level in levels:
        if not isinstance(level, Mapping):
            continue
        result.append(
            {
                "price": _number(level.get("price")),
                "available_shares": _number(
                    level.get("available_shares", level.get("size"))
                ),
            }
        )
    return result


def _normalise_book(
    bucket: str,
    token_side: str,
    market: Mapping[str, Any],
    depth: Mapping[str, Any] | None,
    *,
    decision_iso: str | None,
) -> dict[str, Any]:
    raw = jsonable(depth) if isinstance(depth, Mapping) else None
    depth = depth if isinstance(depth, Mapping) else {}
    token_id = market.get("yes_token_id") if token_side == "YES" else market.get("no_token_id")
    asset_id = depth.get("asset_id", depth.get("token_id"))
    book_iso = _iso_datetime(depth.get("timestamp"), naive_timezone=timezone.utc)
    decision_dt = _parse_datetime(decision_iso, naive_timezone=timezone.utc)
    book_dt = _parse_datetime(book_iso, naive_timezone=timezone.utc)
    age = None
    if decision_dt is not None and book_dt is not None:
        age = (decision_dt - book_dt).total_seconds()
    if age is None:
        age = _number(depth.get("book_age_seconds"))
    source_errors = depth.get("validation_errors")
    if not isinstance(source_errors, list):
        source_errors = list(source_errors) if isinstance(source_errors, tuple) else []
    derived_errors: list[str] = []
    if not token_id:
        derived_errors.append("token_id_missing")
    if not asset_id:
        derived_errors.append("asset_id_missing")
    if not book_iso:
        derived_errors.append("book_timestamp_missing")
    if depth.get("fetch_cycle_id") in (None, ""):
        derived_errors.append("fetch_cycle_id_missing")
    if depth.get("source_name") in (None, ""):
        derived_errors.append("source_name_missing")
    if not isinstance(depth.get("bids", depth.get("top_bids", [])), (list, tuple)):
        derived_errors.append("bids_missing_or_not_list")
    if not isinstance(depth.get("asks", depth.get("top_asks", [])), (list, tuple)):
        derived_errors.append("asks_missing_or_not_list")
    errors = _unique([str(item) for item in [*source_errors, *derived_errors]])
    if not depth:
        validation_status = "missing"
    elif errors:
        validation_status = "invalid"
    else:
        validation_status = str(depth.get("validation_status") or "valid")
    return {
        "market_id": market.get("market_id"),
        "condition_id": market.get("condition_id"),
        "bucket": bucket,
        "token_side": token_side,
        "token_id": jsonable(token_id),
        "asset_id": jsonable(asset_id),
        "book_timestamp": book_iso,
        "decision_timestamp": decision_iso,
        "book_age_seconds": age,
        "source_name": jsonable(depth.get("source_name")),
        "fetch_cycle_id": jsonable(depth.get("fetch_cycle_id")),
        "tick_size": jsonable(_first(depth, "tick_size", "orderPriceMinTickSize") or market.get("tick_size")),
        "minimum_order_size": jsonable(
            _first(depth, "minimum_order_size", "min_order_size") or market.get("minimum_order_size")
        ),
        "bids": _normalise_levels(depth.get("bids", depth.get("top_bids", []))),
        "asks": _normalise_levels(depth.get("asks", depth.get("top_asks", []))),
        "validation_status": validation_status,
        "validation_errors": errors,
        "raw_book": raw,
    }


def _book_sources(context: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    sources: dict[tuple[str, str], Mapping[str, Any]] = {}
    direct = context.get("books") or context.get("clob_books")
    if isinstance(direct, Mapping):
        for bucket, value in direct.items():
            if isinstance(value, Mapping):
                for side, book in value.items():
                    side_name = str(side).upper()
                    if side_name in {"YES", "NO"} and isinstance(book, Mapping):
                        sources[(str(bucket), side_name)] = book
            elif isinstance(bucket, tuple) and len(bucket) == 2 and isinstance(value, Mapping):
                sources[(str(bucket[0]), str(bucket[1]).upper())] = value
    elif isinstance(direct, (list, tuple)):
        for book in direct:
            if isinstance(book, Mapping) and book.get("bucket") and book.get("token_side"):
                sources[(str(book["bucket"]), str(book["token_side"]).upper())] = book
    yes = context.get("market_depth")
    no = context.get("market_depth_no")
    if isinstance(yes, Mapping):
        for bucket, book in yes.items():
            if isinstance(book, Mapping):
                sources[(str(bucket), "YES")] = book
    if isinstance(no, Mapping):
        for bucket, book in no.items():
            if isinstance(book, Mapping):
                sources[(str(bucket), "NO")] = book
    return sources


def _build_market_and_books(context: Mapping[str, Any], decision_iso: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_markets = context.get("market_identities") or context.get("markets") or []
    if isinstance(raw_markets, Mapping):
        raw_markets = list(raw_markets.values())
    markets = [item for item in raw_markets if isinstance(item, Mapping)]
    sources = _book_sources(context)
    buckets = [str(market.get("bucket") or "") for market in markets if market.get("bucket")]
    for bucket, _side in sources:
        if bucket not in buckets:
            buckets.append(bucket)
    market_by_bucket: dict[str, dict[str, Any]] = {}
    for market in markets:
        bucket = str(market.get("bucket") or "")
        if not bucket:
            continue
        market_by_bucket[bucket] = _normalise_market(
            market,
            yes_book=sources.get((bucket, "YES")),
            no_book=sources.get((bucket, "NO")),
        )
    for bucket in buckets:
        market_by_bucket.setdefault(
            bucket,
            _normalise_market({"bucket": bucket}, yes_book=sources.get((bucket, "YES")), no_book=sources.get((bucket, "NO"))),
        )
    market_records = [market_by_bucket[bucket] for bucket in buckets]
    books: list[dict[str, Any]] = []
    for bucket in buckets:
        market = market_by_bucket[bucket]
        for side in ("YES", "NO"):
            books.append(
                _normalise_book(
                    bucket,
                    side,
                    market,
                    sources.get((bucket, side)),
                    decision_iso=decision_iso,
                )
            )
    return market_records, books


def _model_inputs(context: Mapping[str, Any], context_json: Mapping[str, Any], status: Mapping[str, Any]) -> list[dict[str, Any]]:
    lineage = status.get("model_lineage")
    if not isinstance(lineage, Mapping):
        lineage = context_json.get("model_lineage")
    if not isinstance(lineage, Mapping):
        lineage = context_json.get("feature_metadata", {}).get("model_lineage", {}) if isinstance(context_json.get("feature_metadata"), Mapping) else {}
    raw_models = context.get("model_states") or context.get("models")
    if isinstance(raw_models, Mapping):
        entries = [(str(key), value) for key, value in raw_models.items()]
    elif isinstance(raw_models, (list, tuple)):
        entries = [(str(value.get("model_name") or f"model_{index}"), value) for index, value in enumerate(raw_models) if isinstance(value, Mapping)]
    else:
        all_results = context.get("all_results") or context.get("model_outputs") or {}
        if isinstance(all_results, Mapping):
            entries = [(str(key), value) for key, value in all_results.items() if key not in {"_feature_metadata", "_intraday_error", "_error"} and isinstance(value, Mapping)]
        else:
            entries = []
    return [
        _normalise_model(name, value, lineage=lineage, context_json=context_json, status=status)
        for name, value in entries
    ]


def _market_kind(context: Mapping[str, Any]) -> str:
    explicit = context.get("market_kind")
    if explicit:
        return str(explicit)
    if context.get("is_min_temp"):
        return "lowest_temperature"
    return "highest_temperature"


def build_layer_a_record(
    context: Mapping[str, Any] | None = None,
    *,
    decision_cycle_id: str | None = None,
    schema_version: str = SCHEMA_VERSION,
    decision_timestamp: Any = None,
    capture_timestamp: Any = None,
    event_date: str | date | None = None,
    location: str = "Hong Kong",
    market_kind: str | None = None,
    event_slug: str | None = None,
    weather_state: Mapping[str, Any] | None = None,
    model_states: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    market_identities: Sequence[Mapping[str, Any]] | None = None,
    books: Any = None,
    gamma_reference_prices: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one complete-or-incomplete, strategy-independent Layer A record."""
    source: dict[str, Any] = dict(context or {})
    if weather_state is not None:
        source["weather_state"] = weather_state
    if model_states is not None:
        source["model_states"] = model_states
    if market_identities is not None:
        source["market_identities"] = market_identities
    if books is not None:
        source["books"] = books
    context_json = source.get("context_json")
    if not isinstance(context_json, Mapping):
        context_json = {}
    status = _status_bundle(source, context_json)
    decision_value = decision_timestamp or source.get("decision_timestamp") or context_json.get("decision_timestamp") or status.get("decision_timestamp")
    if decision_value is None:
        decision_value = datetime.now(timezone.utc)
    decision_iso = _iso_datetime(decision_value)
    if decision_iso is None:
        raise LayerASchemaError("decision_timestamp is invalid")
    event_date_value = event_date or source.get("event_date") or source.get("target_date_str") or context_json.get("snapshot_date")
    if isinstance(event_date_value, date):
        event_date_iso = event_date_value.isoformat()
    elif event_date_value:
        event_date_iso = str(event_date_value)
    else:
        event_date_iso = _parse_datetime(decision_value).astimezone(HKT).date().isoformat()  # type: ignore[union-attr]
    slug = event_slug or source.get("event_slug") or source.get("slug") or ""
    kind = market_kind or _market_kind(source)
    cycle_id = decision_cycle_id or source.get("decision_cycle_id")
    if not cycle_id:
        cycle_id = make_decision_cycle_id(
            decision_iso,
            event_date=event_date_iso,
            location=location,
            event_slug=str(slug),
            market_kind=kind,
        )
    weather = _build_weather_state(source, context_json)
    # Explicit direct model/market/book kwargs are installed into a temporary
    # source so the same normalization path is used for app and test payloads.
    models = _model_inputs(source, context_json, status)
    markets, book_records = _build_market_and_books(source, decision_iso)
    refs = gamma_reference_prices
    if refs is None:
        refs = source.get("gamma_reference_prices") or context_json.get("gamma_reference_prices") or context_json.get("market_prices") or {}
    record = {
        "decision_cycle_id": str(cycle_id),
        "schema_version": schema_version,
        "decision_timestamp": decision_iso,
        "capture_timestamp": _iso_datetime(capture_timestamp or datetime.now(timezone.utc), naive_timezone=timezone.utc),
        "event_date": event_date_iso,
        "location": str(location),
        "market_kind": str(kind),
        "event_slug": str(slug),
        "weather_state": weather,
        "models": models,
        "market_identity": markets,
        "clob_books": book_records,
        "gamma_reference_prices": jsonable(refs),
        "source_status": jsonable(status),
    }
    record["completeness"] = assess_completeness(record)
    validate_layer_a_record(record)
    return record


def _add_reason(reasons: list[str], value: str) -> None:
    if value not in reasons:
        reasons.append(value)


def assess_completeness(record: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate per-cycle completeness and exact missing-field reasons."""
    reasons: list[str] = []
    weather = record.get("weather_state")
    if not isinstance(weather, Mapping):
        _add_reason(reasons, "weather_state_missing")
        weather_complete = False
    else:
        observations = weather.get("observations")
        has_temperature = isinstance(observations, Mapping) and any(
            observations.get(key) is not None for key in ("temperature", "temp_current", "temp_now")
        )
        if not has_temperature:
            _add_reason(reasons, "weather.observations.temperature")
        if weather.get("max_so_far") is None:
            _add_reason(reasons, "weather.max_so_far")
        if weather.get("min_so_far") is None:
            _add_reason(reasons, "weather.min_so_far")
        if not isinstance(weather.get("status"), Mapping):
            _add_reason(reasons, "weather.status")
        weather_complete = has_temperature and weather.get("max_so_far") is not None and weather.get("min_so_far") is not None and isinstance(weather.get("status"), Mapping)

    models = record.get("models")
    model_state_complete = isinstance(models, list) and bool(models)
    if not model_state_complete:
        _add_reason(reasons, "models")
    else:
        for index, model in enumerate(models):
            prefix = f"models[{index}]"
            if not isinstance(model, Mapping):
                model_state_complete = False
                _add_reason(reasons, f"{prefix}")
                continue
            for field in ("model_name", "artifact_identity", "feature_spec", "point_prediction", "full_bucket_probabilities", "numeric_features", "diagnostic_features", "model_input_status_summary"):
                value = model.get(field)
                missing = value is None or value == {} or value == ""
                if missing:
                    model_state_complete = False
                    _add_reason(reasons, f"{prefix}.{field}")
            if not isinstance(model.get("full_bucket_probabilities"), Mapping) or not model.get("full_bucket_probabilities"):
                model_state_complete = False

    market_records = record.get("market_identity")
    market_identity_complete = isinstance(market_records, list) and bool(market_records)
    token_identity_complete = market_identity_complete
    if not market_identity_complete:
        _add_reason(reasons, "market_identity")
    else:
        for index, market in enumerate(market_records):
            prefix = f"market_identity[{index}]"
            if not isinstance(market, Mapping):
                market_identity_complete = False
                token_identity_complete = False
                _add_reason(reasons, prefix)
                continue
            for field in ("market_id", "condition_id", "bucket", "explicit_outcomes", "tick_size", "minimum_order_size", "market_schema_version"):
                if market.get(field) in (None, "", []):
                    market_identity_complete = False
                    _add_reason(reasons, f"{prefix}.{field}")
            outcomes = [str(item).lower() for item in market.get("explicit_outcomes", [])] if isinstance(market.get("explicit_outcomes"), list) else []
            if outcomes != ["yes", "no"]:
                market_identity_complete = False
                _add_reason(reasons, f"{prefix}.explicit_outcomes")
            for field in ("yes_token_id", "no_token_id"):
                if market.get(field) in (None, ""):
                    token_identity_complete = False
                    _add_reason(reasons, f"{prefix}.{field}")

    books = record.get("clob_books")
    books_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    if not isinstance(books, list):
        books = []
    for index, book in enumerate(books):
        if not isinstance(book, Mapping):
            _add_reason(reasons, f"clob_books[{index}]")
            continue
        key = (str(book.get("bucket") or ""), str(book.get("token_side") or "").upper())
        books_by_key[key] = book
        if not book.get("token_id"):
            token_identity_complete = False
            _add_reason(reasons, f"clob_books[{index}].token_id")
        if not book.get("asset_id"):
            token_identity_complete = False
            _add_reason(reasons, f"clob_books[{index}].asset_id")

    depth_pair_complete = True
    book_timestamp_complete = True
    fetch_cycle_coherent = True
    cycles: set[str] = set()
    expected_buckets = [str(item.get("bucket")) for item in market_records or [] if isinstance(item, Mapping)]
    for bucket in expected_buckets:
        for side in ("YES", "NO"):
            book = books_by_key.get((bucket, side))
            prefix = f"clob_books[{bucket}/{side}]"
            if book is None:
                depth_pair_complete = False
                book_timestamp_complete = False
                fetch_cycle_coherent = False
                _add_reason(reasons, prefix)
                continue
            if book.get("validation_status") != "valid":
                depth_pair_complete = False
                _add_reason(reasons, prefix)
                _add_reason(reasons, f"{prefix}.validation_status")
            if not isinstance(book.get("bids"), list) or not isinstance(book.get("asks"), list):
                depth_pair_complete = False
                _add_reason(reasons, f"{prefix}.depth")
            if not book.get("book_timestamp") or not isinstance(book.get("book_age_seconds"), (int, float)) or float(book.get("book_age_seconds")) < 0:
                book_timestamp_complete = False
                _add_reason(reasons, f"{prefix}.book_timestamp")
            cycle = book.get("fetch_cycle_id")
            if not cycle:
                fetch_cycle_coherent = False
                _add_reason(reasons, f"{prefix}.fetch_cycle_id")
            else:
                cycles.add(str(cycle))
    if len(cycles) > 1:
        fetch_cycle_coherent = False
        _add_reason(reasons, "fetch_cycle_id_incoherent")
    if not books and expected_buckets:
        depth_pair_complete = False
        book_timestamp_complete = False
        fetch_cycle_coherent = False
        _add_reason(reasons, "clob_books")

    replay_model = bool(weather_complete and model_state_complete)
    replay_clob = bool(
        replay_model
        and market_identity_complete
        and token_identity_complete
        and depth_pair_complete
        and book_timestamp_complete
        and fetch_cycle_coherent
    )
    rejection_reasons = list(reasons)
    if not replay_model:
        _add_reason(rejection_reasons, "replay_model_analysis_requirements_not_met")
    if not replay_clob:
        _add_reason(rejection_reasons, "replay_clob_strategy_requirements_not_met")
    return {
        "weather_complete": bool(weather_complete),
        "model_state_complete": bool(model_state_complete),
        "market_identity_complete": bool(market_identity_complete),
        "token_identity_complete": bool(token_identity_complete),
        "depth_pair_complete": bool(depth_pair_complete),
        "book_timestamp_complete": bool(book_timestamp_complete),
        "fetch_cycle_coherent": bool(fetch_cycle_coherent),
        "replay_eligible_for_model_analysis": replay_model,
        "replay_eligible_for_clob_strategy": replay_clob,
        "missing_fields": list(reasons),
        "rejection_reasons": rejection_reasons,
    }


def validate_layer_a_record(record: Mapping[str, Any]) -> None:
    """Validate envelope invariants while allowing incomplete cycles."""
    if not isinstance(record, Mapping):
        raise LayerASchemaError("Layer A record must be a mapping")
    for field in (
        "decision_cycle_id",
        "schema_version",
        "decision_timestamp",
        "capture_timestamp",
        "event_date",
        "location",
        "market_kind",
        "event_slug",
        "weather_state",
        "models",
        "market_identity",
        "clob_books",
        "completeness",
    ):
        if field not in record:
            raise LayerASchemaError(f"required Layer A field is missing: {field}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise LayerASchemaError(f"unsupported Layer A schema: {record.get('schema_version')!r}")
    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                field = str(key).strip().lower()
                child_path = f"{path}.{key}" if path else str(key)
                if field in _MODEL_PROHIBITED_FIELDS:
                    raise LayerASchemaError(
                        f"strategy/account field is not allowed in Layer A: {child_path}"
                    )
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(record)
    for timestamp_field in ("decision_timestamp", "capture_timestamp"):
        if _parse_datetime(record.get(timestamp_field), naive_timezone=timezone.utc) is None:
            raise LayerASchemaError(f"invalid Layer A timestamp: {timestamp_field}")
    try:
        json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LayerASchemaError(f"Layer A record is not JSON serializable: {exc}") from exc


__all__ = [
    "SCHEMA_VERSION",
    "LayerASchemaError",
    "assess_completeness",
    "build_layer_a_record",
    "jsonable",
    "make_decision_cycle_id",
    "validate_layer_a_record",
]
