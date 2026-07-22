"""Shared upstream canonical sampling-cycle builder.

Both the background scheduler and the manual strategy runner use this module.
The cache key is a deterministic market-cycle slot, never an account ID.  A
single frozen payload owns the weather/model/market/depth objects; downstream
account contexts only select from that payload and add derived execution
parameters.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping

from execution.strategy_account import StrategyAccount
from layer_a.schema import build_layer_a_record, jsonable, make_decision_cycle_id

logger = logging.getLogger(__name__)


class CanonicalCycleUnavailable(RuntimeError):
    """Raised when the upstream cycle cannot provide market/model inputs."""


@dataclass(frozen=True)
class CanonicalCycle:
    """One process-local frozen canonical sampling payload.

    The contained service mappings are treated as read-only by downstream
    adapters; the frozen dataclass also prevents replacing cycle fields after
    capture.  Account contexts receive references to these same mappings.
    """

    decision_cycle_id: str
    decision_timestamp: datetime
    event_date: date
    event_slug: str
    is_min_temp: bool
    markets: list[dict[str, Any]]
    state: dict[str, Any]
    hko: dict[str, Any]
    rain_kwargs: dict[str, Any]
    all_results: dict[str, Any]
    market_depth: dict[str, Any]
    market_depth_no: dict[str, Any]
    depth_fetch_cycle_id: str | None
    execution_snapshots: dict[str, dict[str, Any]]
    execution_snapshot_error: str | None
    paper_execution_mode: str
    partial_fill_policy: str
    context_json: dict[str, Any]
    gamma_reference_prices: dict[str, Any]
    market_snapshot_id: str | None
    weather_snapshot_id: str | None
    layer_a_record: dict[str, Any]


_CACHE: dict[str, CanonicalCycle] = {}
_SLUG_CACHE: dict[tuple[str, bool], str] = {}
_CACHE_LOCK = threading.RLock()


def _cycle_slot_key(decision_timestamp: datetime, target_date: date, event_slug: str, is_min_temp: bool) -> str:
    cycle_id = make_decision_cycle_id(
        decision_timestamp,
        event_date=target_date,
        location="Hong Kong",
        event_slug=event_slug,
        market_kind="lowest_temperature" if is_min_temp else "highest_temperature",
    )
    return cycle_id


def _model_results_for_clob(results: Mapping[str, Any]) -> dict[str, float]:
    for key, value in results.items():
        if str(key).startswith("_") or not isinstance(value, Mapping):
            continue
        probabilities = value.get("probs")
        if isinstance(probabilities, Mapping) and probabilities:
            return {str(bucket): float(probability) for bucket, probability in probabilities.items()}
    return {}


def _context_json(
    *,
    state: Mapping[str, Any],
    hko: Mapping[str, Any],
    rain_kwargs: Mapping[str, Any],
    results: Mapping[str, Any],
    markets: list[Mapping[str, Any]],
    market_depth: Mapping[str, Any],
    market_depth_no: Mapping[str, Any],
    depth_fetch_cycle_id: str | None,
    execution_snapshot_error: str | None,
    execution_snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key in (
        "temp_now",
        "temp_30m_ago",
        "temp_60m_ago",
        "temp_120m_ago",
        "min_so_far",
        "rh_now",
        "temp_change_30m",
        "temp_change_60m",
        "time_since_max",
        "time_since_min",
        "temp_volatility_60m",
        "temp_acceleration_60m",
        "rh_change_60m",
        "dew_point_change_60m",
        "dew_point_spread_change_60m",
        "pressure_current",
        "dew_point_current",
    ):
        value = state.get(key)
        if value is not None:
            context[key] = value
    for key in ("max_since_midnight", "min_since_midnight", "forecast_max", "forecast_min"):
        value = hko.get(key)
        if value is not None:
            context[key] = value
    for key in (
        "rain_60m",
        "rain_120m",
        "rain_data_ok",
        "rainfall_60m_missing_flag",
        "rainfall_120m_missing_flag",
        "rainfall_30m_missing_flag",
        "rainfall_data_age_minutes",
        "rain_data_gap_flag",
        "rain_regime",
        "rainfall_60m",
        "rainfall_120m",
        "has_recent_rainfall_obs",
        "rain_intensity_max_120m",
        "rain_cooling_60m",
        "rain_after_max_flag",
        "post_peak_rain_flag",
    ):
        value = rain_kwargs.get(key)
        if value is not None:
            context[key] = value

    frame = state.get("df_today")
    if frame is not None:
        try:
            context["buffer_len"] = len(frame)
            if len(frame) >= 30:
                context["temp_at_idx30"] = float(frame["temp"].iloc[-30])
            if len(frame) >= 60:
                context["temp_at_idx60"] = float(frame["temp"].iloc[-60])
                if "rh" in frame.columns and frame["rh"].iloc[-60] is not None:
                    context["rh_at_idx60"] = float(frame["rh"].iloc[-60])
        except (KeyError, TypeError, ValueError, IndexError):
            context["buffer_len"] = 0

    model_stds = {
        str(key): value.get("std")
        for key, value in results.items()
        if not str(key).startswith("_") and isinstance(value, Mapping) and value.get("std") is not None
    }
    if model_stds:
        context["model_stds"] = model_stds
    model_probs = {
        str(key): value.get("probs")
        for key, value in results.items()
        if not str(key).startswith("_") and isinstance(value, Mapping) and value.get("probs")
    }
    if model_probs:
        context["model_probs"] = model_probs
    prices = {str(market.get("bucket")): market.get("yes_price") for market in markets if market.get("bucket")}
    context["market_prices"] = prices
    context["gamma_reference_prices"] = dict(prices)
    if market_depth:
        context["market_depth"] = market_depth
    if market_depth_no:
        context["market_depth_no"] = market_depth_no
    if depth_fetch_cycle_id:
        context["depth_fetch_cycle_id"] = depth_fetch_cycle_id
    if execution_snapshot_error:
        context["execution_snapshot_error"] = execution_snapshot_error
    if execution_snapshots:
        context["clob_execution_snapshots"] = {
            bucket: {side: jsonable(snapshot) for side, snapshot in side_map.items()}
            for bucket, side_map in execution_snapshots.items()
        }

    gamma_info: dict[str, Any] = {}
    for market in markets:
        bucket = market.get("bucket")
        if not bucket:
            continue
        fields = {}
        for key in (
            "token_id",
            "no_token_id",
            "conditionId",
            "condition_id",
            "id",
            "slug",
            "outcomes",
            "orderPriceMinTickSize",
            "orderMinSize",
            "minimumOrderSize",
            "market_schema_version",
            "bestBid",
            "bestAsk",
            "spread",
            "lastTradePrice",
            "liquidityClob",
            "volume24hrClob",
        ):
            if market.get(key) is not None:
                fields[key] = market[key]
        if fields:
            gamma_info[str(bucket)] = fields
    if gamma_info:
        context["gamma_market_info"] = gamma_info

    for model_key, prediction in results.items():
        if str(model_key).startswith("_") or not isinstance(prediction, Mapping):
            continue
        raw = prediction.get("raw") if isinstance(prediction.get("raw"), Mapping) else prediction
        if isinstance(raw, Mapping):
            diagnostic = raw.get("_features") or prediction.get("diagnostic_features")
            numeric = raw.get("_numeric_features") or prediction.get("numeric_features")
            if diagnostic:
                context[f"{model_key}_features"] = diagnostic
            if numeric:
                context[f"{model_key}_numeric_features"] = numeric

    metadata = results.get("_feature_metadata")
    if isinstance(metadata, Mapping):
        from features.input_status import attach_status_metadata_to_context

        attach_status_metadata_to_context(context, metadata)
    return jsonable(context)


def _state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "temp_now",
        "max_so_far",
        "min_so_far",
        "rh_now",
        "temp_30m_ago",
        "temp_60m_ago",
        "temp_120m_ago",
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
    return {key: jsonable(state.get(key)) for key in keys if state.get(key) is not None}


def _build_layer_a_record(cycle_payload: Mapping[str, Any]) -> dict[str, Any]:
    record = build_layer_a_record(
        cycle_payload,
        decision_cycle_id=cycle_payload["decision_cycle_id"],
        decision_timestamp=cycle_payload["decision_timestamp"],
        event_date=cycle_payload["event_date"],
        location="Hong Kong",
        market_kind=cycle_payload["market_kind"],
        event_slug=cycle_payload["event_slug"],
        gamma_reference_prices=cycle_payload.get("gamma_reference_prices", {}),
    )
    return record


def _persist_layer_a(record: Mapping[str, Any]) -> None:
    try:
        from layer_a.storage import get_default_store

        get_default_store().capture(record)
    except Exception:
        # Capture must not stop paper strategy execution.  The local health
        # surface still shows a missing cycle when storage did not close.
        logger.exception("Layer A capture failed for cycle %s", record["decision_cycle_id"])


def _capture_linked_layer_a_snapshots(
    *,
    cycle_id: str,
    decision_timestamp: datetime,
    target_date: date,
    event_slug: str,
    is_min_temp: bool,
    markets: list[Mapping[str, Any]],
    state: Mapping[str, Any],
    rain_kwargs: Mapping[str, Any],
    context_json: Mapping[str, Any],
    market_depth: Mapping[str, Any],
    market_depth_no: Mapping[str, Any],
    depth_fetch_cycle_id: str | None,
    gamma_reference_prices: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Persist and return the immutable market/weather IDs for this cycle."""
    market_snapshot_id: str | None = None
    weather_snapshot_id: str | None = None
    market_kind = "lowest_temperature" if is_min_temp else "highest_temperature"
    try:
        from layer_a.market_schema import build_market_snapshot
        from layer_a.market_storage import get_default_market_store

        market_snapshot = build_market_snapshot(
            decision_timestamp=decision_timestamp,
            capture_timestamp=decision_timestamp,
            event_date=target_date,
            location="Hong Kong",
            event_slug=event_slug,
            market_kind=market_kind,
            markets=markets,
            market_depth=market_depth,
            market_depth_no=market_depth_no,
            fetch_cycle_id=depth_fetch_cycle_id or f"cycle:{cycle_id}",
            latest_model_cycle_id=cycle_id,
            latest_model_cycle_timestamp=decision_timestamp,
            gamma_reference_prices=gamma_reference_prices,
            gamma_reference_data={
                str(market.get("bucket")): jsonable(dict(market))
                for market in markets
                if market.get("bucket")
            },
            source_status={
                "collector": "canonical_cycle_link",
                "market_source": "polymarket_gamma_reference",
                "book_source": "polymarket_clob_cache",
                "fetch_cycle_id": depth_fetch_cycle_id,
            },
        )
        result = get_default_market_store().capture(market_snapshot)
        market_snapshot_id = str(result.market_snapshot_id)
    except Exception:
        logger.exception("Linked market snapshot capture failed for cycle %s", cycle_id)

    try:
        from layer_a.schema import _parse_datetime
        from layer_a.weather_capture import _ensure_weather_fields
        from layer_a.weather_schema import build_weather_snapshot
        from layer_a.weather_storage import get_default_weather_store

        weather_state = dict(state)
        weather_state.update(
            {
                key: context_json[key]
                for key in ("pressure_current", "dew_point_current")
                if context_json.get(key) is not None
            }
        )
        if rain_kwargs.get("rain_current") is not None:
            weather_state["rain_current"] = rain_kwargs.get("rain_current")
        weather_state["observation_buffer_status"] = {
            **dict(state.get("observation_buffer_status") or {}),
            **dict((context_json.get("observation_buffer_status") or {}) if isinstance(context_json.get("observation_buffer_status"), Mapping) else {}),
        }
        weather_state["weather_input_status"] = {
            **dict(state.get("weather_input_status") or {}),
            **dict((context_json.get("weather_input_status") or {}) if isinstance(context_json.get("weather_input_status"), Mapping) else {}),
        }
        observation_timestamp = _parse_datetime(
            state.get("time_now") or state.get("decision_timestamp") or decision_timestamp,
            naive_timezone=timezone.utc,
        ) or decision_timestamp
        weather_state = _ensure_weather_fields(
            weather_state,
            observation_timestamp=observation_timestamp,
        )
        model_timestamp = min(decision_timestamp, observation_timestamp)
        weather_snapshot = build_weather_snapshot(
            snapshot_timestamp=observation_timestamp,
            capture_timestamp=decision_timestamp,
            event_date=target_date,
            location="Hong Kong",
            weather_state=weather_state,
            latest_model_cycle_id=cycle_id,
            model_cycle_timestamp=model_timestamp,
            source_status={
                "collector": "canonical_cycle_link",
                "observation_source": "hko_intraday_state",
                "model_link_source": "canonical_cycle",
                "clob_fetched": False,
            },
        )
        result = get_default_weather_store().capture(weather_snapshot)
        weather_snapshot_id = str(result.weather_snapshot_id)
    except Exception:
        logger.exception("Linked weather snapshot capture failed for cycle %s", cycle_id)
    return market_snapshot_id, weather_snapshot_id


def _build_uncached_cycle(
    *,
    target_date: date,
    is_min_temp: bool,
    event_slug: str,
    decision_timestamp: datetime,
) -> CanonicalCycle:
    from app.services.market_depth_service import get_global_depth_cache
    from app.services.market_service import fetch_event_markets
    from app.services.model_service import run_all_models
    from app.services.weather_service import (
        compute_rain_kwargs,
        fetch_hko_data,
        get_intraday_state,
        hkt_now,
    )
    from execution.clob_execution import SnapshotValidationError, build_execution_snapshots
    from execution.paper_execution_config import (
        get_max_book_age_seconds,
        get_paper_execution_mode,
        get_partial_fill_policy,
    )

    target_date_str = target_date.isoformat()
    day_key = target_date_str.replace("-", "")
    markets = fetch_event_markets(event_slug, is_min_temp=is_min_temp)
    if not markets:
        raise CanonicalCycleUnavailable(f"No markets found for {event_slug}")
    hko = fetch_hko_data(target_date_str)
    state = get_intraday_state(day_key)
    if not state:
        raise CanonicalCycleUnavailable(f"No intraday state found for {target_date_str}")
    max_so_far = state.get("max_so_far")
    temp_now = state.get("temp_now")
    drop_from_max = (max_so_far - temp_now) if max_so_far is not None and temp_now is not None else 0.0
    rain_kwargs = compute_rain_kwargs(
        day_key,
        hkt_now(),
        drop_from_max=drop_from_max,
        temp_change_60m=state.get("temp_change_60m", 0.0),
    )
    forecast_key = "forecast_min" if is_min_temp else "forecast_max"

    # Canonical model generation is intentionally independent of account
    # params.  Account-specific strategy values are retained only in the
    # downstream context adapter and never enter the Layer A record.
    results = run_all_models(
        target_date=target_date,
        target_date_str=target_date_str,
        is_min_temp=is_min_temp,
        bias=0.0,
        std_mult=1.0,
        state=state,
        rain_kwargs=rain_kwargs,
        markets=markets,
        forecast_aws_val=hko.get(forecast_key) if hko else None,
        forecast_max=hko.get("forecast_max") if hko else None,
        forecast_min=hko.get("forecast_min") if hko else None,
        is_today=True,
        input_status={
            "forecast_input_status": hko.get("forecast_input_status") if hko else None,
            "forecast_source": hko.get("forecast_source") if hko else None,
            "forecast_issue_time": hko.get("forecast_issue_time") if hko else None,
            "forecast_target_date": hko.get("forecast_target_date") if hko else None,
        },
    )

    prices = {str(market["bucket"]): market.get("yes_price", 0.5) for market in markets}
    yes_tokens = {str(market["bucket"]): market.get("token_id", "") for market in markets}
    no_tokens = {str(market["bucket"]): market.get("no_token_id", "") for market in markets}
    depth_cache = get_global_depth_cache()
    depth_cache.update_token_ids(
        {bucket: token for bucket, token in yes_tokens.items() if token},
        {bucket: token for bucket, token in no_tokens.items() if token},
    )
    market_depth, market_depth_no, depth_fetch_cycle_id = depth_cache.get_bundle()

    snapshots: dict[str, dict[str, Any]] = {}
    snapshot_error: str | None = None
    target_probs = _model_results_for_clob(results)
    try:
        snapshots = build_execution_snapshots(
            markets=markets,
            target_probs=target_probs,
            market_depth=market_depth,
            market_depth_no=market_depth_no,
            event_slug=event_slug,
            decision_timestamp=decision_timestamp,
            expected_market_date=target_date,
            fetch_cycle_id=depth_fetch_cycle_id,
            max_book_age_seconds=get_max_book_age_seconds(),
            is_min_temp=is_min_temp,
        )
    except SnapshotValidationError as exc:
        snapshot_error = str(exc)
        logger.warning("Canonical CLOB snapshot rejected for %s: %s", event_slug, exc)

    context_json = _context_json(
        state=state,
        hko=hko or {},
        rain_kwargs=rain_kwargs,
        results=results,
        markets=markets,
        market_depth=market_depth,
        market_depth_no=market_depth_no,
        depth_fetch_cycle_id=depth_fetch_cycle_id,
        execution_snapshot_error=snapshot_error,
        execution_snapshots=snapshots,
    )
    paper_execution_mode = get_paper_execution_mode()
    partial_fill_policy = get_partial_fill_policy()
    cycle_id = _cycle_slot_key(decision_timestamp, target_date, event_slug, is_min_temp)
    market_snapshot_id, weather_snapshot_id = _capture_linked_layer_a_snapshots(
        cycle_id=cycle_id,
        decision_timestamp=decision_timestamp,
        target_date=target_date,
        event_slug=event_slug,
        is_min_temp=is_min_temp,
        markets=markets,
        state=state,
        rain_kwargs=rain_kwargs,
        context_json=context_json,
        market_depth=market_depth,
        market_depth_no=market_depth_no,
        depth_fetch_cycle_id=depth_fetch_cycle_id,
        gamma_reference_prices=prices,
    )
    context_json["market_snapshot_id"] = market_snapshot_id
    context_json["weather_snapshot_id"] = weather_snapshot_id
    context_json["layer_a_linkage"] = {
        "market_snapshot_id": market_snapshot_id,
        "weather_snapshot_id": weather_snapshot_id,
        "linkage_status": "complete" if market_snapshot_id and weather_snapshot_id else "incomplete",
    }
    context_json["paper_execution_mode"] = paper_execution_mode
    payload: dict[str, Any] = {
        "decision_cycle_id": cycle_id,
        "schema_version": "layer_a.v1",
        "decision_timestamp": decision_timestamp,
        "event_date": target_date,
        "location": "Hong Kong",
        "market_kind": "lowest_temperature" if is_min_temp else "highest_temperature",
        "event_slug": event_slug,
        "market_snapshot_id": market_snapshot_id,
        "weather_snapshot_id": weather_snapshot_id,
        "markets": markets,
        "market_depth": market_depth,
        "market_depth_no": market_depth_no,
        "all_results": results,
        "clob_books": [
            snapshot.to_dict()
            for side_map in snapshots.values()
            for snapshot in side_map.values()
        ],
        "execution_snapshots": snapshots,
        "execution_snapshot_error": snapshot_error,
        "paper_execution_mode": paper_execution_mode,
        "partial_fill_policy": partial_fill_policy,
        "context_json": context_json,
        "gamma_reference_prices": prices,
    }
    # Build the normalized record before freezing the cycle, but persist only
    # after the frozen object exists.  The ordering is therefore explicit:
    # complete inputs -> freeze -> one Layer A write -> account adapters.
    layer_record = _build_layer_a_record(payload)
    cycle = CanonicalCycle(
        decision_cycle_id=cycle_id,
        decision_timestamp=decision_timestamp,
        event_date=target_date,
        event_slug=event_slug,
        is_min_temp=is_min_temp,
        markets=markets,
        state=state,
        hko=hko or {},
        rain_kwargs=rain_kwargs,
        all_results=results,
        market_depth=market_depth,
        market_depth_no=market_depth_no,
        depth_fetch_cycle_id=depth_fetch_cycle_id,
        execution_snapshots=snapshots,
        execution_snapshot_error=snapshot_error,
        paper_execution_mode=payload["paper_execution_mode"],
        partial_fill_policy=payload["partial_fill_policy"],
        context_json=context_json,
        gamma_reference_prices=prices,
        market_snapshot_id=market_snapshot_id,
        weather_snapshot_id=weather_snapshot_id,
        layer_a_record=layer_record,
    )
    _persist_layer_a(cycle.layer_a_record)
    return cycle


def get_canonical_cycle(
    *,
    is_min_temp: bool,
    target_date: date | None = None,
    event_slug: str | None = None,
) -> CanonicalCycle:
    """Return the shared cycle for the current deterministic sampling slot."""
    from app.services.market_service import resolve_event_slug_for_kind, market_kind_from_slug
    from app.services.weather_service import hkt_now

    decision_timestamp = datetime.now(timezone.utc)
    target = target_date or hkt_now().date()
    target_date_str = target.isoformat()
    slug_key = (target_date_str, is_min_temp)
    with _CACHE_LOCK:
        slug = event_slug or _SLUG_CACHE.get(slug_key)
    if slug and market_kind_from_slug(slug) not in (None, "lowest_temperature" if is_min_temp else "highest_temperature"):
        logger.error("Rejecting canonical event slug %s for market kind %s", slug, "lowest_temperature" if is_min_temp else "highest_temperature")
        slug = None
    if not slug:
        slug = resolve_event_slug_for_kind(target, is_min_temp=is_min_temp)
        with _CACHE_LOCK:
            _SLUG_CACHE[slug_key] = slug
    cycle_id = _cycle_slot_key(decision_timestamp, target, slug, is_min_temp)
    with _CACHE_LOCK:
        cached = _CACHE.get(cycle_id)
        if cached is not None:
            return cached
        cycle = _build_uncached_cycle(
            target_date=target,
            is_min_temp=is_min_temp,
            event_slug=slug,
            decision_timestamp=decision_timestamp,
        )
        _CACHE[cycle_id] = cycle
        return cycle


def clear_canonical_cycle_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
        _SLUG_CACHE.clear()


def _account_model_view(cycle: CanonicalCycle, acct: StrategyAccount) -> Mapping[str, Any]:
    """Apply legacy 9d/AWS profile knobs only at the account adapter.

    The canonical record remains neutral and strategy-independent.  The old
    runtime allowed ``bias`` and ``std_mult`` to affect the 9d/AWS outputs;
    reproducing that derived view here preserves those settings without
    rerunning weather, models or CLOB fetches per account.  Intraday models
    were not changed by those two knobs in the old pipeline.
    """
    results = cycle.all_results
    selected = results.get(acct.model, {}) if isinstance(results, Mapping) else {}
    if not isinstance(selected, Mapping) or acct.model not in {"9d", "aws"}:
        return selected if isinstance(selected, Mapping) else {}

    params = acct.params or {}
    try:
        bias = float(params.get("bias", 0.0))
        std_mult = float(params.get("std_mult", 1.0))
        base_mean = float(selected["mean"])
        base_std = float(selected["std"])
    except (KeyError, TypeError, ValueError):
        return selected

    if bias == 0.0 and std_mult == 1.0:
        return selected
    from app.services.model_service import compute_bucket_probs

    mean = base_mean + bias
    std = base_std * std_mult
    probabilities = compute_bucket_probs(
        mean,
        std,
        cycle.markets,
        is_today=True,
        is_min_temp=cycle.is_min_temp,
        max_sf=cycle.state.get("max_so_far"),
        min_sf=cycle.state.get("min_so_far"),
    )
    derived = dict(selected)
    derived.update({"mean": mean, "std": std, "probs": probabilities or selected.get("probs", {})})
    return derived


def build_strategy_context_from_cycle(cycle: CanonicalCycle, acct: StrategyAccount) -> dict[str, Any]:
    """Adapt the shared cycle to one strategy without changing its payload."""
    params = acct.params or {}
    results = cycle.all_results
    selected = _account_model_view(cycle, acct)
    target_probs = selected.get("probs", {}) if isinstance(selected, Mapping) else {}
    if not target_probs:
        logger.error("Model %s produced no probs for strategy %s", acct.model, acct.id)
        return {}
    prices_dict = {str(market["bucket"]): market.get("yes_price", 0.5) for market in cycle.markets}
    token_ids = {str(market["bucket"]): market.get("token_id", "") for market in cycle.markets}
    no_token_ids = {str(market["bucket"]): market.get("no_token_id", "") for market in cycle.markets}
    model_std = selected.get("std", 1.5) if isinstance(selected, Mapping) else 1.5
    post_mean = selected.get("mean") if isinstance(selected, Mapping) else None
    execution_snapshots = cycle.execution_snapshots
    context_json = cycle.context_json
    # Downstream consumers receive the exact shared depth/snapshot objects.
    return {
        "capital": acct.capital,
        "model_key": acct.model,
        "mock_slippage": cycle.paper_execution_mode == "legacy_gamma_mock",
        "paper_execution_mode": cycle.paper_execution_mode,
        "partial_fill_policy": cycle.partial_fill_policy,
        "bias": params.get("bias", 0.0),
        "std_mult": params.get("std_mult", 1.0),
        "kelly_fraction": params.get("kelly_fraction", 0.25),
        "slug": cycle.event_slug,
        "target_probs": target_probs,
        "prices_dict": prices_dict,
        "gamma_reference_prices": cycle.gamma_reference_prices,
        "token_ids_dict": token_ids,
        "no_token_ids_dict": no_token_ids,
        "market_depth": cycle.market_depth,
        "market_depth_no": cycle.market_depth_no,
        "depth_fetch_cycle_id": cycle.depth_fetch_cycle_id,
        "execution_snapshots": execution_snapshots,
        "execution_snapshot_error": cycle.execution_snapshot_error,
        "temp_now": cycle.state.get("temp_now"),
        "max_so_far": cycle.state.get("max_so_far"),
        "rain_regime": cycle.rain_kwargs.get("rain_regime", "no_rain"),
        "model_std": model_std if model_std is not None else 1.5,
        "recent_price_volatility": 0.0,
        "hours_to_settlement": 24.0,
        "nowcast_stale": False,
        "data_missing": False,
        "drawdown_pct": 0.0,
        "markets": cycle.markets,
        "post_mean": post_mean,
        "is_min_temp": cycle.is_min_temp,
        "target_date_str": cycle.event_date.isoformat(),
        "all_results": results,
        "context_json": context_json,
        "decision_cycle_id": cycle.decision_cycle_id,
        "market_snapshot_id": getattr(cycle, "market_snapshot_id", None),
        "weather_snapshot_id": getattr(cycle, "weather_snapshot_id", None),
        "canonical_cycle": cycle,
    }


def build_strategy_context(acct: StrategyAccount) -> dict[str, Any]:
    cycle = get_canonical_cycle(is_min_temp=acct.market_template == "hk-tmin")
    return build_strategy_context_from_cycle(cycle, acct)


__all__ = [
    "CanonicalCycle",
    "CanonicalCycleUnavailable",
    "build_strategy_context",
    "build_strategy_context_from_cycle",
    "clear_canonical_cycle_cache",
    "get_canonical_cycle",
]
