"""Canonical CLOB depth contract, validation, walking and size-aware sizing.

This module is deliberately independent from Gamma/UI pricing.  Gamma prices
may be passed to the sizing function only as a diagnostic reference; they are
never used to choose a quote, size an order, simulate a fill, or value a
liquidation.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping

SCHEMA_VERSION = "clob_execution.v1"
SOURCE_NAME = "polymarket_clob"
FEE_RATE = 0.05
EPSILON = 1e-9


class CLOBExecutionError(ValueError):
    """Base error for an invalid or non-executable CLOB snapshot."""


class SnapshotValidationError(CLOBExecutionError):
    """Raised when market/token/book metadata is ambiguous or stale."""


@dataclass(frozen=True)
class DepthLevel:
    """One normalized CLOB price level."""

    price: float
    available_shares: float

    def __post_init__(self) -> None:
        price = float(self.price)
        shares = float(self.available_shares)
        if not math.isfinite(price) or not 0.0 < price < 1.0:
            raise SnapshotValidationError(f"invalid CLOB price: {self.price!r}")
        if not math.isfinite(shares) or shares <= 0.0:
            raise SnapshotValidationError(
                f"invalid CLOB available_shares: {self.available_shares!r}"
            )
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "available_shares", shares)

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.price,
            "available_shares": self.available_shares,
        }


@dataclass(frozen=True)
class CLOBExecutionSnapshot:
    """One decision-time snapshot for one outcome token.

    YES and NO snapshots are separate objects.  ``book_timestamp`` is always
    sourced from the CLOB response; the builder never manufactures it from
    fetch time or decision time.
    """

    market_id: str
    condition_id: str
    bucket: str
    token_side: str
    token_id: str
    decision_timestamp: datetime
    book_timestamp: datetime
    book_age_seconds: float
    tick_size: float
    minimum_order_size: float
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]
    fetch_cycle_id: str
    schema_version: str = SCHEMA_VERSION
    source_name: str = SOURCE_NAME

    def __post_init__(self) -> None:
        side = str(self.token_side).upper()
        if side not in {"YES", "NO"}:
            raise SnapshotValidationError(f"invalid token_side: {self.token_side!r}")
        if not self.market_id or not self.condition_id or not self.bucket:
            raise SnapshotValidationError("market_id, condition_id and bucket are required")
        if not self.token_id:
            raise SnapshotValidationError("token_id is required")
        if not isinstance(self.fetch_cycle_id, str) or not self.fetch_cycle_id:
            raise SnapshotValidationError("fetch_cycle_id is required")
        if self.schema_version != SCHEMA_VERSION:
            raise SnapshotValidationError(
                f"unsupported execution snapshot schema: {self.schema_version!r}"
            )
        if self.source_name != SOURCE_NAME:
            raise SnapshotValidationError(
                f"unsupported execution source: {self.source_name!r}"
            )
        if not isinstance(self.decision_timestamp, datetime):
            raise SnapshotValidationError("decision_timestamp is required")
        if not isinstance(self.book_timestamp, datetime):
            raise SnapshotValidationError("book_timestamp is required")
        age = float(self.book_age_seconds)
        if not math.isfinite(age) or age < 0.0:
            raise SnapshotValidationError(f"invalid book age: {self.book_age_seconds!r}")
        decision_utc = _as_utc(self.decision_timestamp, "decision_timestamp")
        book_utc = _as_utc(self.book_timestamp, "book_timestamp")
        computed_age = (decision_utc - book_utc).total_seconds()
        if computed_age < 0.0:
            raise SnapshotValidationError("CLOB book timestamp is in the future")
        if abs(computed_age - age) > 1e-3:
            raise SnapshotValidationError(
                f"book age does not match decision/book timestamps: "
                f"declared={age:.6f} computed={computed_age:.6f}"
            )
        tick = float(self.tick_size)
        min_size = float(self.minimum_order_size)
        if not math.isfinite(tick) or tick <= 0.0:
            raise SnapshotValidationError(f"invalid tick_size: {self.tick_size!r}")
        if not math.isfinite(min_size) or min_size <= 0.0:
            raise SnapshotValidationError(
                f"invalid minimum_order_size: {self.minimum_order_size!r}"
            )
        bids = tuple(self.bids)
        asks = tuple(self.asks)
        if any(not isinstance(level, DepthLevel) for level in (*bids, *asks)):
            raise SnapshotValidationError("all book levels must be DepthLevel objects")
        if tuple(level.price for level in bids) != tuple(
            sorted((level.price for level in bids), reverse=True)
        ):
            raise SnapshotValidationError("bid levels must be sorted high-to-low")
        if tuple(level.price for level in asks) != tuple(
            sorted(level.price for level in asks)
        ):
            raise SnapshotValidationError("ask levels must be sorted low-to-high")
        object.__setattr__(self, "token_side", side)
        object.__setattr__(self, "book_age_seconds", age)
        object.__setattr__(self, "tick_size", tick)
        object.__setattr__(self, "minimum_order_size", min_size)
        object.__setattr__(self, "bids", bids)
        object.__setattr__(self, "asks", asks)

    @property
    def best_bid(self) -> DepthLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> DepthLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def midpoint(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid.price + self.best_ask.price) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "condition_id": self.condition_id,
            "bucket": self.bucket,
            "token_side": self.token_side,
            "token_id": self.token_id,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "book_timestamp": self.book_timestamp.isoformat(),
            "book_age_seconds": self.book_age_seconds,
            "tick_size": self.tick_size,
            "minimum_order_size": self.minimum_order_size,
            "bids": [level.to_dict() for level in self.bids],
            "asks": [level.to_dict() for level in self.asks],
            "fetch_cycle_id": self.fetch_cycle_id,
            "schema_version": self.schema_version,
            "source_name": self.source_name,
        }


@dataclass(frozen=True)
class FillLevel:
    price: float
    filled_shares: float
    gross_notional: float
    fee: float
    net_cash_effect: float

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.price,
            "filled_shares": self.filled_shares,
            "gross_notional": self.gross_notional,
            "fee": self.fee,
            "net_cash_effect": self.net_cash_effect,
        }


@dataclass(frozen=True)
class DepthExecutionResult:
    """Detailed result of walking one side of one token book."""

    side: str
    requested_shares: float
    filled_shares: float
    fill_ratio: float
    gross_vwap: float | None
    all_in_buy_vwap: float | None
    net_sell_vwap: float | None
    total_fee: float
    gross_notional: float
    net_cash_flow: float
    worst_fill_price: float | None
    depth_levels_consumed: int
    unfilled_shares: float
    fills: tuple[FillLevel, ...] = field(default_factory=tuple)

    @property
    def is_full_fill(self) -> bool:
        return self.unfilled_shares <= EPSILON

    @property
    def execution_vwap(self) -> float | None:
        if self.side == "BUY":
            return self.all_in_buy_vwap
        return self.net_sell_vwap

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "requested_shares": self.requested_shares,
            "filled_shares": self.filled_shares,
            "fill_ratio": self.fill_ratio,
            "gross_vwap": self.gross_vwap,
            "all_in_buy_vwap": self.all_in_buy_vwap,
            "net_sell_vwap": self.net_sell_vwap,
            "total_fee": self.total_fee,
            "gross_notional": self.gross_notional,
            "net_cash_flow": self.net_cash_flow,
            "worst_fill_price": self.worst_fill_price,
            "depth_levels_consumed": self.depth_levels_consumed,
            "unfilled_shares": self.unfilled_shares,
            "fills": [fill.to_dict() for fill in self.fills],
        }


@dataclass(frozen=True)
class DepthSizingResult:
    """Iterative Kelly sizing result for one decision snapshot."""

    adjusted_bets: dict[str, dict[str, Any]]
    executable_prices: dict[str, float]
    diagnostic_edges: dict[str, float | None]
    rejected: dict[str, str]
    iterations: int
    converged: bool
    partial_fill_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjusted_bets": self.adjusted_bets,
            "executable_prices": self.executable_prices,
            "diagnostic_edges": self.diagnostic_edges,
            "rejected": self.rejected,
            "iterations": self.iterations,
            "converged": self.converged,
            "partial_fill_policy": self.partial_fill_policy,
        }


def _parse_timestamp(value: Any) -> datetime:
    if value is None or value == "":
        raise SnapshotValidationError("CLOB book timestamp is missing")
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise SnapshotValidationError(f"invalid CLOB timestamp: {value!r}")
        # Polymarket book timestamps are normally epoch milliseconds.
        if numeric > 100_000_000_000:
            numeric /= 1000.0
        parsed = datetime.fromtimestamp(numeric, tz=timezone.utc)
    else:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise SnapshotValidationError(
                f"invalid CLOB timestamp: {value!r}"
            ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise SnapshotValidationError(f"{field_name} is required")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_number(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotValidationError(f"{field_name} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise SnapshotValidationError(f"{field_name} is not finite: {value!r}")
    return number


def _parse_outcomes(raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SnapshotValidationError("market outcomes are not valid JSON") from exc
    if not isinstance(raw, (list, tuple)):
        raise SnapshotValidationError("market outcomes are missing")
    return [str(value).strip().lower() for value in raw]


def _event_date_and_kind(event_slug: str) -> tuple[date, str]:
    if not isinstance(event_slug, str) or "temperature-in-hong-kong" not in event_slug.lower():
        raise SnapshotValidationError("event location/schema is not Hong Kong temperature")
    match = re.search(
        r"^(highest|lowest)-temperature-in-hong-kong-on-"
        r"([a-z]+)-(\d{1,2})-(\d{4})$",
        event_slug.strip().lower(),
    )
    if not match:
        raise SnapshotValidationError(f"cannot validate temperature event slug: {event_slug!r}")
    month, day_number, year = match.group(2), int(match.group(3)), int(match.group(4))
    try:
        event_date = datetime.strptime(
            f"{month} {day_number} {year}", "%B %d %Y"
        ).date()
    except ValueError as exc:
        raise SnapshotValidationError(f"invalid event date in slug: {event_slug!r}") from exc
    return event_date, match.group(1)


def _book_levels(book: Mapping[str, Any], key: str) -> tuple[DepthLevel, ...]:
    levels = book.get(key)
    if levels is None:
        levels = book.get("top_bids" if key == "bids" else "top_asks", [])
    if not isinstance(levels, (list, tuple)):
        raise SnapshotValidationError(f"{key} levels are not a list")
    parsed: list[DepthLevel] = []
    for level in levels:
        if not isinstance(level, Mapping):
            raise SnapshotValidationError(f"invalid {key} level: {level!r}")
        price = level.get("price")
        shares = level.get("available_shares", level.get("size"))
        parsed.append(
            DepthLevel(
                _parse_number(price, f"{key}.price"),
                _parse_number(shares, f"{key}.available_shares"),
            )
        )
    return tuple(parsed)


def build_execution_snapshots(
    markets: list[Mapping[str, Any]],
    target_probs: Mapping[str, float],
    market_depth: Mapping[str, Mapping[str, Any] | None],
    market_depth_no: Mapping[str, Mapping[str, Any] | None],
    event_slug: str,
    decision_timestamp: datetime,
    expected_market_date: date,
    fetch_cycle_id: str | None = None,
    max_book_age_seconds: float = 60.0,
    is_min_temp: bool = False,
) -> dict[str, dict[str, CLOBExecutionSnapshot]]:
    """Validate market/token/depth consistency and build YES/NO snapshots."""
    event_date, event_kind = _event_date_and_kind(event_slug)
    if event_date != expected_market_date:
        raise SnapshotValidationError(
            f"market date mismatch: event={event_date} expected={expected_market_date}"
        )
    expected_kind = "lowest" if is_min_temp else "highest"
    if event_kind != expected_kind:
        raise SnapshotValidationError(
            f"market kind mismatch: event={event_kind} expected={expected_kind}"
        )
    if not markets:
        raise SnapshotValidationError("market list is empty")
    market_buckets = [str(m.get("bucket", "")) for m in markets]
    if any(not bucket for bucket in market_buckets) or len(set(market_buckets)) != len(market_buckets):
        raise SnapshotValidationError("market bucket keys are missing or duplicated")
    if set(market_buckets) != set(target_probs):
        raise SnapshotValidationError(
            f"bucket schema mismatch: market={sorted(market_buckets)} "
            f"model={sorted(target_probs)}"
        )
    if set(market_depth) != set(market_buckets) or set(market_depth_no) != set(market_buckets):
        raise SnapshotValidationError("complete YES/NO depth snapshot is required")
    if fetch_cycle_id is None:
        raise SnapshotValidationError("depth fetch cycle id is missing")
    decision = _as_utc(decision_timestamp, "decision_timestamp")
    if max_book_age_seconds <= 0:
        raise SnapshotValidationError("max_book_age_seconds must be positive")

    seen_tokens: dict[str, tuple[str, str]] = {}
    seen_conditions: dict[str, str] = {}
    seen_market_ids: dict[str, str] = {}
    snapshots: dict[str, dict[str, CLOBExecutionSnapshot]] = {}
    observed_cycles: set[str] = set()
    for market in markets:
        bucket = str(market["bucket"])
        condition_id = str(market.get("conditionId") or market.get("condition_id") or "")
        market_id = str(
            market.get("market_id") or market.get("id") or market.get("slug") or condition_id
        )
        outcomes = _parse_outcomes(market.get("outcomes"))
        if outcomes != ["yes", "no"]:
            raise SnapshotValidationError(
                f"token outcome mapping is not explicit YES/NO for {bucket}: {outcomes!r}"
            )
        if not condition_id or not market_id:
            raise SnapshotValidationError(f"missing market identity for bucket {bucket}")
        previous_condition = seen_conditions.get(condition_id)
        if previous_condition is not None and previous_condition != bucket:
            raise SnapshotValidationError(
                f"duplicate condition id {condition_id!r}: "
                f"{previous_condition!r} and {bucket!r}"
            )
        previous_market = seen_market_ids.get(market_id)
        if previous_market is not None and previous_market != bucket:
            raise SnapshotValidationError(
                f"duplicate market id {market_id!r}: "
                f"{previous_market!r} and {bucket!r}"
            )
        seen_conditions[condition_id] = bucket
        seen_market_ids[market_id] = bucket
        yes_token = str(market.get("token_id") or "")
        no_token = str(market.get("no_token_id") or "")
        if not yes_token or not no_token or yes_token == no_token:
            raise SnapshotValidationError(f"invalid YES/NO token mapping for {bucket}")
        for token_id, side in ((yes_token, "YES"), (no_token, "NO")):
            previous = seen_tokens.get(token_id)
            if previous is not None:
                raise SnapshotValidationError(
                    f"duplicate token id {token_id!r}: {previous} and {(bucket, side)}"
                )
            seen_tokens[token_id] = (bucket, side)

        snapshots[bucket] = {}
        for token_side, token_id, depth in (
            ("YES", yes_token, market_depth.get(bucket)),
            ("NO", no_token, market_depth_no.get(bucket)),
        ):
            if not isinstance(depth, Mapping):
                raise SnapshotValidationError(
                    f"missing {token_side} depth for bucket {bucket}"
                )
            asset_id = str(depth.get("asset_id") or depth.get("token_id") or "")
            if not asset_id:
                raise SnapshotValidationError(
                    f"{token_side} token identity is missing from depth for {bucket}"
                )
            if asset_id != token_id:
                raise SnapshotValidationError(
                    f"{token_side} token identity mismatch for {bucket}: "
                    f"market={token_id!r} book={asset_id!r}"
                )
            errors = depth.get("validation_errors") or []
            if errors:
                raise SnapshotValidationError(
                    f"invalid {token_side} depth for {bucket}: {errors!r}"
                )
            depth_cycle_raw = depth.get("fetch_cycle_id")
            if depth_cycle_raw in (None, ""):
                raise SnapshotValidationError(
                    f"{token_side} depth fetch cycle is missing for {bucket}"
                )
            depth_cycle = str(depth_cycle_raw)
            if depth_cycle != fetch_cycle_id:
                raise SnapshotValidationError("YES/NO books are from different fetch cycles")
            observed_cycles.add(depth_cycle)
            book_timestamp = _parse_timestamp(depth.get("timestamp"))
            age = (decision - book_timestamp).total_seconds()
            if age < 0.0 or age > max_book_age_seconds:
                raise SnapshotValidationError(
                    f"stale/future {token_side} book for {bucket}: age={age:.3f}s"
                )
            tick_raw = depth.get("tick_size")
            if tick_raw is None:
                tick_raw = market.get("tick_size", market.get("orderPriceMinTickSize"))
            min_raw = depth.get("minimum_order_size", depth.get("min_order_size"))
            if min_raw is None:
                min_raw = market.get(
                    "minimum_order_size", market.get("orderMinSize", market.get("minimumOrderSize"))
                )
            if tick_raw is None or min_raw is None:
                raise SnapshotValidationError(
                    f"tick_size/minimum_order_size missing for {bucket}/{token_side}"
                )
            snapshots[bucket][token_side] = CLOBExecutionSnapshot(
                market_id=market_id,
                condition_id=condition_id,
                bucket=bucket,
                token_side=token_side,
                token_id=token_id,
                decision_timestamp=decision,
                book_timestamp=book_timestamp,
                book_age_seconds=age,
                tick_size=_parse_number(tick_raw, "tick_size"),
                minimum_order_size=_parse_number(min_raw, "minimum_order_size"),
                bids=_book_levels(depth, "bids"),
                asks=_book_levels(depth, "asks"),
                fetch_cycle_id=depth_cycle,
                source_name=str(depth.get("source_name") or ""),
            )
    if observed_cycles != {fetch_cycle_id}:
        raise SnapshotValidationError("depth fetch cycle is not coherent")
    return snapshots


def _empty_fill(side: str, requested_shares: float) -> DepthExecutionResult:
    return DepthExecutionResult(
        side=side,
        requested_shares=requested_shares,
        filled_shares=0.0,
        fill_ratio=0.0,
        gross_vwap=None,
        all_in_buy_vwap=None,
        net_sell_vwap=None,
        total_fee=0.0,
        gross_notional=0.0,
        net_cash_flow=0.0,
        worst_fill_price=None,
        depth_levels_consumed=0,
        unfilled_shares=requested_shares,
        fills=(),
    )


def walk_depth(
    snapshot: CLOBExecutionSnapshot,
    side: str,
    requested_shares: float,
    fee_rate: float = FEE_RATE,
) -> DepthExecutionResult:
    """Walk asks for BUY or bids for SELL, charging fee at every level."""
    normalized_side = str(side).upper()
    if normalized_side not in {"BUY", "SELL"}:
        raise CLOBExecutionError(f"unsupported order side: {side!r}")
    requested = float(requested_shares)
    if not math.isfinite(requested) or requested <= 0.0:
        return _empty_fill(normalized_side, max(requested, 0.0))
    rate = float(fee_rate)
    if not math.isfinite(rate) or rate < 0.0:
        raise CLOBExecutionError(f"invalid fee rate: {fee_rate!r}")
    levels = snapshot.asks if normalized_side == "BUY" else snapshot.bids
    if not levels:
        return _empty_fill(normalized_side, requested)

    remaining = requested
    fills: list[FillLevel] = []
    gross_notional = 0.0
    total_fee = 0.0
    filled_shares = 0.0
    for level in levels:
        if remaining <= EPSILON:
            break
        filled = min(remaining, level.available_shares)
        gross = filled * level.price
        fee = filled * rate * level.price * (1.0 - level.price)
        net_effect = -(gross + fee) if normalized_side == "BUY" else gross - fee
        fills.append(FillLevel(level.price, filled, gross, fee, net_effect))
        filled_shares += filled
        gross_notional += gross
        total_fee += fee
        remaining -= filled

    if filled_shares <= EPSILON:
        return _empty_fill(normalized_side, requested)
    gross_vwap = gross_notional / filled_shares
    fill_ratio = min(1.0, filled_shares / requested)
    return DepthExecutionResult(
        side=normalized_side,
        requested_shares=requested,
        filled_shares=filled_shares,
        fill_ratio=fill_ratio,
        gross_vwap=gross_vwap,
        all_in_buy_vwap=(gross_notional + total_fee) / filled_shares
        if normalized_side == "BUY"
        else None,
        net_sell_vwap=(gross_notional - total_fee) / filled_shares
        if normalized_side == "SELL"
        else None,
        total_fee=total_fee,
        gross_notional=gross_notional,
        net_cash_flow=-(gross_notional + total_fee)
        if normalized_side == "BUY"
        else gross_notional - total_fee,
        worst_fill_price=(max(fill.price for fill in fills) if normalized_side == "BUY"
                          else min(fill.price for fill in fills)),
        depth_levels_consumed=len(fills),
        unfilled_shares=max(0.0, remaining),
        fills=tuple(fills),
    )


def _best_ask(snapshot: CLOBExecutionSnapshot | None) -> float | None:
    return snapshot.best_ask.price if snapshot and snapshot.best_ask else None


def _snapshot_side(
    snapshots: Mapping[str, Mapping[str, CLOBExecutionSnapshot]],
    bucket: str,
    side: str,
) -> CLOBExecutionSnapshot | None:
    bucket_snapshots = snapshots.get(bucket) or {}
    return bucket_snapshots.get(side)


def _top_price_map(
    target_probs: Mapping[str, float],
    snapshots: Mapping[str, Mapping[str, CLOBExecutionSnapshot]],
) -> tuple[dict[str, float | None], dict[str, dict[str, bool]]]:
    prices: dict[str, float | None] = {}
    available: dict[str, dict[str, bool]] = {}
    for bucket in target_probs:
        yes = _best_ask(_snapshot_side(snapshots, bucket, "YES"))
        no = _best_ask(_snapshot_side(snapshots, bucket, "NO"))
        prices[bucket] = yes if yes is not None else (1.0 - no if no is not None else None)
        available[bucket] = {"YES": yes is not None, "NO": no is not None}
    return prices, available


def _diagnostic_edge(
    model_probability: float,
    gamma_yes_price: Any,
    token_side: str,
) -> float | None:
    if gamma_yes_price is None:
        return None
    try:
        gamma = float(gamma_yes_price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(gamma):
        return None
    return float(model_probability) - gamma if token_side == "YES" else gamma - float(model_probability)


def _bet_signature(bets: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[str, float]]:
    return {
        bucket: (str(bet.get("action")), float(bet.get("amount", 0.0)))
        for bucket, bet in bets.items()
    }


def _signatures_converged(
    previous: Mapping[str, tuple[str, float]] | None,
    current: Mapping[str, tuple[str, float]],
) -> bool:
    if previous is None or set(previous) != set(current):
        return False
    for bucket, (action, amount) in current.items():
        old_action, old_amount = previous[bucket]
        if action != old_action:
            return False
        if abs(amount - old_amount) >= 1.0:
            return False
        if old_amount > EPSILON and abs(amount - old_amount) / old_amount >= 0.01:
            return False
    return True


def compute_depth_adjusted_bets(
    target_probs: Mapping[str, float],
    gamma_reference_prices: Mapping[str, float] | None,
    capital: float,
    snapshots: Mapping[str, Mapping[str, CLOBExecutionSnapshot]],
    max_per_bucket: float,
    total_max: float,
    partial_fill_policy: str = "fail_closed",
    max_iterations: int = 6,
) -> DepthSizingResult:
    """Iteratively size Kelly bets using depth-adjusted all-in prices.

    The only price map passed to Kelly is derived from the CLOB top ask or the
    complement of the NO-token top ask.  ``gamma_reference_prices`` is used
    solely for the returned diagnostic edge.
    """
    if partial_fill_policy not in {"fail_closed", "accept_partial", "reduce_to_available"}:
        raise CLOBExecutionError(f"invalid partial fill policy: {partial_fill_policy!r}")
    try:
        from execution.kelly_betting import compute_multi_kelly_bets
    except ImportError as exc:  # pragma: no cover - import failure is environment-specific
        raise CLOBExecutionError("Kelly sizing dependency is unavailable") from exc

    top_prices, side_available = _top_price_map(target_probs, snapshots)
    prices: dict[str, float | None] = dict(top_prices)
    previous_signature: dict[str, tuple[str, float]] | None = None
    last_adjusted: dict[str, dict[str, Any]] = {}
    last_executable_prices: dict[str, float] = {}
    rejected: dict[str, str] = {}
    iterations = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        bets = compute_multi_kelly_bets(
            dict(target_probs),
            prices,
            capital,
            max_per_bucket=max_per_bucket,
            total_max=total_max,
            executable_sides=side_available,
        )
        if not bets:
            if previous_signature is None:
                converged = True
            last_adjusted = {}
            break

        adjusted: dict[str, dict[str, Any]] = {}
        next_prices = dict(prices)
        iteration_rejected: dict[str, str] = {}
        executable_prices: dict[str, float] = {}
        for bucket, bet in bets.items():
            token_side = "YES" if bet.get("action") == "BUY_YES" else "NO"
            snapshot = _snapshot_side(snapshots, bucket, token_side)
            top = _best_ask(snapshot)
            if snapshot is None or top is None:
                iteration_rejected[bucket] = "no_valid_executable_quote"
                continue
            # ``amount`` is the Kelly cash allocation.  After the first
            # iteration, size shares from the previous all-in executable
            # price rather than the top quote; otherwise depth slippage can
            # silently increase the actual cash exposure above the Kelly
            # allocation.
            sizing_yes_price = prices.get(bucket)
            sizing_token_price = (
                sizing_yes_price
                if token_side == "YES"
                else (1.0 - sizing_yes_price if sizing_yes_price is not None else None)
            )
            if sizing_token_price is None or sizing_token_price <= 0.0:
                iteration_rejected[bucket] = "no_valid_executable_quote"
                continue
            requested_shares = float(bet.get("amount", 0.0)) / sizing_token_price
            fill = walk_depth(snapshot, "BUY", requested_shares)
            if fill.filled_shares <= EPSILON:
                iteration_rejected[bucket] = "no_liquidity"
                continue
            if not fill.is_full_fill and partial_fill_policy == "fail_closed":
                iteration_rejected[bucket] = "partial_fill_fail_closed"
                continue
            execution_price = fill.all_in_buy_vwap
            if execution_price is None:
                iteration_rejected[bucket] = "no_valid_executable_quote"
                continue
            execution_yes_price = execution_price if token_side == "YES" else 1.0 - execution_price
            next_prices[bucket] = execution_yes_price
            executable_prices[bucket] = execution_yes_price
            diagnostic = _diagnostic_edge(
                target_probs[bucket],
                (gamma_reference_prices or {}).get(bucket),
                token_side,
            )
            executable_edge = (
                target_probs[bucket] - execution_yes_price
                if token_side == "YES"
                else execution_yes_price - target_probs[bucket]
            )
            adjusted[bucket] = {
                **bet,
                "adjusted_quantity": fill.filled_shares,
                "requested_shares": fill.requested_shares,
                "actual_cost": fill.gross_notional,
                "total_cash_outflow": -fill.net_cash_flow,
                "avg_fill_price": execution_price,
                "execution_price": execution_price,
                "execution_yes_price": execution_yes_price,
                "best_ask": top,
                "sizing_token_price": sizing_token_price,
                "slippage_pct": ((execution_price - top) / top * 100.0) if top else 0.0,
                "depth_slippage_pct": ((execution_price - top) / top * 100.0) if top else 0.0,
                "filled": fill.is_full_fill,
                "is_partial": not fill.is_full_fill,
                "fill_ratio": fill.fill_ratio,
                "unfilled_shares": fill.unfilled_shares,
                "fee": fill.total_fee,
                "gross_notional": fill.gross_notional,
                "net_cash_flow": fill.net_cash_flow,
                "worst_fill_price": fill.worst_fill_price,
                "depth_levels_consumed": fill.depth_levels_consumed,
                "depth_fill": fill.to_dict(),
                "diagnostic_edge": diagnostic,
                "executable_edge_at_final_size": executable_edge,
                "execution_price_is_all_in": True,
                "partial_fill_policy": partial_fill_policy,
            }

        current_signature = _bet_signature(bets)
        if _signatures_converged(previous_signature, current_signature):
            converged = True
            last_adjusted = adjusted
            last_executable_prices = executable_prices
            rejected = iteration_rejected
            break
        previous_signature = current_signature
        prices = next_prices
        last_adjusted = adjusted
        last_executable_prices = executable_prices
        rejected = iteration_rejected

    if not converged:
        # A non-converged fixed point is not safe to trade.  Do not return the
        # last optimistic quote as a fallback.
        last_adjusted = {}
        last_executable_prices = {}
        rejected = {"__global__": "kelly_depth_sizing_did_not_converge", **rejected}

    diagnostic_edges = {
        bucket: bet.get("diagnostic_edge")
        for bucket, bet in last_adjusted.items()
    }
    return DepthSizingResult(
        adjusted_bets=last_adjusted,
        executable_prices=last_executable_prices,
        diagnostic_edges=diagnostic_edges,
        rejected=rejected,
        iterations=iterations,
        converged=converged,
        partial_fill_policy=partial_fill_policy,
    )


def compute_sell_execution(
    snapshot: CLOBExecutionSnapshot,
    requested_shares: float,
    partial_fill_policy: str = "fail_closed",
) -> DepthExecutionResult:
    """Walk bids for an exit and apply the configured partial-fill policy."""
    if partial_fill_policy not in {"fail_closed", "accept_partial", "reduce_to_available"}:
        raise CLOBExecutionError(f"invalid partial fill policy: {partial_fill_policy!r}")
    result = walk_depth(snapshot, "SELL", requested_shares)
    if not result.filled_shares:
        return result
    if not result.is_full_fill and partial_fill_policy == "fail_closed":
        return _empty_fill("SELL", float(requested_shares))
    return result


def mark_to_market(
    snapshot: CLOBExecutionSnapshot,
    shares: float,
) -> dict[str, Any]:
    """Expose midpoint and immediate bid-depth liquidation diagnostics."""
    quantity = max(0.0, float(shares))
    midpoint_value = snapshot.midpoint * quantity if snapshot.midpoint is not None else None
    liquidation = walk_depth(snapshot, "SELL", quantity) if quantity > 0 else _empty_fill("SELL", 0.0)
    return {
        "token_side": snapshot.token_side,
        "shares": quantity,
        "midpoint_mark": midpoint_value,
        "immediate_liquidation_value": liquidation.net_cash_flow,
        "immediate_liquidation_fill_ratio": liquidation.fill_ratio,
        "immediate_liquidation_fee": liquidation.total_fee,
        "immediate_liquidation_vwap": liquidation.net_sell_vwap,
        "book_age_seconds": snapshot.book_age_seconds,
        "book_timestamp": snapshot.book_timestamp.isoformat(),
    }
