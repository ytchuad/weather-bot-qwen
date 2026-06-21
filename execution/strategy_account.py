# execution/strategy_account.py
"""StrategyAccount — per-strategy independent account with persistence.

Replaces the old portfolio-centric model.  Each strategy account is
self-contained:
  - its own capital allocation
  - its own model (baseline, rain_nowcast, model_a, …)
  - its own per-strategy params (bias, std_mult, kelly_fraction)
  - a reference to its gate definition in paper_strategies.json
  - independent ON/OFF toggle for live paper trading
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────

ACCOUNTS_PATH = Path("data/strategy_accounts.json")

# Templates that can be resolved to a Polymarket event slug.
# "hk-tmax" → "highest-temperature-in-hong-kong-on-{month}-{day}-{year}"
VALID_MARKET_TEMPLATES = ("hk-tmax", "hk-tmin")

VALID_STATUSES = ("running", "paused", "stopped")

DEFAULT_PARAMS: dict[str, Any] = dict(
    bias=0.0,
    std_mult=1.0,
    kelly_fraction=0.25,
)

DEFAULT_DOC: dict[str, Any] = dict(
    version=1,
    strategies={},
)


# ── Dataclass ─────────────────────────────────────────────────────────

@dataclass
class StrategyAccount:
    """One independently-operating strategy.

    Attributes
    ----------
    id : str
        Unique strategy key (e.g. ``"enhanced_v2_aggressive"``).
    label : str
        Human-readable display name.
    model : str
        Model key to use (e.g. ``"baseline"``, ``"rain_nowcast"``, ``"model_a"``).
    capital : float
        Virtual capital allocated to this strategy.
    initial_capital : float
        Starting capital for ROI tracking.
    market_template : str
        Market template (e.g. ``"hk-tmax"``) resolved to a daily Polymarket
        event slug at runtime.
    status : str
        One of ``"running"`` | ``"paused"`` | ``"stopped"``.
    scheduler_on : bool
        Whether the auto-scheduler should run this strategy.
    last_run : str | None
        ISO-format timestamp of the last execution cycle.
    params : dict
        Per-strategy knobs: ``bias``, ``std_mult``, ``kelly_fraction``.
    from_strategy_key : str | None
        Base strategy key in ``config/paper_strategies.json`` whose gate
        definition this account uses (e.g. ``"enhanced_v2_paper"``).
    gate_config_override : dict | None
        Optional extra overrides merged on top of the base gate config.
    """

    id: str
    label: str = ""
    model: str = "baseline"
    capital: float = 10_000.0
    initial_capital: float = 10_000.0
    market_template: str = "hk-tmax"
    status: str = "paused"
    scheduler_on: bool = False
    last_run: str | None = None
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    from_strategy_key: str | None = None
    gate_config_override: dict | None = None

    def __post_init__(self):
        # fill missing default params
        merged = dict(DEFAULT_PARAMS)
        merged.update(self.params or {})
        self.params = merged

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure initial_capital is included
        if "initial_capital" not in d:
            d["initial_capital"] = self.initial_capital
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StrategyAccount:
        # Backward compatibility: if initial_capital missing, use capital
        if "initial_capital" not in d and "capital" in d:
            d["initial_capital"] = d["capital"]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Store ─────────────────────────────────────────────────────────────

class StrategyAccountStore:
    """Read/write ``data/strategy_accounts.json`` and provide CRUD."""

    def __init__(self, path: Path = ACCOUNTS_PATH):
        self.path = path

    # ── read ───────────────────────────────────────────────────────────

    def load_all(self) -> dict[str, dict]:
        """Return raw ``{id: {...}}`` dict from disk."""
        raw = self._read()
        return raw.get("strategies", {})

    def load(self, sid: str) -> StrategyAccount | None:
        raw = self.load_all().get(sid)
        if raw is None:
            return None
        return StrategyAccount.from_dict(raw)

    def list(self) -> list[StrategyAccount]:
        return [StrategyAccount.from_dict(d) for d in self.load_all().values()]

    # ── write ──────────────────────────────────────────────────────────

    def save(self, acct: StrategyAccount) -> None:
        raw = self._read()
        raw.setdefault("strategies", {})[acct.id] = self._serialise(acct)
        self._write(raw)

    def delete(self, sid: str) -> bool:
        raw = self._read()
        removed = raw.get("strategies", {}).pop(sid, None)
        if removed is not None:
            self._write(raw)
            return True
        return False

    def set_status(self, sid: str, status: str) -> bool:
        raw = self._read()
        sd = raw.get("strategies", {}).get(sid)
        if sd is None:
            return False
        sd["status"] = status
        sd["scheduler_on"] = status == "running"
        self._write(raw)
        return True

    def set_last_run(self, sid: str, dt: datetime | None = None) -> None:
        raw = self._read()
        sd = raw.get("strategies", {}).get(sid)
        if sd is None:
            return
        sd["last_run"] = (dt or datetime.now(timezone.utc)).isoformat()
        self._write(raw)

    def get_running(self) -> list[StrategyAccount]:
        """Return all strategies with ``scheduler_on == True``."""
        return [a for a in self.list() if a.scheduler_on and a.status == "running"]

    # ── serialisation helpers ──────────────────────────────────────────

    @staticmethod
    def _serialise(acct: StrategyAccount) -> dict:
        d = asdict(acct)
        d["params"] = dict(DEFAULT_PARAMS)
        d["params"].update(acct.params or {})
        return d

    def _read(self) -> dict:
        if not self.path.exists():
            return dict(DEFAULT_DOC)
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Can't read %s: %s — starting fresh", self.path, exc)
            return dict(DEFAULT_DOC)

    def _write(self, raw: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2, default=str)


# ── Migration helper ──────────────────────────────────────────────────

def migrate_from_portfolios(
    portfolios_path: Path = Path("config/portfolios.json"),
    accounts_path: Path = ACCOUNTS_PATH,
) -> int:
    """Read old ``config/portfolios.json`` and create per-strategy accounts.

    If ``data/strategy_accounts.json`` already has entries, does nothing.
    Returns the number of accounts created (0 if skipped).
    """
    store = StrategyAccountStore(accounts_path)
    if store.list():
        logger.info("strategy_accounts.json already has entries — skipping migration")
        return 0

    if not portfolios_path.exists():
        logger.info("No portfolios.json found — nothing to migrate")
        return 0

    try:
        with open(portfolios_path, encoding="utf-8") as f:
            pf_raw = json.load(f)
    except Exception as exc:
        logger.warning("Can't read %s: %s — skipping migration", portfolios_path, exc)
        return 0

    created = 0
    for pid, pdef in pf_raw.get("portfolios", {}).items():
        cap = float(pdef.get("capital", 10_000.0))
        strategies = pdef.get("strategies", [])
        strategy_models = pdef.get("strategy_models", {})
        cap_per = cap / max(len(strategies), 1)

        for sk in strategies:
            model = strategy_models.get(sk, "baseline")
            # Strip "_paper" suffix for model key if present
            if model.endswith("_paper"):
                model = model[:-6]
            acct = StrategyAccount(
                id=sk,
                label=pdef.get("label", sk),
                model=model,
                capital=cap_per,
                initial_capital=cap_per,
                market_template="hk-tmax",
                status="paused",
                scheduler_on=False,
                from_strategy_key=sk if sk in _all_v2_keys() else None,
            )
            store.save(acct)
            created += 1

    logger.info("Migrated %d strategy account(s) from portfolios.json", created)
    return created


def _all_v2_keys() -> set[str]:
    """Return known strategy keys from ``config/paper_strategies.json``."""
    try:
        from execution.strategy_factory import get_factory
        return set(get_factory().keys())
    except Exception:
        return set()


# ── Module-level convenience ──────────────────────────────────────────

_default_store = StrategyAccountStore()


def get_store() -> StrategyAccountStore:
    return _default_store


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manage strategy accounts")
    parser.add_argument("action", nargs="?", default="list",
                        choices=["list", "migrate", "count"])
    parser.add_argument("--id", help="Strategy ID")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.action == "migrate":
        n = migrate_from_portfolios()
        print(f"Created {n} account(s)")
    elif args.action == "count":
        store = get_store()
        print(f"Total accounts: {len(store.list())}")
    else:
        store = get_store()
        for a in store.list():
            print(f"  {a.id:40s}  model={a.model:15s}  capital={a.capital:>8.0f}  "
                  f"status={a.status:10s}  scheduler={a.scheduler_on}")