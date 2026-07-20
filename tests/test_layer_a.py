from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from layer_a.export import export_layer_a
from layer_a.replay import replay_layer_a
from layer_a.schema import LayerASchemaError, build_layer_a_record
from layer_a.storage import LayerAStore
from layer_a.upload import DatasetUploader


UTC = timezone.utc


def _book(token: str, decision: datetime, cycle: str = "fetch-1", offset: float = 0.0) -> dict:
    timestamp = decision - timedelta(seconds=2)
    return {
        "asset_id": token,
        "timestamp": timestamp.isoformat(),
        "source_name": "polymarket_clob",
        "fetch_cycle_id": cycle,
        "tick_size": 0.01,
        "minimum_order_size": 1.0,
        "bids": [
            {"price": 0.35 + offset, "size": 5.0},
            {"price": 0.30 + offset, "size": 10.0},
        ],
        "asks": [
            {"price": 0.40 + offset, "size": 4.0},
            {"price": 0.45 + offset, "size": 8.0},
        ],
        "validation_errors": [],
    }


def _context(
    *,
    event_date: str = "2026-07-20",
    decision: datetime | None = None,
    cycle: str = "fetch-1",
    complete: bool = True,
    include_account_state: bool = False,
) -> dict:
    decision = decision or datetime(2026, 7, 20, 9, 0, tzinfo=UTC)
    market = {
        "bucket": "30-31",
        "id": "market-1",
        "conditionId": "condition-1",
        "outcomes": ["Yes", "No"],
        "token_id": "yes-token-1",
        "no_token_id": "no-token-1",
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 1.0,
        "market_schema_version": "gamma-market.v1",
        "yes_price": 0.38,
    }
    context = {
        "decision_timestamp": decision,
        "event_date": event_date,
        "target_date_str": event_date,
        "event_slug": "highest-temperature-in-hong-kong-on-july-20-2026",
        "market_kind": "highest_temperature",
        "markets": [market],
        "market_depth": {"30-31": _book("yes-token-1", decision, cycle)},
        "market_depth_no": {"30-31": _book("no-token-1", decision, cycle, 0.02)},
        "weather_state": {
            "observations": {"temperature": 30.0, "humidity": 70.0},
            "max_so_far": 30.0,
            "min_so_far": 25.0,
            "lags": {"temp_60m_ago": 29.5},
            "trends": {"temp_change_60m": 0.5},
            "forecast": {"forecast_max": 31.0, "forecast_min": 25.0},
            "status": {
                "temp_current": {
                    "value": 30.0,
                    "source_timestamp": "2026-07-20T08:58:00+00:00",
                    "age_minutes": 2.0,
                    "is_missing": False,
                    "is_stale": False,
                    "is_fallback": False,
                    "fallback_method": None,
                    "source_name": "hko_rhrread",
                    "quality_flags": ["direct"],
                    "raw_status": "observed",
                },
            },
        },
        "model_states": {
            "model_a": {
                "model_name": "model_a",
                "model_version": "v2",
                "artifact_identity": "artifact-sha-1",
                "feature_spec": "config/model_a.yaml",
                "numeric_features": {"temp_now": 30.0, "wind_ref_mean": 5.0},
                "diagnostic_features": {"temp_now": 30.0, "wind_ref_mean": 5.0},
                "q10": 29.0,
                "q25": 29.5,
                "q50": 30.2,
                "q75": 30.8,
                "q90": 31.5,
                "point_prediction": 30.2,
                "full_bucket_probabilities": {"30-31": 0.80},
                "classifier_probability": 0.25,
                "model_input_status_summary": {"status_contract_version": "phase2a.v1"},
            },
        },
        "gamma_reference_prices": {"30-31": 0.38},
    }
    if not complete:
        context["market_depth_no"] = {}
    if include_account_state:
        context.update(
            {
                "account_id": "secret-account",
                "capital": 500.0,
                "paper_positions": {"30-31": 2},
                "target_orders": [{"bucket": "30-31"}],
                "simulated_fills": [{"shares": 1}],
                "realized_pnl": 12.0,
            }
        )
    return context


def _record(**kwargs):
    return build_layer_a_record(_context(**kwargs))


def test_complete_cycle_and_all_required_sections():
    record = _record()
    assert record["schema_version"] == "layer_a.v1"
    assert record["completeness"]["weather_complete"] is True
    assert record["completeness"]["model_state_complete"] is True
    assert record["completeness"]["replay_eligible_for_clob_strategy"] is True
    assert record["models"][0]["full_bucket_probabilities"] == {"30-31": 0.8}
    assert record["models"][0]["artifact_identity"] == "artifact-sha-1"
    assert {book["token_side"] for book in record["clob_books"]} == {"YES", "NO"}


def test_incomplete_cycle_is_retained_with_exact_reasons(tmp_path):
    record = _record(complete=False)
    assert record["completeness"]["depth_pair_complete"] is False
    assert "clob_books[30-31/NO]" in record["completeness"]["missing_fields"]
    result = LayerAStore(tmp_path).capture(record)
    assert result.status == "captured"
    stored = LayerAStore(tmp_path).read_partition_records(LayerAStore(tmp_path).scan()[0])[0]
    assert stored["decision_cycle_id"] == record["decision_cycle_id"]
    assert stored["completeness"]["replay_eligible_for_clob_strategy"] is False


def test_exact_status_and_full_books_are_preserved():
    record = _record()
    assert record["weather_state"]["status"]["temp_current"]["quality_flags"] == ["direct"]
    assert record["weather_state"]["status"]["temp_current"]["age_minutes"] == 2.0
    yes = next(book for book in record["clob_books"] if book["token_side"] == "YES")
    assert yes["asset_id"] == "yes-token-1"
    assert yes["fetch_cycle_id"] == "fetch-1"
    assert yes["bids"] == [
        {"price": 0.35, "available_shares": 5.0},
        {"price": 0.3, "available_shares": 10.0},
    ]
    assert yes["asks"] == [
        {"price": 0.4, "available_shares": 4.0},
        {"price": 0.45, "available_shares": 8.0},
    ]


def test_epoch_millisecond_string_book_timestamps_are_complete():
    context = _context()
    book_timestamp = int((context["decision_timestamp"] - timedelta(seconds=2)).timestamp() * 1000)
    context["market_depth"]["30-31"]["timestamp"] = str(book_timestamp)
    context["market_depth_no"]["30-31"]["timestamp"] = str(book_timestamp)
    record = build_layer_a_record(context)
    assert record["completeness"]["book_timestamp_complete"] is True
    assert record["completeness"]["replay_eligible_for_clob_strategy"] is True


def test_nested_strategy_state_is_rejected_if_it_reaches_layer_a():
    context = _context()
    context["weather_state"]["account_id"] = "must-not-persist"
    with pytest.raises(LayerASchemaError, match="account_id"):
        build_layer_a_record(context)


def test_fetch_cycle_coherence_is_validated():
    record = _record(cycle="fetch-yes")
    record["clob_books"][1]["fetch_cycle_id"] = "fetch-no"
    record["completeness"] = __import__("layer_a.schema", fromlist=["assess_completeness"]).assess_completeness(record)
    assert record["completeness"]["fetch_cycle_coherent"] is False
    assert "fetch_id_incoherent" not in record["completeness"]["missing_fields"]
    assert "fetch_cycle_id_incoherent" in record["completeness"]["missing_fields"]


def test_strategy_independent_record_rejects_no_account_state():
    record = _record(include_account_state=True)
    prohibited = {
        "account_id", "capital", "paper_positions", "target_orders",
        "simulated_fills", "realized_pnl", "unrealized_pnl",
    }
    assert prohibited.isdisjoint(record)
    assert prohibited.isdisjoint(record["weather_state"])
    assert all(
        prohibited.isdisjoint(model)
        for model in record["models"]
        if isinstance(model, dict)
    )


def test_atomic_partition_close_deduplicates_and_writes_checksums(tmp_path):
    store = LayerAStore(tmp_path)
    record = _record()
    first = store.capture(record)
    assert first.status == "captured"
    assert store.capture(record).status == "duplicate"
    assert not list(tmp_path.rglob("*.tmp"))
    info = store.scan()[0]
    assert info.status == "complete"
    assert info.checksum_valid is True
    assert info.manifest["cycle_count"] == 1
    assert info.manifest["file_checksums"]


def test_startup_scan_detects_interrupted_temporary_partition(tmp_path):
    source = LayerAStore(tmp_path / "source")
    source.capture(_record())
    source_info = source.scan()[0]
    target_root = tmp_path / "target"
    target_dir = target_root / "date=2026-07-20" / "hour=09"
    target_dir.mkdir(parents=True)
    shutil.copyfile(source_info.files["cycles"], target_dir / "cycles-crash.parquet.tmp")
    scanned = LayerAStore(target_root).scan()
    assert len(scanned) == 1
    assert scanned[0].status == "incomplete"
    assert "temporary_files_present" in scanned[0].reasons
    assert (target_dir / "cycles-crash.parquet.tmp").exists()


def test_manifest_checksum_detection(tmp_path):
    store = LayerAStore(tmp_path)
    store.capture(_record())
    info = store.scan()[0]
    info.files["books"].write_bytes(info.files["books"].read_bytes() + b"tamper")
    scanned = store.scan()[0]
    assert scanned.checksum_valid is False
    assert any(reason.startswith("checksum_mismatch") for reason in scanned.reasons)


def test_export_archive_and_date_filter(tmp_path):
    store = LayerAStore(tmp_path / "layer_a")
    store.capture(_record(event_date="2026-07-20"))
    store.capture(_record(event_date="2026-07-21", decision=datetime(2026, 7, 21, 9, tzinfo=UTC)))
    output = tmp_path / "export.zip"
    result = export_layer_a(store=store, output=output, date_value="2026-07-20", verify_checksums=True)
    assert result["cycle_count"] == 1
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        assert manifest["cycle_count"] == 1
        assert "checksums.sha256" in archive.namelist()
        assert "docs/layer_a_schema.md" in archive.namelist()
        assert len([name for name in archive.namelist() if name.endswith(".parquet")]) == 1


class _FakeHub:
    def __init__(self):
        self.paths = set()
        self.calls = []

    def file_exists(self, *, repo_id, filename, repo_type):
        return filename in self.paths

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.calls.append((path_or_fileobj, path_in_repo, repo_id, repo_type))
        self.paths.add(path_in_repo)
        return object()


def test_dataset_upload_receipt_and_no_overwrite(tmp_path):
    fake = _FakeHub()
    uploader = DatasetUploader("private/layer-a", "secret-token", api=fake, sleep_fn=lambda _seconds: None)
    store = LayerAStore(tmp_path, uploader=uploader, auto_upload=True, upload_interval_minutes=0)
    result = store.capture(_record())
    assert result.status == "captured"
    info = store.scan()[0]
    assert info.uploaded is True
    assert len(fake.calls) == 3
    store.retry_pending_uploads()
    assert len(fake.calls) == 3
    receipt = next((tmp_path / ".upload_receipts").glob("*.json"))
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_data["repo_type"] == "dataset"
    assert "secret-token" not in receipt.read_text(encoding="utf-8")


class _FailingHub:
    def upload_file(self, **_kwargs):
        raise RuntimeError("request contained secret-token")


def test_upload_failure_does_not_stop_local_capture_or_log_token(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HF_LAYER_A_TOKEN", "secret-token")
    uploader = DatasetUploader("private/layer-a", "secret-token", api=_FailingHub(), max_retries=2, sleep_fn=lambda _seconds: None)
    store = LayerAStore(tmp_path, uploader=uploader, auto_upload=True, upload_interval_minutes=0)
    with caplog.at_level("WARNING"):
        result = store.capture(_record())
    assert result.status == "captured"
    assert store.scan()[0].status == "complete"
    assert store.health_summary()["upload_failures"] == 1
    assert "secret-token" not in caplog.text
    failure_file = next((tmp_path / ".upload_failures").glob("*.json"))
    assert "secret-token" not in failure_file.read_text(encoding="utf-8")


def test_startup_health_reports_pending_and_incomplete(tmp_path):
    store = LayerAStore(tmp_path)
    store.capture(_record())
    summary = store.startup_scan()
    assert summary["complete"] == 1
    assert summary["incomplete"] == 0
    health = store.health_summary()
    assert health["pending_local_partitions"] == 1
    assert "local_disk_usage_bytes" in health
    assert "oldest_unuploaded_partition" in health


def test_replay_smoke_and_threshold_kelly_variants(tmp_path):
    store = LayerAStore(tmp_path / "layer_a")
    store.capture(_record())
    archive = tmp_path / "replay.zip"
    export_layer_a(store=store, output=archive)
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    low = replay_layer_a(archive, strategy_a_threshold=0.1, kelly_fraction=0.1)
    high = replay_layer_a(archive, strategy_a_threshold=0.5, kelly_fraction=0.9)
    assert low["model_probability_snapshots_reconstructed"] == 1
    assert low["clob_books_reconstructed"] == 2
    assert low["depth_walk"]["fee_and_vwap_recomputed"] is True
    assert low["strategy_a_probe"]["candidate_count"] >= high["strategy_a_probe"]["candidate_count"]
    if low["strategy_a_probe"]["candidates"]:
        assert low["strategy_a_probe"]["candidates"][0]["kelly_fraction"] == 0.1
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before


def test_both_context_entry_points_share_one_cycle_object(monkeypatch):
    from app.api import strategies as strategy_api
    from app.services import canonical_cycle
    from app.services import context_builder
    from execution.strategy_account import StrategyAccount

    cycle = SimpleNamespace(decision_cycle_id="cycle-1")
    seen = []

    def fake_cycle(**_kwargs):
        return cycle

    def fake_context(received_cycle, acct):
        seen.append((received_cycle, acct.id))
        return {"decision_cycle_id": received_cycle.decision_cycle_id}

    monkeypatch.setattr(canonical_cycle, "get_canonical_cycle", fake_cycle)
    monkeypatch.setattr(canonical_cycle, "build_strategy_context_from_cycle", fake_context)
    account_a = StrategyAccount(id="a", model="model_a")
    account_b = StrategyAccount(id="b", model="model_a")
    assert context_builder.build_strategy_context(account_a)["decision_cycle_id"] == "cycle-1"
    assert strategy_api._build_strategy_context(account_b)["decision_cycle_id"] == "cycle-1"
    assert seen[0][0] is seen[1][0]


def test_canonical_cycle_cache_builds_once_per_deterministic_slot(monkeypatch):
    from app.services import canonical_cycle

    canonical_cycle.clear_canonical_cycle_cache()
    calls = []
    sentinel = SimpleNamespace(decision_cycle_id="cycle-1")

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(canonical_cycle, "_build_uncached_cycle", fake_builder)
    first = canonical_cycle.get_canonical_cycle(
        is_min_temp=False,
        target_date=datetime(2026, 7, 20, tzinfo=UTC).date(),
        event_slug="highest-temperature-in-hong-kong-on-july-20-2026",
    )
    second = canonical_cycle.get_canonical_cycle(
        is_min_temp=False,
        target_date=datetime(2026, 7, 20, tzinfo=UTC).date(),
        event_slug="highest-temperature-in-hong-kong-on-july-20-2026",
    )
    assert first is second is sentinel
    assert len(calls) == 1


def test_legacy_9d_profile_is_derived_without_mutating_canonical_results(monkeypatch):
    from app.services import canonical_cycle, model_service
    from execution.strategy_account import StrategyAccount

    calls = []

    def fake_probs(mean, std, *_args, **_kwargs):
        calls.append((mean, std))
        return {"30-31": 0.9}

    monkeypatch.setattr(model_service, "compute_bucket_probs", fake_probs)
    cycle = SimpleNamespace(
        all_results={"9d": {"mean": 30.0, "std": 2.0, "probs": {"30-31": 0.5}}},
        markets=[],
        is_min_temp=False,
        state={"max_so_far": 30.0, "min_so_far": 25.0},
    )
    account = StrategyAccount(
        id="profile",
        model="9d",
        params={"bias": 0.3, "std_mult": 1.5},
    )
    view = canonical_cycle._account_model_view(cycle, account)
    assert calls == [(30.3, 3.0)]
    assert view["probs"] == {"30-31": 0.9}
    assert cycle.all_results["9d"]["mean"] == 30.0
