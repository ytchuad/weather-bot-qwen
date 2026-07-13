"""Layer 1 — 模型機率準確度評估 (model probability accuracy).

這一層評的是「模型給的機率準不準」，與下注策略 (rr / edge / hard_flat)
**完全無關**。它衡量的是機率預測本身的品質，是策略能不能賺錢的地基。

對每一個模型 (model_key)，對照當日實際贏家 bucket，計算三個指標：

  1. Multiclass Brier score (布萊爾分數)
       每個 snapshot 的貢獻 = sum_c (p_c - o_c)^2，再對所有 snapshot 平均。
       o_c = 1 若 bucket c 是當日贏家，否則 0。
       range [0, 2]；越低越好；均勻猜 (p=1/K) 的基準 = (K-1)/K。

  2. Multiclass Log loss (對數損失)
       = -mean over snapshots of log(p_winner)。
       對「過度自信又錯」懲罰最重；越低越好；均勻基準 = log(K)。

  3. Calibration / reliability diagram (校準 / 可靠度圖)
       把每個 (snapshot, bucket) 的預測機率 p 與結果 o∈{0,1}
       合併後分 10 箱，看「說 p 時，實際 o 是不是真的 = p」。
       點落在對角線 y=x 上 = 完美校準。

注意：Brier / Log loss 的「尺度」不同，只能在同一指標內比較模型，
不能直接拿 Brier 的數字跟 Log loss 的數字比大小。

資料：data/export/*.csv 的 context_json.model_probs 已存好每個模型每個
bucket 的機率，且每個 snapshot 都存了全部模型，因此可以自由重評，不需重跑模型。
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from execution.ensemble.strategy import parse_bucket_bounds

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/export")

# 預設 ensemble 成員 (與 EnsembleParams.model_weights 一致: a/b/c 各 1/3)
ENSEMBLE_WEIGHTS = {"model_a": 1/3, "model_b": 1/3, "model_c": 1/3}

EPS = 1e-12  # 避免 log(0)


# ── 資料載入 ────────────────────────────────────────────────────────

def load_for_eval(dates=None) -> dict:
    """回傳 {date: {"settle_max": float, "snaps": [(timestamp_str, model_probs)]}}。

    - 每個 (date, timestamp) 只取第一次出現的非空 model_probs (dedup by 1s)。
    - settle_max = 該日所有 snapshot 中 max(actual_temp, max_so_far) 的最大值，
      與 strategy.settle_day 的結算溫度邏輯一致。
    """
    out = {}
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if dates:
        csv_files = [f for f in csv_files if f.stem in dates]
    for fpath in csv_files:
        date_str = fpath.stem
        seen_ts = set()
        snaps = []
        settle_max = -float("inf")
        with open(fpath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts_str = row.get("timestamp", "")
                if not ts_str:
                    continue
                k = ts_str[:19].replace(":", "").replace("-", "").replace("T", "")
                if k in seen_ts:
                    continue
                ctx_raw = row.get("context_json", "")
                if not ctx_raw or not ctx_raw.strip():
                    continue
                try:
                    ctx = json.loads(ctx_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                mp = ctx.get("model_probs")
                if not mp:
                    continue
                seen_ts.add(k)
                snaps.append((ts_str, mp))

                # settle_max: 取該 snapshot 實際溫度與當日最高中的較大者
                try:
                    at = float(row["actual_temp"]) if row.get("actual_temp") not in (None, "") else None
                except (ValueError, TypeError):
                    at = None
                try:
                    msf = float(row["max_so_far"]) if row.get("max_so_far") not in (None, "") else None
                except (ValueError, TypeError):
                    msf = None
                cand = max(x for x in (at, msf) if x is not None)
                settle_max = max(settle_max, cand)

        if snaps:
            out[date_str] = {"settle_max": settle_max, "snaps": snaps}
    return out


def winner_bucket(settle_max: float, buckets: list[str]) -> str | None:
    """給定當日最高溫，回傳哪個 bucket 贏 (落在 [lo, hi) 半開區間)。

    與 strategy.parse_bucket_bounds 一致:
      '25-26' -> (25, 26);  '>=34' -> (34, inf);  '<23' -> (-inf, 23)
    若溫度落在 bucket 之間的縫隙 (資料 gap)，回傳 None。
    """
    for b in buckets:
        lo, hi = parse_bucket_bounds(b)
        if lo <= settle_max < hi:
            return b
    return None


# ── 指標計算 ────────────────────────────────────────────────────────

def _norm(preds: dict) -> dict:
    """把某 snapshot 的 bucket 機率做機率和=1 的歸一化 (避免 log/sum 異常)。"""
    s = sum(v for v in preds.values() if v is not None)
    if s <= 0:
        return preds
    return {k: (v / s if v is not None else 0.0) for k, v in preds.items()}


def brier_multiclass(model_preds: list[dict], winners: list[str | None]) -> float:
    """多類 Brier: 每 snapshot sum_c (p_c - o_c)^2 的平均。

    model_preds[i] = 第 i 個 snapshot 的 {bucket: p} ; winners[i] = 當日贏家。
    忽略 winners[i] is None 或贏家不在該 snapshot bucket 集合的樣本。
    """
    tot, n = 0.0, 0
    for preds, w in zip(model_preds, winners):
        if w is None or w not in preds:
            continue
        p = _norm(preds)
        score = sum((p.get(c, 0.0) - (1.0 if c == w else 0.0)) ** 2 for c in p)
        tot += score
        n += 1
    return tot / n if n else float("nan")


def logloss_multiclass(model_preds: list[dict], winners: list[str | None]) -> float:
    """多類 Log loss: -mean log(p_winner)。p_winner 夾在 [EPS, 1-EPS]。"""
    tot, n = 0.0, 0
    for preds, w in zip(model_preds, winners):
        if w is None or w not in preds:
            continue
        p = _norm(preds)
        pw = min(max(p.get(w, 0.0), EPS), 1.0 - EPS)
        tot += -np.log(pw)
        n += 1
    return tot / n if n else float("nan")


def calibration_pairs(model_preds: list[dict], winners: list[str | None]) -> list[tuple[float, int]]:
    """展平成 (p, o) 配對: 每個 (snapshot, bucket) 一筆，o=1 若該 bucket 贏。

    用於畫 reliability diagram (把所有 bucket 當成獨立二元預測)。
    注意: 同 snapshot 的 11 個 bucket 機率相關，故配對非獨立，僅作診斷用。
    """
    pairs = []
    for preds, w in zip(model_preds, winners):
        if w is None:
            continue
        p = _norm(preds)
        for c, pc in p.items():
            pairs.append((pc, 1 if c == w else 0))
    return pairs


def reliability_table(pairs: list[tuple[float, int]], nbins: int = 10) -> list[dict]:
    """把 (p, o) 配對分箱，回傳每箱的統計。"""
    bins = [[] for _ in range(nbins)]
    for p, o in pairs:
        p = min(max(p, 0.0), 1.0 - 1e-9)
        idx = min(int(p * nbins), nbins - 1)
        bins[idx].append((p, o))
    rows = []
    for i, b in enumerate(bins):
        if not b:
            continue
        ps = [x[0] for x in b]
        os = [x[1] for x in b]
        lo = i / nbins
        hi = (i + 1) / nbins
        rows.append({
            "bin": f"[{lo:.2f},{hi:.2f})",
            "n": len(b),
            "mean_pred": float(np.mean(ps)),
            "emp_rate": float(np.mean(os)),
        })
    return rows


# ── 主流程 ──────────────────────────────────────────────────────────

def evaluate(dates=None, models=None, log_path: str | None = None):
    data = load_for_eval(dates)

    # 收集每個 model 的預測序列 + 每個 date 的 winner
    # model_preds_seq[model][date] = list of {bucket:p}
    model_preds_seq: dict[str, dict[str, list]] = {}
    winners_by_date: dict[str, str | None] = {}
    settle_by_date = {}
    all_model_keys = set()
    for date, info in data.items():
        settle_by_date[date] = info["settle_max"]
        # 該日所有 snapshot 共用的 bucket 集合 (取第一個 snapshot 第一個模型的 keys)
        first_mp = info["snaps"][0][1]
        buckets = list(first_mp.values())[0].keys() if first_mp else []
        # 用 settle_max 決定贏家 bucket (必須傳 bucket 標籤, 不是 model key!)
        winners_by_date[date] = winner_bucket(info["settle_max"], buckets)
        for mk, preds in first_mp.items():
            all_model_keys.add(mk)
            model_preds_seq.setdefault(mk, {})[date] = info["snaps"] and [
                mp[mk] for _, mp in info["snaps"] if mk in mp
            ]

    # 若指定 models，過濾
    if models:
        all_model_keys = {m for m in all_model_keys if m in models}

    # 對每個 model 收集跨日的 (preds_per_snap, winner) 序列
    rows = []
    for mk in sorted(all_model_keys):
        seq_dates = [d for d in sorted(data) if d in model_preds_seq.get(mk, {})]
        preds_list, winners_list = [], []
        for d in seq_dates:
            for pr in model_preds_seq[mk][d]:
                preds_list.append(pr)
                winners_list.append(winners_by_date[d])
        brier = brier_multiclass(preds_list, winners_list)
        ll = logloss_multiclass(preds_list, winners_list)
        n_snaps = len(preds_list)
        rows.append({
            "model": mk,
            "is_ensemble_member": mk in ENSEMBLE_WEIGHTS,
            "n_snaps": n_snaps,
            "brier": brier,
            "logloss": ll,
            "preds_list": preds_list,
            "winners_list": winners_list,
            "pairs": calibration_pairs(preds_list, winners_list),
        })

    # ensemble (加權平均機率) 額外評一次
    ens_rows = []
    for d in sorted(data):
        mp0 = data[d]["snaps"][0][1]
        members = {m: w for m, w in ENSEMBLE_WEIGHTS.items() if m in mp0}
        if len(members) < 2:
            continue
        ens_preds_list = []
        for _, mp in data[d]["snaps"]:
            combined = {}
            for m, w in members.items():
                if m not in mp:
                    continue
                for c, pc in _norm(mp[m]).items():
                    combined[c] = combined.get(c, 0.0) + w * pc
            ens_preds_list.append(combined)
        ens_rows.append((ens_preds_list, winners_by_date[d]))

    ens_preds = []
    ens_win = []
    for ens_preds_list, w in ens_rows:
        for pr in ens_preds_list:
            ens_preds.append(pr)
            ens_win.append(w)  # 同 date 的每個 snapshot 共用當日 winner
    ens_brier = brier_multiclass(ens_preds, ens_win)
    ens_ll = logloss_multiclass(ens_preds, ens_win)
    ens_pairs = calibration_pairs(ens_preds, ens_win)

    lines = []
    lines.append("=" * 100)
    lines.append("  LAYER 1 — MODEL PROBABILITY ACCURACY EVALUATION")
    lines.append("  (獨立於下注策略; 量的是 model_probs 本身的準確度)")
    lines.append("=" * 100)
    lines.append(f"  Dates evaluated : {sorted(data)}")
    lines.append(f"  Settle temps     : " + ", ".join(
        f"{d}={settle_by_date[d]:.1f}({winners_by_date[d]})" for d in sorted(data)
    ))
    lines.append("")

    # Brier / Log loss 總表
    hdr = f"{'Model':<14}{'ens?':>5}{'Snaps':>7}{'Brier':>10}{'LogLoss':>10}{'Brier_base':>12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    # 基準線: 均勻猜 (K = 平均 bucket 數) -> Brier=(K-1)/K, LogLoss=log(K)
    k_avg = int(np.mean([len(r["pairs"]) / max(n_snaps := r["n_snaps"], 1) for r in rows if r["pairs"]]) + 0.5)
    k_avg = max(k_avg, 1)
    base_brier = (k_avg - 1) / k_avg
    base_ll = float(np.log(k_avg))
    for r in sorted(rows, key=lambda x: x["brier"]):
        lines.append(
            f"{r['model']:<14}{('Y' if r['is_ensemble_member'] else ''):>5}"
            f"{r['n_snaps']:>7}{r['brier']:>10.4f}{r['logloss']:>10.4f}{base_brier:>12.4f}"
        )
    lines.append(f"{'ENSEMBLE(a/b/c)':<14}{'Y':>5}{len(ens_preds):>7}"
                 f"{ens_brier:>10.4f}{ens_ll:>10.4f}{base_brier:>12.4f}")
    lines.append("")
    lines.append(f"  基準線說明: 若永遠均勻猜 (每 bucket p=1/K, K≈{k_avg}),")
    lines.append(f"    Brier = (K-1)/K = {base_brier:.4f} ; LogLoss = log(K) = {base_ll:.4f}")
    lines.append("  模型指標若高於基準線 = 比亂猜還差。越低越好。")

    # Calibration 詳細表 (只對 ensemble 成員 + ensemble 整體)
    lines.append("")
    lines.append("-" * 70)
    lines.append("  CALIBRATION / RELIABILITY DIAGRAM (校準: 說 p 時實際贏率是否 = p)")
    lines.append("  對角線 y=x 上 = 完美校準; 線下 = 過度自信; 線上 = 信心不足")
    lines.append("-" * 70)
    cal_targets = [r for r in rows if r["is_ensemble_member"]] + [
        {"model": "ENSEMBLE(a/b/c)", "pairs": ens_pairs}
    ]
    for r in cal_targets:
        lines.append(f"\n  [{r['model']}]")
        rt = reliability_table(r["pairs"])
        if not rt:
            lines.append("    (無資料)")
            continue
        lines.append(f"    {'bin':<14}{'n':>6}{'mean_pred':>11}{'emp_win%':>11}")
        for row in rt:
            lines.append(f"    {row['bin']:<14}{row['n']:>6}{row['mean_pred']:>11.3f}"
                         f"{row['emp_rate']*100:>10.1f}%")

    text = "\n".join(lines)
    print(text)
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=None, help="限制評估日期")
    ap.add_argument("--models", nargs="*", default=None, help="只看指定模型")
    ap.add_argument("--log", default="output/model_eval.log")
    args = ap.parse_args()
    evaluate(dates=args.dates, models=args.models, log_path=args.log)
