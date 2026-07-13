"""B — HF sync pre-check (本地 vs HF 一致性確認).

用途：在「跑分析 / 回測 / push HF」**之前**先跑一次，確認本地 data/export/
跟 HF Space 的 export-snapshots 端點沒有落差，避免像 2026-07-12 那樣
17:24–23:44 的資料「只存在 HF 暫存、還沒抓回來」就悄悄遺失。

它做兩件事：
  1. 抓取 HF export-snapshots 端點，按日期取「每個 date 的最後一筆 timestamp」
     （HF 端目前手上最新的資料到哪）。
  2. 比對本地 CSV 的「每個 date 最後一筆 timestamp」。
  若有任何 date 是「HF > 本地」→ 代表有資料在 HF 上、但本地還沒抓 →
     **亮紅燈 + 印出遺失時窗**，並回傳非零 exit code (CI/排程可據此擋下)。

注意：這是「預防遺失」的守門員，不是備份工具。真正把資料救回本地還是
要靠 scripts/download_snapshots.py（append 新列、不刪本地資料）。

Usage:
    python scripts/check_hf_sync.py [HF_SPACE_URL]
    echo %errorlevel%   # 0 = 同步, 1 = 有落差(警告)
"""

from __future__ import annotations

import json
import sys
import csv as csv_module
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

EXPORT_DIR = Path("data/export")
DEFAULT_HF = "https://shea-hilton-weather-prediction.hf.space"


def _local_last_ts() -> dict[str, datetime]:
    """回傳 {date: 本地最後一筆 timestamp} (CSV 最後一列)。"""
    out: dict[str, datetime] = {}
    for csv_path in sorted(EXPORT_DIR.glob("*.csv")):
        date = csv_path.stem
        last = None
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv_module.DictReader(f):
                ts = row.get("timestamp", "")
                if not ts or "T" not in ts:
                    continue
                try:
                    t = datetime.fromisoformat(ts)
                except ValueError:
                    continue
                if last is None or t > last:
                    last = t
        if last is not None:
            out[date] = last
    return out


def _hf_last_ts(base_url: str) -> dict[str, datetime]:
    """抓 HF export-snapshots 端點，回傳 {date: HF 最後一筆 timestamp}。

    分日查詢：無參數端點會依 ``ORDER BY timestamp ASC LIMIT 10000`` 截斷，
    資料量超過 10000 後只回傳最舊的 10000 筆（最新日期反而不見），導致
    假性「同步」。逐日查詢可規避截斷。
    """
    from datetime import date, timedelta
    out: dict[str, datetime] = {}
    today = date.today()
    dates = [(today - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(21, -1, -1)]
    for d in dates:
        url = f"{base_url.rstrip('/')}/api/data/export-snapshots?date={d}"
        try:
            with urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
        except Exception:
            continue
        snaps = payload.get("snapshots", [])
        for s in snaps:
            sd = s.get("snapshot_date", "")
            ts = s.get("timestamp", "")
            if not sd or "T" not in ts:
                continue
            try:
                t = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if sd not in out or t > out[sd]:
                out[sd] = t
    return out


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HF
    print(f"Checking sync: local data/export/  vs  HF {base_url}\n")

    try:
        hf = _hf_last_ts(base_url)
    except Exception as e:
        print(f"  [ERROR] 無法連上 HF export 端點: {e}")
        print("  請確認 Space 正在運行，或手動跑 download_snapshots.py。")
        return 2

    local = _local_last_ts()

    # 也找出「HF 有、本地沒有」的整個遺失日期
    missing_dates = sorted(set(hf) - set(local))

    # 找「HF 比本地新」的落差日期
    gaps = []
    for d in sorted(set(hf) & set(local)):
        if hf[d] > local[d]:
            gaps.append((d, local[d], hf[d]))

    ok = not gaps and not missing_dates
    if ok:
        print("  ✅ 同步：本地與 HF 最後一筆 timestamp 完全一致。")
        print(f"  共比對 {len(local)} 個日期。")
        return 0

    print("  ⚠️  [警告] 偵測到本地落後於 HF — 有資料尚未抓回本地！\n")
    if gaps:
        print("  ── 部分遺失 (HF 較新) ──")
        for d, lo, hi in gaps:
            delta = hi - lo
            print(f"    {d}: 本地最後 {lo:%m-%d %H:%M}  <  HF 最後 {hi:%m-%d %H:%M}"
                  f"  (落差 {delta})")
    if missing_dates:
        print("  ── 整日遺失 (HF 有、本地無) ──")
        for d in missing_dates:
            print(f"    {d}: 本地完全沒有，HF 最後 {hf[d]:%m-%d %H:%M}")
    print("\n  建議立刻執行：")
    print(f"    python scripts/download_snapshots.py {base_url}")
    print("  把上述落差抓回本地後再跑分析 / 回測 / push HF。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
