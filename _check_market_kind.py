import pandas as pd, json, glob
from collections import Counter

all_kinds = Counter()
for hour_dir in sorted(glob.glob("data/layer_a/date=2026-07-21/hour=*")):
    for p in glob.glob(hour_dir + "/cycles-*.parquet"):
        df = pd.read_parquet(p)
        for r in df["record_json"]:
            rec = json.loads(r)
            mk = rec.get("market_kind")
            if mk:
                all_kinds[mk] += 1
print("Model market_kind distribution:", dict(all_kinds))
print("Total model records:", sum(all_kinds.values()))

mkt_kinds = Counter()
import zstandard as zstd
for hour_dir in sorted(glob.glob("data/layer_a_market/date=2026-07-21/hour=*/*")):
    for p in glob.glob(hour_dir + "/snapshots-*.jsonl.zst"):
        try:
            data = zstd.ZstdDecompressor().decompress(p.read_bytes())
            for line in data.decode("utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    mk = rec.get("market_kind")
                    if mk:
                        mkt_kinds[mk] += 1
        except:
            pass
print("Market market_kind distribution:", dict(mkt_kinds))

# Also check weather records
weather_kinds = Counter()
for hour_dir in sorted(glob.glob("data/layer_a_weather/date=2026-07-21/hour=*/*")):
    for p in glob.glob(hour_dir + "/snapshots-*.jsonl.zst"):
        try:
            data = zstd.ZstdDecompressor().decompress(p.read_bytes())
            for line in data.decode("utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    mk = rec.get("market_kind")
                    if mk:
                        weather_kinds[mk] += 1
        except:
            pass
print("Weather market_kind distribution:", dict(weather_kinds))
