import pandas as pd, json, glob

p = glob.glob("data/layer_a/date=2026-07-21/hour=12/cycles-*.parquet")
if p:
    df = pd.read_parquet(p[0])
    print("Columns:", list(df.columns))
    print("Shape:", df.shape)
    record = json.loads(df["record_json"].iloc[0])
    print("Record keys:", list(record.keys()))
    for k in ["event_date", "market_kind", "decision_cycle_id"]:
        print(f"  {k}: {record.get(k, 'MISSING')!r}")
    dates = set()
    for r in df["record_json"].head(5):
        dates.add(json.loads(r).get("event_date"))
    print("Event dates (first 5):", dates)
    has_mk = sum(1 for r in df["record_json"] if json.loads(r).get("market_kind") is not None)
    print(f"Records with market_kind: {has_mk}/{len(df)}")
    from collections import Counter
    kinds = Counter(json.loads(r).get("market_kind") for r in df["record_json"])
    print("market_kind values:", dict(kinds))
