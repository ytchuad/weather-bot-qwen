import pandas as pd
import re

# 1. 讀取檔案
df = pd.read_parquet("./hk_daily_forecast/daily_forecast_all.parquet")

print(f"📂 原始筆數: {len(df)}")
print(f"📋 原始欄位: {df.columns.tolist()}")

# 2. 過濾掉「發佈日期」欄位中的無效值（只保留 YYYY-MM-DD 格式）
# 同時移除該欄位為空值或亂碼的列
df = df[df['forecast_issue_date'].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)]

print(f"🧹 清理後筆數: {len(df)}")

# 3. 正確轉換日期（手動指定格式，避免警告）
df['issue_dt'] = pd.to_datetime(df['forecast_issue_date'], format='%Y-%m-%d', errors='coerce')
df['target_dt'] = pd.to_datetime(df['query_date'], format='%Y-%m-%d', errors='coerce')

# 4. 檢查邏輯：發布日必須 <= 目標日
invalid = df[df['issue_dt'] > df['target_dt']]
print(f"❌ 邏輯錯誤（發布日晚於預測日）: {len(invalid)} 筆")

# 5. 檢查溫度合理性（最高溫 >= 最低溫）
invalid_temp = df[df['forecast_max_temp'] < df['forecast_min_temp']]
print(f"❌ 邏輯錯誤（最高溫 < 最低溫）: {len(invalid_temp)} 筆")

# 6. 儲存乾淨版本（覆蓋原檔，或另存新檔）
df.to_parquet("./hk_daily_forecast/daily_forecast_clean.parquet", index=False)
print(f"\n✅ 清理完成！乾淨檔案儲存於: ./hk_daily_forecast/daily_forecast_clean.parquet")
print(f"📊 最終有效筆數: {len(df)}")