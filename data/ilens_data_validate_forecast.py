import pandas as pd
import glob
from datetime import datetime

df_forecast = pd.read_parquet("./hk_daily_forecast/daily_forecast_all.parquet")

# 1. 檢查「預測發布日」與「查詢標的日」的關係（預測只能發佈在標的日之前）
df_forecast['issue_dt'] = pd.to_datetime(df_forecast['forecast_issue_date'])
df_forecast['target_dt'] = pd.to_datetime(df_forecast['query_date'])
# 假設發布時間固定為 11:30，若發布日 > 標的日，代表邏輯錯誤
invalid = df_forecast[df_forecast['issue_dt'] > df_forecast['target_dt']]
print(f"❌ 邏輯錯誤（發布日晚於預測日）: {len(invalid)} 筆")

# 2. 檢查預報溫度的常識（最高溫 >= 最低溫）
invalid_temp = df_forecast[df_forecast['forecast_max_temp'] < df_forecast['forecast_min_temp']]
print(f"❌ 邏輯錯誤（最高溫 < 最低溫）: {len(invalid_temp)} 筆")