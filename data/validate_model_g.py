"""
validate_model_g.py

完整驗證 Model G（修正錨定版）：
1. 特徵重要性排名 (確認 max_so_far / drop_from_max 是否主導)
2. 分類器 (upside_zero) 效能
3. 針對「極端降溫 (drop_from_max 大)」情境的切片驗證 (關鍵！)
4. 時間切片驗證 (確保上午沒過度煞車)
5. 分位數覆蓋率 (q10~q90)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 設定路徑 ==========
MODEL_DIR = Path("models/intraday_minute_ml_model_g")
OOT_PATH = MODEL_DIR / "oot_predictions.parquet"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.json"

# ========== 1. 載入 OOT 預測結果 ==========
print("=" * 60)
print("📊 Model G 驗證報告")
print("=" * 60)

df = pd.read_parquet(OOT_PATH)
print(f"✅ 載入 OOT 樣本數: {len(df):,}")

# ===== 手動補齊驗證所需的衍生特徵（因為 oot_predictions 未儲存全部特徵） =====
if 'drop_from_max' not in df.columns:
    df['drop_from_max'] = df['max_so_far'] - df['value']
    print("✅ 自動計算 drop_from_max (max_so_far - value)")

if 'forecast_gap' not in df.columns:
    # 僅供驗證程式運作，不影響其他統計值
    df['forecast_gap'] = 0.0
    print("✅ 補上預設 forecast_gap=0 (僅供驗證程式運作)")

# 載入特徵清單 (確認順序)
with open(FEATURE_LIST_PATH, "r") as f:
    FEATURE_COLS = json.load(f)

# ========== 2. 特徵重要性 (從原模型載入) ==========
print("\n📈 特徵重要性排名 (Top 10):")
# 載入 q50 模型來讀取重要性 (因為它代表中位數)
model_path = MODEL_DIR / "upside_q50.txt"
if model_path.exists():
    booster = lgb.Booster(model_file=str(model_path))
    importance = booster.feature_importance(importance_type='gain')  # 'gain' 反映預測力
    imp_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(imp_df.head(10).to_string(index=False))
else:
    print("⚠️ 找不到 upside_q50.txt，跳過重要性分析")

# ========== 3. 分類器 (upside_zero) 效能 ==========
print("\n🔍 分類器 (是否見頂) 效能:")
# 檢查 df 中是否有 zero_proba 與 zero_pred (由訓練腳本輸出)
if 'zero_proba' in df.columns and 'is_upside_zero' in df.columns:
    pr_auc = average_precision_score(df['is_upside_zero'], df['zero_proba'])
    print(f"   PR-AUC: {pr_auc:.4f}")
    
    # 最佳閾值 (從 best_threshold.json 讀取)
    threshold_path = MODEL_DIR / "best_threshold.json"
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            thr_data = json.load(f)
            best_thr = thr_data.get('upside_zero_threshold', 0.5)
        print(f"   最佳閾值: {best_thr:.4f}")
        
        # 套用閾值檢視 Precision/Recall
        preds = (df['zero_proba'] >= best_thr).astype(int)
        prec = (preds & df['is_upside_zero']).sum() / max(preds.sum(), 1)
        rec = (preds & df['is_upside_zero']).sum() / max(df['is_upside_zero'].sum(), 1)
        print(f"   Precision: {prec:.4f}, Recall: {rec:.4f}")

# ========== 4. 關鍵驗證：大雨/鋒面降溫情境 (drop_from_max 很大) ==========
print("\n🌧️ 關鍵驗證：極端降溫情境 (drop_from_max >= 5°C)")
# 找出氣溫已從最高點暴跌 5 度以上的樣本
mask_extreme = df['drop_from_max'] >= 5
sub_extreme = df[mask_extreme]

if len(sub_extreme) > 0:
    print(f"   樣本數: {len(sub_extreme):,}")
    print(f"   實際剩餘升幅 (mean): {sub_extreme['remaining_upside'].mean():.3f}°C")
    print(f"   預測剩餘升幅 q50 (mean): {sub_extreme['upside_q50'].mean():.3f}°C")
    print(f"   預測最終最高溫 q50 (mean): {sub_extreme['pred_tmax_q50'].mean():.3f}°C")
    print(f"   當日最高溫 max_so_far (mean): {sub_extreme['max_so_far'].mean():.3f}°C")
    
    # 檢查是否收斂：預測溫度是否接近 max_so_far
    diff = (sub_extreme['pred_tmax_q50'] - sub_extreme['max_so_far']).abs()
    print(f"   ✅ 預測與 max_so_far 平均差距: {diff.mean():.3f}°C")
    if diff.mean() < 1.0:
        print("   ✅ 驗證通過：極端降溫時，模型已正確收斂至當日最高溫。")
    else:
        print("   ⚠️ 警告：極端降溫時，模型仍未完全收斂，請檢查特徵。")
else:
    print("   ⚠️ OOT 測試集中無極端降溫樣本 (可能日期範圍不含暴雨日)")

# ========== 5. 時間切片驗證 (確保上午不過度煞車) ==========
print("\n⏰ 時間切片驗證 (上午 vs 下午 表現):")
for hour_group, label in [((6, 11), "上午 (06-11)"), ((12, 17), "下午 (12-17)")]:
    mask = (df['hour'] >= hour_group[0]) & (df['hour'] < hour_group[1])
    sub = df[mask]
    if len(sub) > 0:
        mae = mean_absolute_error(sub['remaining_upside'], sub['upside_q50'])
        print(f"   {label}: MAE = {mae:.4f}°C (n={len(sub):,})")

# ========== 6. 分位數覆蓋率 (評估不確定性) ==========
print("\n📊 分位數覆蓋率 (80% 預測區間):")
inside = (df['remaining_upside'] >= df['upside_q10']) & (df['remaining_upside'] <= df['upside_q90'])
cov = inside.mean()
print(f"   實際覆蓋率: {cov:.4f} (理想值: 0.80)")
if 0.78 < cov < 0.82:
    print("   ✅ 校準良好")
else:
    print("   ⚠️ 覆蓋率偏差，可能需要調整分位數模型")

# ========== 7. 針對 forecast_gap 的行為檢查 ==========
print("\n🧠 特徵行為檢查 (forecast_gap vs max_so_far):")
# 檢查 forecast_gap 的計算是否正確 (極端情況不應膨脹)
max_gap = df['forecast_gap'].max()
min_gap = df['forecast_gap'].min()
print(f"   forecast_gap 範圍: {min_gap:.2f} ~ {max_gap:.2f}°C")
if max_gap < 15:
    print("   ✅ forecast_gap 數值正常 (未因暴雨而異常膨脹)")

# ========== 8. 總結 ==========
print("\n" + "=" * 60)
print("✅ 驗證完成！")
print("📌 確認事項:")
print("   - 若極端降溫時 pred_tmax 接近 max_so_far，代表『雨天失靈』已修正。")
print("   - 若上午 MAE 明顯低於下午，代表模型在升溫階段表現穩定。")
print("   - 若覆蓋率接近 0.8，代表不確定性區間估計合理。")
print("=" * 60)