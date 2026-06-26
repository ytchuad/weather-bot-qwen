import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# 1. 讀取強化版 Feature Store
print("📂 讀取強化版特徵資料...")
df = pd.read_parquet("./feature_store_enhanced.parquet")

# 2. 準備特徵與標籤
X = df.drop(columns=['decision_time', 'remaining_upside'])
y = df['remaining_upside']

print(f"✅ 總樣本數: {len(X):,}")
print(f"✅ 特徵維度: {X.shape[1]}")

# 3. 按時間分割 (嚴守「不使用未來資料」原則)
split_date = pd.Timestamp("2024-01-01")
train_idx = df['decision_time'] < split_date
test_idx = df['decision_time'] >= split_date

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\n📊 訓練集: {len(X_train):,} 筆 (2016-2023)")
print(f"📊 測試集: {len(X_test):,} 筆 (2024-2026)")

# 4. 訓練 XGBoost 模型
print("\n🚀 訓練 XGBoost 模型中...")
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

# 5. 評估模型 (修正版：手動計算 RMSE 以相容舊版 sklearn)
y_pred = model.predict(X_test)

# ✅ 手動計算 RMSE (開根號)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📈 測試集表現:")
print(f"   ✅ RMSE: {rmse:.4f}°C")
print(f"   ✅ MAE:  {mae:.4f}°C")

# 6. 特徵重要性分析
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 特徵重要性排名 (Top 10):")
print(importance.head(10))

# 7. 繪製特徵重要性（選擇性）
plt.figure(figsize=(10, 6))
plt.barh(importance.head(10)['feature'], importance.head(10)['importance'])
plt.xlabel('重要性分數')
plt.title('XGBoost 特徵重要性 (預測剩餘上行空間)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\n📁 特徵重要性圖已儲存為 feature_importance.png")