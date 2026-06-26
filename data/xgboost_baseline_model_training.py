import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_parquet("./feature_store_enhanced.parquet")

# 分割特徵與標籤
X = df.drop(columns=['decision_time', 'remaining_upside'])
y = df['remaining_upside']

# 按時間分割 (避免未來資訊洩漏)
split_date = "2024-01-01"
train = df[df['decision_time'] < split_date]
test = df[df['decision_time'] >= split_date]

X_train = train.drop(columns=['decision_time', 'remaining_upside'])
y_train = train['remaining_upside']
X_test = test.drop(columns=['decision_time', 'remaining_upside'])
y_test = test['remaining_upside']

model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"✅ 測試集 RMSE: {rmse:.4f}°C")

# 顯示特徵重要性
importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print("\n📊 特徵重要性 Top 10:")
print(importance.head(10))