import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# 1. سحب البيانات من MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

symbol = "XAUUSDm"
timeframe = mt5.TIMEFRAME_M1
utc_now = datetime.now(timezone.utc)
utc_from = utc_now - timedelta(days=60)

# سحب البيانات من MetaTrader 5
rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_now)
mt5.shutdown()

# تحويل البيانات إلى DataFrame
if rates is None:
    print("لم يتم سحب أي بيانات.")
    quit()

rates_frame = pd.DataFrame(rates)
rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

# 2. إضافة ميزات تزحلق (Lagging Features)
rates_frame['high_lag_1'] = rates_frame['high'].shift(1)
rates_frame['low_lag_1'] = rates_frame['low'].shift(1)
rates_frame['open_lag_1'] = rates_frame['open'].shift(1)
rates_frame['high_lag_2'] = rates_frame['high'].shift(2)
rates_frame['low_lag_2'] = rates_frame['low'].shift(2)

rates_frame = rates_frame.dropna()

# إعداد بيانات التدريب والاختبار
X = rates_frame[['open_lag_1', 'high_lag_1', 'low_lag_1', 'high_lag_2', 'low_lag_2', 'tick_volume']]
y = rates_frame['close']

# تقسيم البيانات إلى تدريب واختبار (80% تدريب و20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. بناء نموذج RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# تدريب النموذج على بيانات التدريب
model.fit(X_train, y_train)

# 4. التوقع على بيانات التدريب والاختبار
y_train_pred = model.predict(X_train)  # توقعات على بيانات التدريب
y_test_pred = model.predict(X_test)    # توقعات على بيانات الاختبار

# 5. حساب المقاييس لكل من بيانات التدريب والاختبار
# مقاييس بيانات التدريب
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

# مقاييس بيانات الاختبار
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# 6. طباعة النتائج
print("Result Train")
print(f"R² (Training): {r2_train}")
print(f"MAE (Training): {mae_train}")
print(f"RMSE (Training): {rmse_train}")

print("\n Result Test")
print(f"R² (Testing): {r2_test}")
print(f"MAE (Testing): {mae_test}")
print(f"RMSE (Testing): {rmse_test}")

# 7. توقع الخطوة القادمة (سعر الإغلاق المتوقع)
last_row = rates_frame.iloc[-1]

X_next = pd.DataFrame([[
    last_row['open_lag_1'],
    last_row['high_lag_1'],
    last_row['low_lag_1'],
    last_row['high_lag_2'],
    last_row['low_lag_2'],
    last_row['tick_volume']
]], columns=['open_lag_1', 'high_lag_1', 'low_lag_1', 'high_lag_2', 'low_lag_2', 'tick_volume'])

next_close_prediction = model.predict(X_next)

print(f"\nProduction Price {next_close_prediction[0]}")
