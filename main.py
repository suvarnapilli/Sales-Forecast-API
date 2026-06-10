import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_excel("data.xlsx")

# Clean column names
df.columns = df.columns.str.lower().str.strip()
print("Columns:", df.columns)

# Rename columns
df.rename(columns={
    'date': 'date',
    'state': 'state',
    'total': 'sales'
}, inplace=True)

# Keep only required columns
df = df[['date', 'state', 'sales']]

# Convert date properly
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Remove invalid dates
df = df.dropna(subset=['date'])

# Sort data
df = df.sort_values(['state', 'date'])

# Check data scale (optional but useful)
print("\nSales Summary:\n", df['sales'].describe())

# =========================
# 2. HANDLE MISSING DATES
# =========================
def fill_missing_dates(df):
    states = df['state'].unique()
    final = []

    for s in states:
        temp = df[df['state'] == s].copy()
        temp = temp.set_index('date').sort_index()

        # 🔥 FIXED LINE (important)
        temp = temp.asfreq('D')

        temp['sales'] = temp['sales'].interpolate()
        temp['state'] = s

        final.append(temp.reset_index())

    return pd.concat(final)

df = fill_missing_dates(df)

# =========================
# 3. FEATURE ENGINEERING
# =========================
def create_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['lag_30'] = df['sales'].shift(30)

    df['rolling_mean_7'] = df['sales'].rolling(7).mean()

    return df.dropna()

df = create_features(df)

# =========================
# 4. TRAIN TEST SPLIT
# =========================
split_date = df['date'].quantile(0.8)

train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

features = ['lag_1','lag_7','lag_30','rolling_mean_7','day_of_week','month']

# Fill any remaining NaN
train = train.fillna(0)
test = test.fillna(0)

# =========================
# 5. XGBOOST MODEL
# =========================
xgb = XGBRegressor()
xgb.fit(train[features], train['sales'])
xgb_preds = xgb.predict(test[features])

mae_xgb = mean_absolute_error(test['sales'], xgb_preds)

# =========================
# 6. SARIMA MODEL
# =========================
# 🔥 Using simpler params to avoid issues
sarima = SARIMAX(train['sales'], order=(1,1,0), seasonal_order=(0,1,1,7))
sarima_fit = sarima.fit(disp=False)
sarima_preds = sarima_fit.forecast(len(test))

mae_sarima = mean_absolute_error(test['sales'], sarima_preds)

# =========================
# 7. COMPARE MODELS
# =========================
print("\nMAE Scores:")
print("XGBoost:", mae_xgb)
print("SARIMA:", mae_sarima)

if mae_xgb < mae_sarima:
    best_model = "xgboost"
else:
    best_model = "sarima"

print("Best Model:", best_model)

# =========================
# 8. SAVE MODEL
# =========================
if best_model == "xgboost":
    joblib.dump(xgb, "best_model.pkl")
else:
    joblib.dump(sarima_fit, "best_model.pkl")

# =========================
# 9. FORECAST 56 DAYS
# =========================
print("\nGenerating 56-day forecast...")

if best_model == "sarima":
    future_preds = sarima_fit.forecast(56)
    print("Forecast:", future_preds[:10])
else:
    future_preds = xgb.predict(test[features])[:56]
    print("Forecast:", future_preds[:10])