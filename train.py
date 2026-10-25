import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import joblib

# ==============================================================================
# PROJECT: Air Quality Index (AQI) Prediction (India CPCB Standards)
# ==============================================================================

# ==============================================================================
# Step 1: Load Indian Air Quality Dataset
# ==============================================================================
data_path = Path(__file__).resolve().parent / "data" / "air-quality-india.csv"
if not data_path.exists():
    data_path = Path(__file__).resolve().parent / "dataset" / "air-quality-india.csv"
if not data_path.exists():
    data_path = Path("data/air-quality-india.csv")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# ==============================================================================
# Step 2: Exploratory Data Analysis (EDA)
# ==============================================================================
print("\n--- Dataset Overview ---")
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:\n", df.head(3))
print("\nMissing values:\n", df.isnull().sum())
print("\nPM2.5 Summary Stats:\n", df['PM2.5'].describe())

# ==============================================================================
# Step 3: Compute Official CPCB Indian Air Quality Index (AQI)
# ==============================================================================
# Central Pollution Control Board (CPCB) India Sub-index formula for PM2.5 (ug/m3):
# Breakpoints:
# 0-30   -> AQI 0-50     (Good)
# 31-60  -> AQI 51-100   (Satisfactory)
# 61-90  -> AQI 101-200  (Moderate)
# 91-120 -> AQI 201-300  (Poor)
# 121-250-> AQI 301-400  (Very Poor)
# 250+   -> AQI 401-500  (Severe)

def calculate_cpcb_aqi(pm):
    """Calculates official Indian CPCB AQI from PM2.5 concentration."""
    if pm <= 30:
        return pm * (50 / 30)
    elif pm <= 60:
        return 50 + (pm - 30) * (50 / 30)
    elif pm <= 90:
        return 100 + (pm - 60) * (100 / 30)
    elif pm <= 120:
        return 200 + (pm - 90) * (100 / 30)
    elif pm <= 250:
        return 300 + (pm - 120) * (100 / 130)
    else:
        return min(500, 400 + (pm - 250) * (100 / 130))

def get_aqi_category(aqi):
    """Maps AQI score to official CPCB Health Advisory Category."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def get_health_advisory(category):
    """Returns official public health precautions based on AQI category."""
    advisories = {
        "Good": "Minimal health impact. Clean air.",
        "Satisfactory": "Minor breathing discomfort to sensitive people.",
        "Moderate": "Breathing discomfort to people with asthma, lungs, and heart diseases.",
        "Poor": "Breathing discomfort to most people on prolonged exposure.",
        "Very Poor": "Respiratory illness on prolonged exposure. Avoid strenuous outdoor activity.",
        "Severe": "Affects healthy people and seriously impacts those with existing diseases."
    }
    return advisories.get(category, "No advisory available.")

df['AQI'] = df['PM2.5'].apply(calculate_cpcb_aqi)
df['AQI_Category'] = df['AQI'].apply(get_aqi_category)

print("\n--- Indian AQI Distribution in Dataset ---")
print(df['AQI_Category'].value_counts())

# ==============================================================================
# Step 4: Feature Engineering (Temporal & Lag Features)
# ==============================================================================
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['DayOfWeek'] = df['Timestamp'].dt.day_of_week
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Cyclical Encoding for Hour and Month (preserves time continuity)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Time-Series Lag Features (past pollution history to predict current/future AQI)
df['PM2.5_lag1'] = df['PM2.5'].shift(1)               # 1 hour ago
df['PM2.5_lag2'] = df['PM2.5'].shift(2)               # 2 hours ago
df['PM2.5_lag24'] = df['PM2.5'].shift(24)             # 24 hours ago (same time yesterday)
df['PM2.5_rolling24'] = df['PM2.5'].shift(1).rolling(24).mean() # 24-hour moving average

# Drop first 24 rows containing NaN due to lag computation
df_clean = df.dropna().copy()
print(f"\nUsable records after lag engineering: {len(df_clean)}")

# Define input features and target
feature_cols = [
    'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'IsWeekend',
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
    'PM2.5_lag1', 'PM2.5_lag2', 'PM2.5_lag24', 'PM2.5_rolling24'
]

X = df_clean[feature_cols]
y_aqi = df_clean['AQI']

# ==============================================================================
# Step 5: Train-Test Split
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_aqi, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ==============================================================================
# Step 6: Model Initialization and Training
# ==============================================================================
print("\nTraining Random Forest Regressor for AQI prediction...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save model for future deployment / API / Streamlit
model_file = Path(__file__).resolve().parent / "aqi_model.pkl"
joblib.dump(model, model_file)
print(f"Saved trained model to: {model_file.name}")

# ==============================================================================
# Step 7: Model Evaluation
# ==============================================================================
# Training Performance
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Testing Performance
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\n--- Model Regression Metrics (AQI Score) ---")
print(f"Training R²: {train_r2:.4f} | MAE: {train_mae:.2f}")
print(f"Testing  R²: {test_r2:.4f} | MAE: {test_mae:.2f} | RMSE: {test_rmse:.2f}")

# Category Classification Metrics
actual_cats = [get_aqi_category(a) for a in y_test]
pred_cats = [get_aqi_category(a) for a in y_test_pred]
cat_accuracy = accuracy_score(actual_cats, pred_cats)

print(f"\n--- AQI Health Category Classification Accuracy: {cat_accuracy * 100:.2f}% ---")
print("\nClassification Report:")
print(classification_report(actual_cats, pred_cats, zero_division=0))

# Feature Importances
print("\n--- Top Features Contributing to AQI Prediction ---")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.round(4).to_string())

# ==============================================================================
# Step 8: Sample Prediction & Health Advisory Simulation
# ==============================================================================
print("\n--- Real-Time Prediction Simulation ---")
sample_features = X_test.iloc[0:1]
actual_aqi = y_test.iloc[0]
predicted_aqi = model.predict(sample_features)[0]

actual_cat = get_aqi_category(actual_aqi)
predicted_cat = get_aqi_category(predicted_aqi)

print(f"Actual AQI Score:    {actual_aqi:.2f} ({actual_cat})")
print(f"Predicted AQI Score: {predicted_aqi:.2f} ({predicted_cat})")
print(f"Prediction Error:    {abs(actual_aqi - predicted_aqi):.2f} AQI points")
print(f"Health Advisory:     {get_health_advisory(predicted_cat)}")