import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import pickle
import os

os.makedirs("model", exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/weather.csv")
df = df.dropna()

# =========================
# FEATURE ENGINEERING
# =========================
df["Temp_Diff"] = df["Temp_C"] - df["Dew Point Temp_C"]
df["Wind_Visibility_Ratio"] = df["Wind Speed_km/h"] / (df["Visibility_km"] + 1)
df["Pressure_Normalized"] = df["Press_kPa"] / 100

# =========================
# FEATURES
# =========================
X = df[[
    "Temp_C",
    "Dew Point Temp_C",
    "Rel Hum_%",
    "Wind Speed_km/h",
    "Visibility_km",
    "Press_kPa",
    "Temp_Diff",
    "Wind_Visibility_Ratio",
    "Pressure_Normalized"
]]

# =========================
# TEMPERATURE MODEL
# =========================
y_temp = df["Temp_C"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_temp, test_size=0.2, random_state=42
)

temp_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

temp_model.fit(X_train, y_train)

# Evaluation
pred = temp_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))

# Cross Validation
scores = cross_val_score(temp_model, X, y_temp, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-scores).mean()

# Save model
pickle.dump(temp_model, open("model/temp_model.pkl", "wb"))

with open("model/rmse.txt", "w") as f:
    f.write(f"RMSE: {round(rmse,2)} | CV: {round(cv_rmse,2)}")

# =========================
# WEATHER MODEL
# =========================
y_weather = df["Weather"]

le = LabelEncoder()
y_encoded = le.fit_transform(y_weather)

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

weather_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

weather_model.fit(X_train_w, y_train_w)

# Accuracy
weather_pred = weather_model.predict(X_test_w)
accuracy = accuracy_score(y_test_w, weather_pred)

# Save
pickle.dump(weather_model, open("model/weather_model.pkl", "wb"))
pickle.dump(le, open("model/label_encoder.pkl", "wb"))

print("✅ Training Done")
print("RMSE:", rmse)
print("CV RMSE:", cv_rmse)
print("Weather Accuracy:", accuracy)