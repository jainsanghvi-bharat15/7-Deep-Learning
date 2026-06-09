# Air Pollution Level Estimation (PM2.5) using Artificial Neural Network (ANN)
# Predict PM2.5 concentration from weather conditions and time-related features.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings

# Import Libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping

# Load Dataset
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(7) Deep Learning/Air_Pollution.csv")

# Remove rows where target value (PM2.5) is missing
df = df[df["pm2.5"].notna()]

print("Check missing values:", df.isnull().sum())
print("Dataset:", df.head())
print("Dataset Shape:", df.shape)

# Create a datetime column using year, month, day and hour
# Helps in extracting useful time-based information
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df.set_index("datetime", inplace=True)  # Set datetime as index

# Extract hour and month from datetime
# Pollution often varies by time of day and season
df["hour"] = df.index.hour
df["month"] = df.index.month

# Select Input Features and Target Variable
FEATURES = [
    "DEWP",     # Dew Point
    "TEMP",     # Temperature
    "PRES",     # Pressure
    "Iws",      # Wind Speed
    "Is",       # Snow Hours
    "Ir",       # Rain Hours
    "hour",     # Hour of Day
    "month"]     # Month of Year
TARGET = "pm2.5"


X = df[FEATURES]    # Input Features
y = df[TARGET]      # Output Variable

# Split Dataset
# 80% Training Data  20% Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize Features
# ANN trains better when features have similar scales
scaler = StandardScaler()

# Learn scaling parameters and transform training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply same scaling to testing data
X_test_scaled = scaler.transform(X_test)

# Save scaler for future predictions
joblib.dump(scaler, "scaler.joblib")

# Build ANN Model
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)), # Input Layer
    Dense(64, activation='relu'),   # Hidden Layer 1
    Dense(32, activation='relu'),   # Hidden Layer 2
    Dense(1)])  # Output Layer/Predicts PM2.5 value
    
# Configure ANN
# Adam = Optimizer
# MSE = Loss Function for Regression
# MAE = Evaluation Metric
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae'])

# Stop training when validation loss stops improving
early_stop = EarlyStopping(
    patience=5,
    restore_best_weights=True)

# Train Model
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.1,  # 10% of training data used for validation
    epochs=50,
    batch_size=256,        # Train 256 records at a time
    callbacks=[early_stop],
    verbose=0)

# Predict PM2.5 values on test data
y_pred = model.predict(
    X_test_scaled,
    verbose=0).flatten()

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nModel Performance")
print(f"MAE  : {mae:.2f} µg/m³")
print(f"RMSE : {rmse:.2f} µg/m³")
print(f"R²   : {r2:.3f}")

# Plot Training and Validation Loss
plt.figure(figsize=(8,5))
plt.plot(
    history.history["loss"],
    label="Training Loss")

plt.plot(
    history.history["val_loss"],
    label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.show()

# Actual vs Predicted PM2.5
plt.figure(figsize=(6,6))
sns.scatterplot(
    x=y_test,
    y=y_pred,
    alpha=0.4)

# Perfect prediction reference line
plt.plot(
    [0, 600],
    [0, 600])
plt.xlabel("Actual PM2.5 (µg/m³)")
plt.ylabel("Predicted PM2.5 (µg/m³)")
plt.title("Actual vs Predicted PM2.5")
plt.grid(True)
plt.show()

# Save Trained Model
model.save("pm25_ann.h5")
# Save Feature Order
joblib.dump(FEATURES, "feature_order.joblib")

# Predict PM2.5 for New Weather Data
print("\n=== PM2.5 Estimator ===")
new_values = {}

# Take input for each feature
for feature in FEATURES:
    new_values[feature] = float(input(f"Enter {feature}: "))

# Convert user input into DataFrame
new_df = pd.DataFrame([new_values])[FEATURES]

# Apply same scaling used during training
new_scaled = scaler.transform(new_df)

# Predict PM2.5
predicted_pm25 = model.predict(new_scaled, verbose=0)[0][0]
print(
    f"\nEstimated PM2.5 Concentration: "
    f"{predicted_pm25:.2f} µg/m³")