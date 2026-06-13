# Used Car Price Prediction Using Artificial Neural Networks (ANN)
# Predict the selling price of a used car based on vehicle specifications.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

# Load Dataset
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(7) Deep Learning/Car_Price_Predict.csv")
print("Dataset Shape:", df.shape)

# Convert categorical columns into numeric form
df['fuel'] = LabelEncoder().fit_transform(df['fuel'])
df['seller_type'] = LabelEncoder().fit_transform(df['seller_type'])
df['transmission'] = LabelEncoder().fit_transform(df['transmission'])
df['owner'] = LabelEncoder().fit_transform(df['owner'])
print(df.head())

# Features (Input) and Target (Output)
X = df.drop(['name', 'selling_price'], axis=1)
y = df['selling_price']

# Standardize the data
# ANN performs better when features are on a similar scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# Build ANN Model
model = Sequential([
    Input(shape=(X_train.shape[1],)),   # Input Layer
    Dense(64, activation='relu'),       # Hidden Layer 1
    Dense(32, activation='relu'),       # Hidden Layer 2
    Dense(16, activation='relu'),       # Hidden Layer 3
    Dense(1)])                            # Output Layer (Regression)

# Compile Model
# MSE is used because this is a regression problem
model.compile(optimizer='adam',loss='mean_squared_error')

# Train Model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=10, verbose=0)
y_pred = model.predict(X_test, verbose=0).flatten()

# Model Evaluation
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot Training and Validation Loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Actual vs Predicted Prices
plt.figure(figsize=(6,6))
sns.scatterplot( x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Used Car Prices")
plt.grid(True)
plt.show()

# Predict Price for a New Car
# Format: [year, km_driven, fuel, seller_type, transmission, owner]
new_data = np.array([[2017, 79000, 2, 1, 1, 1]])

# Apply the same scaling used during training
new_data_scaled = scaler.transform(new_data)
# Predict selling price
predicted_price = model.predict(new_data_scaled, verbose=0)
print("\nPredicted Selling Price: ₹{:.2f}".format(predicted_price[0][0]))