"""Heart Disease Predictor using ANN (Keras)
We want to predict whether a person has heart disease or not based on multiple health parameters.
This is a binary classification problem (0 = No disease, 1 = Disease).
| Feature      | Meaning                     | Example |
| ------------ | --------------------------- | ------- |
| Age          | Person's age                | 45      |
| RestingBP    | Resting Blood Pressure      | 130     |
| Cholesterol  | Cholesterol level           | 250     |
| MaxHR        | Maximum heart rate achieved | 140     |
| Oldpeak      | ST depression (ECG data)    | 2.5     |
| HeartDisease | Target (0=no, 1=yes)        | 1       | """

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Hide TensorFlow warnings

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

# Load Dataset
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(7) Deep Learning/Heart.csv")
print(df.isnull().sum())    # Check for missing values
X = df.drop('target', axis=1) # Features (Input)
y = df['target']              # Target (Output)
print("Dataset Shape:", df.shape)

# Scale features between 0 and 1
# Neural Networks perform better on normalized data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
# 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42)
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# Build ANN Model
model = Sequential([
    Input(shape=(X.shape[1],)), # Input Layer (13 features)
    Dense(32, activation='relu'), # Hidden Layer 1
    Dense(16, activation='relu'), # Hidden Layer 2
    Dense(8, activation='relu'), # Hidden Layer 3

    # Output Layer
    # Sigmoid returns probability between 0 and 1
    Dense(1, activation='sigmoid') ])

# Compile Model
model.compile(
    optimizer='adam',                 # Learning algorithm
    loss='binary_crossentropy',       # Binary classification loss
    metrics=['accuracy'] )            # Track accuracy

# Train Model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    verbose=0)

# Plot Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Predictions on Test Data
y_pred = model.predict(X_test, verbose=0)

# Convert probabilities into 0 or 1
y_pred_classes = (y_pred > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Visualize Confusion Matrix
plt.figure(figsize=(5,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Predict for New Patient
# age, sex, cp, trestbps, chol, fbs,
# restecg, thalach, exang, oldpeak,
# slope, ca, thal

new_data = np.array([[48, 1, 1, 90, 123, 1, 0, 162, 0, 1.1, 2, 1, 1]])

# Apply same scaling used during training
new_data_scaled = scaler.transform(new_data)

# Predict probability
prediction = model.predict(new_data_scaled, verbose=0)

# Convert probability to class
result = (prediction > 0.5).astype(int)
print("\nHeart Disease Present? :", result[0][0])

if result[0][0] == 1:
    print("Prediction: Heart Disease Detected")
else:
    print("Prediction: No Heart Disease Detected")