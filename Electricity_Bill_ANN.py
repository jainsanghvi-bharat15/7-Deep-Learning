import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

# Sample Dataset
data = {
    'Units_Consumed': [150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500],
    'People': [2, 3, 4, 5, 2, 3, 4, 5, 2, 3],
    'AC_Used': [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
    'Geyser_Used': [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    'Solar_Panel': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    'Bill_Amount': [500, 1200, 1800, 2200, 3000, 3500, 4000, 4500, 5000, 5500]}
df = pd.DataFrame(data)

X = df.drop('Bill_Amount', axis=1)  # Features (Input)
y = df['Bill_Amount']               # Target (Output)

# Normalize features between 0 and 1
# Helps ANN train faster and more accurately
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
# 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Build ANN Model
model = Sequential([
    Input(shape=(X.shape[1],)),     # Input layer 5 input features
    Dense(10, activation='relu'),   # Hidden Layer 1
    Dense(5, activation='relu'),    # Hidden Layer 2
    Dense(1)    # Output Layer (Predicted Bill Amount)
])

# Compile model
# Adam -> Optimizer
# MSE -> Loss function for regression
model.compile(
    optimizer='adam',
    loss='mean_squared_error')

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    verbose=0)

# Evaluate model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss:.2f}")

# Predict Bill
# Example: 1000 Units, 3 People, AC=Yes, Geyser=Yes, Solar=No
new_input = np.array([[1000, 3, 1, 1, 0]])

# Apply same scaling used during training
new_input_scaled = scaler.transform(new_input)

# Predict bill amount
predicted_bill = model.predict(new_input_scaled, verbose=0)

print(f"Predicted Bill Amount: ₹{predicted_bill[0][0]:.2f}")

# Plot Training vs Validation Loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()