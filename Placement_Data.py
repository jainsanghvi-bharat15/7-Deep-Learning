# Student Placement Prediction Using ANN on Academic and Skill Data
# Predict whether a student will be Placed or Not Placed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

# Load Dataset
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(7) Deep Learning/Placement_Data.csv")
print("Missing Values: ",df.isnull().sum())

# Encode Categorical Columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
print("\nEncoded Dataset:",df.head())

# Features (Input) and Target (Output)
X = df.drop('status', axis=1)
y = df['status']

# Feature Scaling
# ANN performs better when data is standardized
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Dataset
# 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.2, random_state=42)
print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Build ANN Model
model = Sequential([Input(shape=(X_train.shape[1],)),   # Input Layer
                    Dense(16, activation='relu'),       # Hidden Layer 1
                    Dense(8, activation='relu'),        # Hidden Layer 2
                    Dense(1, activation='sigmoid')])    # Output Layer (Binary Classification)
# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Train Model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    verbose=0
)

# Plot Loss Curve
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy Curve
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

y_pred = model.predict(X_test, verbose=0)   # Predictions on Test Data
y_pred_classes = (y_pred > 0.5).astype(int) # Convert probabilities to binary classes (0 or 1)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:\n")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True,fmt='d',cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Predict Placement for New Student
print("\n----- New Student Prediction -----")
def get_user_input():
    input_data = {}
    for col in X.columns:
        if col in label_encoders:
            print(f"\n{col} options:")
            print(list(label_encoders[col].classes_))
            value = input(f"Enter {col}: ")
            input_data[col] = label_encoders[col].transform([value])[0]
        else:
            value = float(input(f"Enter {col}: "))
            input_data[col] = value
    return pd.DataFrame([input_data])
user_df = get_user_input()
user_scaled = scaler.transform(user_df)
prediction = model.predict(user_scaled, verbose=0)

# Convert Probability to Class
result = "Placed" if prediction[0][0] > 0.5 else "Not Placed"
print(f"\nPredicted Placement Status: {result}")
print(f"Placement Probability: {prediction[0][0]:.2f}")