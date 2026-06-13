"""Classifying Clothing Images Using Artificial Neural Network (ANN)
We want create a model which teach a machine to look at a small grayscale image of a fashion item (like a T-shirt or sneaker) and guess what it is.
Built-in dataset in Keras (no need to download separately).
Each image is 28 × 28 pixels (black & white).
Each image shows one clothing item.
There are 10 categories (T-shirt/top, trouser, dress, etc.)"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import Fashion-MNIST dataset
# Provides 70,000 labeled clothing images for training and testing.
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Flatten  # Flatten = Converts image matrix into a single vector

# Import one-hot encoding utility
# Converts class labels into categorical format required for multi-class classification.
from tensorflow.keras.utils import to_categorical

# Load Fashion-MNIST dataset
# Returns training and testing images along with their labels.
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Normalize pixel values
"""Pixel values range from 0 to 255.
Scaling them between 0 and 1 improves ANN training speed and stability."""
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels into one-hot encoded vectors
"""Required because output layer has 10 neurons.
# Example: Label 3 becomes [0,0,0,1,0,0,0,0,0,0]"""
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)


# Create ANN model
# Sequential() arranges layers one after another.
model = Sequential([
    # Converts 28x28 image into 784 values.
    # ANN accepts only 1D input, not 2D images.
    Flatten(input_shape=(28, 28)),

    # First hidden layer : Learns basic image features and patterns.
    Dense(128, activation='relu'),

    # Second hidden layer : Learns deeper and more complex patterns.
    Dense(64, activation='relu'),

    # Output layer
    # 10 neurons because there are 10 clothing categories.
    # Softmax converts outputs into probabilities
    Dense(10, activation='softmax')
    ])

# Compile the model
"""optimizer='adam' : Automatically updates weights efficiently during training.
loss='categorical_crossentropy' : Calculates prediction error for multi-class classification.
metrics=['accuracy']: Tracks percentage of correct predictions."""
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Train the ANN
"""X_train -> Input images        y_train_cat -> Correct labels
epochs=10: Entire dataset is used 10 times for learning.
batch_size=128: Processes 128 images at a time.
Reduces memory usage and speeds up training."""

# validation_split=0.1 :  Uses 10% of training data to check model performance during training.
history = model.fit(
    X_train,
    y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1)

# Evaluate model performance on unseen test data.
# Returns final loss and accuracy.
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# Predict class probabilities for each test image.
y_pred_probs = model.predict(X_test)

# Select class having highest probability.
# Converts probabilities into actual class labels.
y_pred = np.argmax(y_pred_probs, axis=1)


# Generate Classification Report
"""Precision: Out of predicted class samples, how many were correct.
# Recall: Out of actual class samples, how many were found.
# F1-Score: Balance between Precision and Recall."""
print(classification_report(y_test, y_pred))


# Create Confusion Matrix
cm = confusion_matrix(y_test, y_pred)   # Shows actual vs predicted class counts.

# Display confusion matrix as heatmap.
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Class names corresponding to labels 0-9.
class_names = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Display sample predictions.
# Helps visually verify whether the model is predicting correctly.
plt.figure(figsize=(12,4))
for i in range(10):
    plt.subplot(2, 5, i+1)  # Create subplot for each image.
    plt.imshow(X_test[i], cmap='gray')  # Display grayscale image.

    # Show actual and predicted class.
    plt.title(
        f"Actual: {class_names[y_test[i]]}\n"
        f"Pred: {class_names[y_pred[i]]}")
    plt.axis('off') # Hide axis values for cleaner display.
plt.tight_layout()
plt.show()