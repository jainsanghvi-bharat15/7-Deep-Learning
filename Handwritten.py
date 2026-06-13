# Handwritten Digit Recognition with ANN
"""To build a ML model using an ANN that can accurately classify handwritten digits (0–9) from grayscale images of size 28×28 pixels.
Dataset : MNIST  (60 000 train images / 10 000 test images)
Goal    : Classify digits 0–9 from 28×28 grayscale images
Model   : Simple feed-forward neural network (Dense layers)"""
# 1. Imports
import numpy as np                      
import matplotlib.pyplot as plt         
from tensorflow.keras.datasets import mnist      # Loads the built-in MNIST handwritten digit dataset
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Flatten, Dense, Dropout  
from tensorflow.keras.utils import to_categorical 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# 2. Load the MNIST dataset
"""mnist.load_data() downloads and loads:
    60,000 training images
    10,000 testing images
    Each image is 28x28 pixels and represents a handwritten digit (0-9)"""
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Training Images Shape :", X_train.shape)
print("Testing Images Shape  :", X_test.shape)


# 3. Normalize pixel values
""" Original pixel values range from 0 to 255
Neural networks learn faster when values are scaled between 0 and 1
Dividing by 255 converts:
    0   -> 0.0
    255 -> 1.0"""
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0


# 4. Convert labels to One-Hot Encoding
""" ANN output layer contains 10 neurons (one for each digit)
Therefore labels must also be represented using 10 positions
Example: Digit 5 becomes: [0 0 0 0 0 1 0 0 0 0]""" 
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)


# 5. Build ANN Model
model = Sequential([
    # Flatten Layer
    # Converts each 28x28 image into a single row of 784 pixels
    # ANN Dense layers accept only 1D input vectors 
    Flatten(input_shape=(28, 28)),

    # First Hidden Layer : 256 neurons learn important image patterns such as curves, edges and shapes
    Dense(256, activation="relu"),

    # Dropout Layer
    # Randomly turns off 30% neurons during training
    # Helps reduce overfitting and improves generalization
    Dropout(0.3),

    # Second Hidden Layer
    # Learns higher-level digit patterns from previous layer output
    Dense(128, activation="relu"),
    # Another Dropout layer for regularization
    Dropout(0.3),

    # Output Layer
    # 10 neurons because digits range from 0 to 9
    # Softmax converts outputs into probabilities
    Dense(num_classes, activation="softmax")])

# 6. Compile the Model
"""optimizer="adam" : Automatically updates weights using adaptive learning rates
loss="categorical_crossentropy" : Measures prediction error for multi-class classification problems
metrics=["accuracy"] : Tracks percentage of correctly classified digits"""
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])
print(model.summary())

# 7. Train the Model
""" epochs=15 : Entire training dataset is passed through the model 15 times
    batch_size=128 : Instead of processing all images at once, model trains on groups of 128 images
    validation_split=0.1 : Uses 10% of training data for validation"""
history = model.fit(
    X_train,
    y_train_cat,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1)

# 8. Plot Training and Validation Accuracy
# Shows how model accuracy improves over epochs
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# 9. Plot Training and Validation Loss
"""Loss indicates prediction error
Lower loss means better model performance"""

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# 10. Evaluate Model on Test Data
# Uses completely unseen images
# Gives final accuracy after training
test_loss, test_acc = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# 11. Generate Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)    # np.argmax(): Selects the index having highest probability. That index becomes the predicted digit

# 12. Classification Report
"""# Precision:Out of predicted digit X, how many were actually X
# Recall: Out of actual digit X, how many were correctly found
# F1-Score: Balance between Precision and Recall"""
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 13. Confusion Matrix
# Shows where predictions are correct or incorrect
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues")
plt.title("MNIST Confusion Matrix")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.tight_layout()
plt.show()

# 14. Display Random Sample Predictions
# Shows actual digit and model prediction side by side
plt.figure(figsize=(10,4))
for i in range(10):
    # Random test image index
    idx = np.random.randint(0, len(X_test))
    img = X_test[idx]
    true_digit = y_test[idx]
    pred_digit = y_pred[idx]

    # Green title if prediction is correct
    # Red title if prediction is wrong
    color = "green" if true_digit == pred_digit else "red"
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(
        f"T:{true_digit}  P:{pred_digit}",
        color=color,
        fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.show()