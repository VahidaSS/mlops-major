"""
test.py
Loads trained model and evaluates on test data
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

import os

print(" Loading dataset for testing...")
data = fetch_olivetti_faces()
X = data.data
y = data.target

# Same split 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Load model

print("Loading saved model...")
if not os.path.exists("artifacts/savedmodel.pth"):
    print(" Model file not found!")
    exit()

model = joblib.load("artifacts/savedmodel.pth")


# Predict
print(" Running predictions...")
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)


print(f"Test Accuracy: {accuracy:.4f}")

with open("test_metrics.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")

print(" Testing completed successfully.")