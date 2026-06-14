"""
test.py

Loads trained model and evaluates on Olivetti Faces test set
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix
)
import joblib
import os

print("=" * 60)
print("MODEL TESTING STARTED")
print("=" * 60)

# Load dataset

print("Loading dataset...")

data = fetch_olivetti_faces()

X = data.data
y = data.target

print(f"Dataset Shape: {X.shape}")
print(f"Classes       : {len(set(y))}")

# Same split used during training

print("\nCreating test split...")

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.30,
random_state=42,
stratify=y
)

# Load trained model

print("\nLoading trained model...")

if not os.path.exists("savedmodel.pth"):
    print("ERROR: savedmodel.pth not found")
    exit()

model = joblib.load("savedmodel.pth")

print("Model loaded successfully")

# Predictions

print("\nRunning predictions...")

y_pred = model.predict(X_test)

# Metrics

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Test Accuracy: {accuracy:.4f}")

# Classification Report

report = classification_report(
y_test,
y_pred,
zero_division=0
)

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

# Save metrics

with open("test_metrics.txt", "w") as f:
    f.write("Decision Tree Test Metrics\n")
    f.write("=" * 40 + "\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")

# Save report

with open("test_classification_report.txt", "w") as f:
    f.write(report)

# Save confusion matrix

with open("test_confusion_matrix.txt", "w") as f:
    f.write(str(cm))

print("\nArtifacts Generated:")
print("- test_metrics.txt")
print("- test_classification_report.txt")
print("- test_confusion_matrix.txt")

print("\nTesting completed successfully.")
print("=" * 60)
