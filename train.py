"""
train.py
Enhanced Decision Tree training for Olivetti Faces dataset
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix
)
import joblib
import os

print("=" * 60)
print("Loading Olivetti Faces Dataset...")
print("=" * 60)

# Load dataset

data = fetch_olivetti_faces()
X = data.data
y = data.target

print(f"Dataset Shape: {X.shape}")
print(f"Number of Classes: {len(set(y))}")

# Train/Test Split

print("\nSplitting dataset (70% Train / 30% Test)...")

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.30,
random_state=42,
stratify=y
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")

# Decision Tree Model

print("\nTraining Decision Tree Classifier...")

model = DecisionTreeClassifier(
criterion="entropy",
splitter="best",
max_depth=30,
min_samples_split=2,
min_samples_leaf=1,
random_state=42
)

model.fit(X_train, y_train)

# Predictions

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Accuracy

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Training Accuracy : {train_acc:.4f}")
print(f"Testing Accuracy  : {test_acc:.4f}")

# Classification Report

report = classification_report(
y_test,
test_pred,
zero_division=0
)

# Confusion Matrix

cm = confusion_matrix(y_test, test_pred)

# Create artifacts directory

os.makedirs("artifacts", exist_ok=True)

# Save model

model_path = "savedmodel.pth"
joblib.dump(model, model_path)

print(f"\nModel saved successfully at: {model_path}")

# Save metrics

with open("metrics.txt", "w") as f:
    f.write("Decision Tree Classifier Metrics\n")
    f.write("=" * 40 + "\n")
    f.write(f"Training Accuracy: {train_acc:.4f}\n")
    f.write(f"Testing Accuracy : {test_acc:.4f}\n")

# Save classification report

with open("classification_report.txt", "w") as f:
        f.write(report)

# Save confusion matrix

with open("confusion_matrix.txt", "w") as f:
    f.write(str(cm))

print("Metrics saved to metrics.txt")
print("Classification report saved")
print("Confusion matrix saved")

print("\nTraining completed successfully.")
print("=" * 60)