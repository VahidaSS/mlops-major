"""
train.py
Trains Decision Tree model on Olivetti Faces dataset
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
print("📥 Loading dataset...")
data = fetch_olivetti_faces()
X = data.data
y = data.target

# Split dataset (70-30)
print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Initialize model (controlled for better generalization)
print("🤖 Training model...")
model = DecisionTreeClassifier(
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"✅ Training Accuracy: {train_acc:.4f}")
print(f"✅ Testing Accuracy: {test_acc:.4f}")

# Save model
model_path = "savedmodel.pth"
joblib.dump(model, model_path)

print(f"📦 Model saved at {model_path}")

# Save metrics (extra improvement)
with open("metrics.txt", "w") as f:
    f.write(f"Train Accuracy: {train_acc}\n")
    f.write(f"Test Accuracy: {test_acc}\n")

print("✅ Training pipeline completed successfully.")
