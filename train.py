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
print("Loading dataset...")
data = fetch_olivetti_faces()
X = data.data
y = data.target

# Split dataset (70-30)
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Initialize model
print("Training model...")
model = DecisionTreeClassifier(
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

# Save model
os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/savedmodel.pth"
joblib.dump(model, model_path)

print(f"Model saved at {model_path}")

# Save metrics
with open("metrics.txt", "w") as f:
    f.write(f"Train Accuracy: {train_acc}\n")