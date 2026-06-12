"""
test.py
Loads trained model and evaluates on test data
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("📥 Loading dataset for testing...")
data = fetch_olivetti_faces()
X = data.data
y = data.target

# Same split (important for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Load model
print("📦 Loading saved model...")
model = joblib.load("savedmodel.pth")

# Predict
print("🔍 Running predictions...")
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Test Accuracy: {accuracy:.4f}")

print("✅ Testing completed successfully.")
