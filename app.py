"""
Flask app for face classification
"""

from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("savedmodel.pth")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        if file.filename == "":
            return "⚠️ No file uploaded"

        # Process image
        img = Image.open(file).convert("L").resize((64, 64))
        img = np.array(img).flatten() / 255.0

        prediction = model.predict([img])[0]

        return f"✅ Predicted Person ID: {prediction}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)