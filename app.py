"""
app.py
Flask application for serving ML model predictions
"""

from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import joblib
import os
import logging

# Logging configuration (PRO LEVEL)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/savedmodel.pth")


# Load model safely
def load_model():
    if not os.path.exists(MODEL_PATH):
        logging.error("Model file not found!")
        raise FileNotFoundError("Model not found. Train model first.")
    logging.info("Loading model...")
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    logging.error(f"Startup error: {e}")
    model = None



# Home route
@app.route("/")
def home():
    return render_template("index.html")



# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # MODEL CHECK
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
     
        # ADD FILE TYPE
        if not file.content_type or not file.content_type.startswith("image"):
            return jsonify({"error": "Invalid file type. Upload an image."}), 400

        if file.filename == "":
            return jsonify({"error": "Empty file name"}), 400

        logging.info("Processing uploaded image")

        # Image preprocessing
        img = Image.open(file).convert("L").resize((64, 64))
        img_array = np.array(img).flatten() / 255.0

        logging.info("Running prediction")

        prediction = model.predict([img_array])[0]
        logging.info(f"Prediction successful: {prediction}")
        return jsonify({
            "status": "success",
            "prediction": int(prediction)
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# Health check route 
@app.route("/health")
def health():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)