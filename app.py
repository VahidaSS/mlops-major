"""
Advanced Flask Application for Olivetti Faces Prediction
"""

from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import os
import logging
import cv2

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# Flask App
# --------------------------------------------------

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "savedmodel.pth")

# --------------------------------------------------
# Face Detector
# --------------------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    logging.info("Loading trained model...")
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    logging.info("Model loaded successfully")

except Exception as e:
    logging.error(f"Startup Error: {e}")
    model = None

# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None
    })

# --------------------------------------------------
# Prediction Route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model not loaded"
            }), 500

        if "file" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "status": "error",
                "message": "Empty filename"
            }), 400

        if (
            not file.content_type or
            not file.content_type.startswith("image")
        ):
            return jsonify({
                "status": "error",
                "message": "Please upload a valid image"
            }), 200

        logging.info(f"Image received: {file.filename}")

        # Convert image
        image_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return jsonify({
                "status": "error",
                "message": "Unable to read image"
            }), 200

        # Face detection
        faces = face_detector.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            logging.warning("No face detected")
            return jsonify({
                "status": "error",
                "message": "No face detected in image"
            }), 200

        logging.info(f"Faces detected: {len(faces)}")

        # Take first detected face
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))

        img_array = (face.flatten().astype(np.float32) / 255.0)

        # Prediction
        prediction = int(model.predict([img_array])[0])
        confidence = None

        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba([img_array])
                confidence = round(float(np.max(probabilities)) * 100, 2)
            except Exception:
                pass

        logging.info(f"Prediction successful: {prediction}")

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "faces_detected": int(len(faces))
        })

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# --------------------------------------------------
# Run App
# --------------------------------------------------

if __name__ == "__main__":
    print(app.url_map)
    app.run(
        host="::",
        port=5000,
        debug=False
    )
