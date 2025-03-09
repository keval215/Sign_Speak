# app.py (Step 6 - Minimal predict endpoint)
import os
import base64
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from config import IMAGE_WIDTH, IMAGE_HEIGHT
from utils import detect_hand_mediapipe, fallback_guide_box, prepare_roi_for_model

app = Flask(__name__)

MODEL_PATH = os.path.join("model_checkpoint", "final_model.h5")
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

@app.route("/")
def index():
    return "Hello, World! Flask is running with a minimal predict endpoint!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image data"}), 400

    image_data = data["image"]
    header, encoded = image_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Try to detect hand using MediaPipe
    cropped, hand_detected = detect_hand_mediapipe(image)
    
    if hand_detected and cropped is not None:
        roi = cropped
        used_fallback = False
    else:
        roi = fallback_guide_box(image)
        used_fallback = True
    
    roi_prepared = prepare_roi_for_model(roi, target_size=(128, 128))
    preds = model.predict(roi_prepared)
    pred_class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    
    response = {
        "hand_detected": hand_detected,
        "used_fallback": used_fallback,
        "predicted_class": pred_class_idx,
        "confidence": round(confidence, 2)
    }
    return jsonify(response)

if __name__ == "__main__":
    print("Starting Flask server with /predict endpoint...")
    app.run(debug=True)
