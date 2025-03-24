from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import cv2
import numpy as np
import base64
import tensorflow as tf
import uuid
import logging
from datetime import datetime

from config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, GUIDE_BOX_X, GUIDE_BOX_Y, 
    GUIDE_BOX_W, GUIDE_BOX_H, CLASS_MAPPING, IMAGE_SIZE,
    MODEL_CHECKPOINT_DIR, FINAL_MODEL_PATH
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='frontend/templates',
            static_folder='frontend/static')

# Path for storing temporary images
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'static', 'temp_images')
os.makedirs(TEMP_DIR, exist_ok=True)
app.logger.info(f"Temp directory: {TEMP_DIR}")

# Load the model
try:
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    print(f"Model loaded from {FINAL_MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def save_image(image, prefix="img"):
    """Save image to temp directory and return the filename."""
    if image is None:
        return None
    try:
        filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(TEMP_DIR, filename)
        success = cv2.imwrite(filepath, image)
        
        if success and os.path.exists(filepath):
            app.logger.info(f"Image saved to {filepath}")
            return filename
        else:
            app.logger.error(f"Failed to save image to {filepath}")
            return None
    except Exception as e:
        app.logger.error(f"Error saving image: {str(e)}")
        return None

def detect_and_crop_hand(image_data):
    """Detect and crop the hand from the image."""
    try:
        # Convert to numpy array from base64
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            encoded_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        else:
            nparr = np.frombuffer(image_data, np.uint8)
        
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, img, None
        
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        
        hand_crop = img[y:y+h, x:x+w]
        
        # We draw a bounding box for reference, but no side-by-side comparison
        vis_img = img.copy()
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return True, vis_img, hand_crop
    
    except Exception as e:
        app.logger.error(f"Error in hand detection: {e}")
        if 'img' in locals():
            return False, img, None
        return False, None, None

def preprocess_image(image, target_size=IMAGE_SIZE):
    """Preprocess image for model input."""
    try:
        img = cv2.resize(image, target_size)
        img = img.astype(np.float32) / 255.0
        # Add batch and sequence dimensions
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
        return img
    except Exception as e:
        app.logger.error(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html', 
                          image_width=IMAGE_WIDTH,
                          image_height=IMAGE_HEIGHT,
                          guide_box_x=GUIDE_BOX_X,
                          guide_box_y=GUIDE_BOX_Y,
                          guide_box_w=GUIDE_BOX_W,
                          guide_box_h=GUIDE_BOX_H)

@app.route('/get_config')
def get_config():
    """Return configuration values to frontend."""
    return jsonify({
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'guide_box_x': GUIDE_BOX_X,
        'guide_box_y': GUIDE_BOX_Y,
        'guide_box_w': GUIDE_BOX_W,
        'guide_box_h': GUIDE_BOX_H
    })

@app.route('/get_class_mapping')
def get_class_mapping():
    """Return the class mapping."""
    return jsonify(CLASS_MAPPING)

@app.route('/predict', methods=['POST'])
def predict():
    """Real prediction endpoint using the trained model."""
    app.logger.info("Prediction request received")
    
    if not request.json or 'image' not in request.json:
        app.logger.error("No image provided in request")
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_data = request.json['image']
        hand_detected, input_img, cropped_hand = detect_and_crop_hand(image_data)
        
        if not hand_detected or input_img is None:
            app.logger.warning("No hand detected in the image")
            return jsonify({
                'hand_detected': False,
                'error': 'Failed to process image'
            })
        
        # Save images
        input_filename = save_image(input_img, "input")
        cropped_filename = save_image(cropped_hand, "cropped")
        
        # Create image URLs
        input_url = f"/static/temp_images/{input_filename}" if input_filename else None
        cropped_url = f"/static/temp_images/{cropped_filename}" if cropped_filename else None
        
        # Make prediction if model is available
        if model is not None and cropped_hand is not None:
            processed_img = preprocess_image(cropped_hand)
            if processed_img is not None:
                predictions = model.predict(processed_img)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])
                
                app.logger.info(f"Prediction: class={predicted_class}, confidence={confidence:.4f}")
                
                # Map class index to label
                class_label = None
                for label, idx in CLASS_MAPPING.items():
                    if idx == predicted_class:
                        class_label = label
                        break
                
                return jsonify({
                    'predicted_class': int(predicted_class),
                    'class_label': class_label or f"Class {predicted_class}",
                    'confidence': confidence,
                    'hand_detected': True,
                    'input_image_url': input_url,
                    'processed_image_url': cropped_url
                })
        
        app.logger.warning("No prediction made - model not available or processing failed")
        return jsonify({
            'hand_detected': True,
            'input_image_url': input_url,
            'processed_image_url': cropped_url,
            'error': 'No prediction available'
        })
        
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': f"Prediction error: {str(e)}"
        }), 500

@app.route('/static/temp_images/<filename>')
def serve_temp_image(filename):
    """Serve images from the temp directory."""
    app.logger.info(f"Attempt to access image: {filename}")
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        app.logger.error(f"File not found: {file_path}")
        return "File not found", 404
    
    try:
        return send_from_directory(TEMP_DIR, filename)
    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {str(e)}")
        return f"Error serving file: {str(e)}", 500

@app.errorhandler(404)
def not_found(e):
    app.logger.warning(f"404 error: {request.url}")
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"500 error: {str(e)}")
    return jsonify({"error": "Server error"}), 500

def clean_temp_files():
    """Remove temporary files older than 1 hour."""
    try:
        current_time = datetime.now()
        count = 0
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if not os.path.isfile(file_path):
                continue
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if (current_time - file_time).total_seconds() > 3600:
                os.remove(file_path)
                count += 1
        if count > 0:
            app.logger.info(f"Cleaned {count} old temporary files")
    except Exception as e:
        app.logger.error(f"Error cleaning temporary files: {e}")

if __name__ == '__main__':
    os.makedirs(TEMP_DIR, exist_ok=True)
    app.logger.info(f"Ensuring temp directory exists: {TEMP_DIR}")
    clean_temp_files()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
