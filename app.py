from flask import Flask, render_template, jsonify, request
import os
from config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, GUIDE_BOX_X, GUIDE_BOX_Y, 
    GUIDE_BOX_W, GUIDE_BOX_H, CLASS_MAPPING
)

# Change the Flask app initialization to specify custom template and static folders
app = Flask(__name__, 
            template_folder='frontend/templates',
            static_folder='frontend/static')

@app.route('/')
def index():
    """Render the main application page with config values injected"""
    return render_template('index.html', 
                          image_width=IMAGE_WIDTH,
                          image_height=IMAGE_HEIGHT,
                          guide_box_x=GUIDE_BOX_X,
                          guide_box_y=GUIDE_BOX_Y,
                          guide_box_w=GUIDE_BOX_W,
                          guide_box_h=GUIDE_BOX_H)

@app.route('/get_config')
def get_config():
    """Return configuration values to frontend"""
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
    """Return the class mapping"""
    return jsonify(CLASS_MAPPING)

@app.route('/dummy_predict', methods=['POST'])
def dummy_predict():
    """
    Dummy prediction endpoint for frontend testing
    Will be replaced with actual model prediction later
    """
    # Just return dummy data for now
    return jsonify({
        'predicted_class': 10,  # 'A'
        'confidence': 0.95,
        'hand_detected': True
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)