import os
import numpy as np
import cv2
import tensorflow as tf
from utils import preprocess_image
from config import IMAGE_SIZE, MODEL_CHECKPOINT_DIR, LABEL_MAPPING

# Function to load the very_lightweight_model
def load_very_lightweight_model():
    """Loads the 'very_lightweight_model'."""
    model_path = os.path.join(MODEL_CHECKPOINT_DIR, 'very_lightweight_model.h5')
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model 'very_lightweight_model' not found!")
        return None

# Test function for single image
def predict_image(img_path):
    """Predict the class of a given image using the very lightweight model."""
    try:
        # Load the model
        model = load_very_lightweight_model()
        if model is None:
            print("Error: Model not found!")
            return
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to read image {img_path}")
            return

        # Preprocess image
        processed_img = preprocess_image(img)
        
        # Reshape the image for LSTM model input (add sequence dimension)
        processed_img = np.expand_dims(processed_img, axis=1)  # Shape (1, 128, 128, 3)
        
        # Predict using the model
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        # Map the predicted class index to its corresponding label
        predicted_label = LABEL_MAPPING.get(predicted_class, "Unknown")

        # Print prediction details
        print(f"Image: {img_path}")
        print(f"Predicted class: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")

    except Exception as e:
        print(f"Error predicting image: {str(e)}")

# Test function for all images in the test folder
def test_images_in_folder(test_folder):
    """Test all images in the given folder and display results"""
    images = os.listdir(test_folder)
    
    if not images:
        print("No images found in the test folder.")
        return
    
    for img_file in images:
        if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(test_folder, img_file)
            predict_image(img_path)


if __name__ == "__main__":
    # Specify your test folder path
    test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_images')

    # Run the test on all images in the folder
    test_images_in_folder(test_folder)
