import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from model import ASLModel
from config import FINAL_MODEL_PATH, IMAGE_SIZE, CLASS_MAPPING
from utils import HandDetector, preprocess_image

def load_and_preprocess_image(image_path, detector):
    """Load and preprocess an image for model prediction."""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        # Detect hand and extract ROI
        roi, bbox, hand_detected = detector.extract_hand_roi(img, target_size=IMAGE_SIZE)
        
        # Preprocess the ROI
        roi_preprocessed = preprocess_image(roi, target_size=IMAGE_SIZE)
        
        # Add batch and sequence dimensions for LSTM model input (batch_size, sequence_length, height, width, channels)
        roi_preprocessed = np.expand_dims(np.expand_dims(roi_preprocessed, axis=0), axis=1)
        
        return roi_preprocessed, hand_detected
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, False

def main():
    # Create directory for test images if it doesn't exist
    test_images_dir = 'data/test_images'
    os.makedirs(test_images_dir, exist_ok=True)
    
    # Check if test images directory has files
    if not os.listdir(test_images_dir):
        print(f"Warning: No test images found in {test_images_dir}")
        print("Please add some test images to this directory and run the script again.")
        return
    
    # Check if model exists
    if not os.path.exists(FINAL_MODEL_PATH):
        print(f"Error: Model file not found at {FINAL_MODEL_PATH}")
        print("Please train the model first or check the path in config.py.")
        return
    
    # Load the trained model
    try:
        asl_model = ASLModel()
        model = asl_model.load_model(FINAL_MODEL_PATH)
        print(f"Successfully loaded model from {FINAL_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize hand detector
    detector = HandDetector(static_image_mode=True, min_detection_confidence=0.7)
    
    # Iterate through test images and make predictions
    print("\nProcessing test images:")
    successful_predictions = 0
    
    for image_name in os.listdir(test_images_dir):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Skip non-image files
            
        image_path = os.path.join(test_images_dir, image_name)
        
        # Load and preprocess the image
        processed_image, hand_detected = load_and_preprocess_image(image_path, detector)
        
        if processed_image is not None:
            try:
                # Predict the class
                class_idx, confidence = asl_model.predict(processed_image)
                predicted_class = CLASS_MAPPING[class_idx]
                
                # Print the result
                print(f"Image: {image_name} | Predicted Class: {predicted_class} | Confidence: {confidence:.2f} | Hand detected: {'Yes' if hand_detected else 'No'}")
                successful_predictions += 1
            except Exception as e:
                print(f"Error predicting {image_name}: {e}")
    
    if successful_predictions == 0:
        print("\nNo successful predictions. Please check your images and model.")
    else:
        print(f"\nSuccessfully processed {successful_predictions} images.")

if __name__ == "__main__":
    main()