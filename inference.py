import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import time

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils import HandDetector, preprocess_image
from config import IMAGE_SIZE, MODEL_CHECKPOINT_DIR, CLASS_MAPPING

def load_sign_speak_model():
    """
    Load the trained Sign_Speak model.
    Returns the loaded model or None if loading fails.
    """
    try:
        # Try to find the latest model in the MODEL_DIR
        model_files = [f for f in os.listdir(MODEL_CHECKPOINT_DIR) if f.endswith('.h5')]
        if not model_files:
            print("[Error] No model files found in", MODEL_CHECKPOINT_DIR)
            return None
        
        # Use the most recent model file (or you can specify a particular one)
        model_path = os.path.join(MODEL_CHECKPOINT_DIR, sorted(model_files)[-1])
        print(f"[INFO] Loading model from: {model_path}")
        
        # Load the model
        model = load_model(model_path)
        print("[INFO] Model loaded successfully")
        return model
    
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return None

def get_prediction(model, image, class_mapping):
    # Ensure image is a 4D tensor (batch, height, width, channels)
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    # For models expecting a sequence dimension, add it if missing
    if len(image.shape) == 4 and model.input_shape[1] != image.shape[1]:
        image = np.expand_dims(image, axis=1)
    
    predictions = model.predict(image)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    return class_idx, confidence


def try_camera_connection(max_attempts=3):
    """
    Try to connect to the camera with multiple attempts.
    Returns VideoCapture object or None if connection fails.
    """
    available_cameras = []
    
    # Try to find available cameras
    for i in range(3):  # Try cameras 0, 1, 2
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Try DirectShow API on Windows
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("[Error] No working cameras found!")
        return None
    
    print(f"[INFO] Found working cameras at indices: {available_cameras}")
    
    # Try to connect to the first available camera
    selected_camera = available_cameras[0]
    for attempt in range(max_attempts):
        print(f"[INFO] Attempting to connect to camera {selected_camera} (attempt {attempt+1}/{max_attempts})")
        
        # Try with DirectShow backend on Windows
        cap = cv2.VideoCapture(selected_camera, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"[Warning] Failed to open camera {selected_camera} on attempt {attempt+1}")
            time.sleep(1)  # Wait a bit before retrying
            continue
        
        # Try to read a test frame
        ret, frame = cap.read()
        if not ret:
            print(f"[Warning] Could open camera {selected_camera} but couldn't read frames")
            cap.release()
            time.sleep(1)  # Wait a bit before retrying
            continue
            
        print(f"[INFO] Successfully connected to camera {selected_camera}")
        return cap
    
    return None

def run_inference():
    """
    Run real-time inference using webcam.
    """
    # Load the Sign_Speak model
    model = load_sign_speak_model()
    if model is None:
        print("[Error] Could not load model. Exiting.")
        return
    
    # Initialize webcam with more robust connection attempt
    cap = try_camera_connection()
    if cap is None:
        print("[Error] Could not establish a reliable camera connection. Exiting.")
        return
    
    # Set webcam properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create a named window and set it to fullscreen
    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Language Recognition', 1280, 720)  # Resize to larger dimensions
    
    # Initialize hand detector
    detector = HandDetector(
        static_image_mode=False,  # Dynamic mode for video
        max_num_hands=1,          # Only detect one hand
        min_detection_confidence=0.5
    )
    
    # Load class mapping
    class_mapping = None
    if 'CLASS_MAPPING' in globals() and CLASS_MAPPING:
        class_mapping = CLASS_MAPPING
    else:
        # Default class mappings if not defined in config
        alphabets = [chr(ord('A') + i) for i in range(26)]
        numbers = [str(i) for i in range(10)]
        class_mapping = alphabets + numbers
        print("[Warning] Using default class mapping:", class_mapping)
    
    print("[INFO] Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            # Try to recover from frame read failure
            print("[Warning] Failed to grab frame. Attempting to recover...")
            time.sleep(0.1)  # Short pause
            
            # Check if we've been failing for too long
            if frame_count == 0 and (time.time() - start_time) > 5:
                print("[Error] Could not read frames after 5 seconds. Exiting.")
                break
                
            continue  # Skip this iteration and try again
        
        # We got a frame, increment counter
        frame_count += 1
        
        # Mirror the frame horizontally for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Apply lighting adjustment (if needed)
        # Make a copy for display
        display_frame = frame.copy()
        
        # Frame dimensions
        height, width, _ = frame.shape
        
        # Extract hand region
        hand_roi, bbox, hand_detected = detector.extract_hand_roi(
            frame, 
            padding=30, 
            target_size=IMAGE_SIZE
        )
        
        # Display status and prediction
        status_text = "No hand detected"
        confidence_text = ""
        
        if hand_detected and hand_roi is not None:
            # Preprocess the hand ROI for the model
            processed_img = preprocess_image(
                hand_roi,
                target_size=IMAGE_SIZE,
                normalize=True,
                grayscale=False,
            )
            
            # Get prediction
            predicted_class, confidence = get_prediction(model, processed_img, class_mapping)
            
            # Update status
            status_text = f"Detected: {predicted_class}"
            confidence_text = f"Confidence: {confidence:.2f}"
            
            # Change bounding box color based on confidence
            if confidence < 0.5:
                color = (0, 0, 255)  # Red for low confidence
            else:
                color = (0, 255, 0)  # Green for high confidence
            
            # Draw hand bounding box
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
        
        # Add text to the frame
        cv2.putText(display_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if confidence_text:
            cv2.putText(display_frame, confidence_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show the processed frame
        cv2.imshow('Sign Language Recognition', display_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    run_inference()
