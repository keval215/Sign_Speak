import cv2
import numpy as np
import mediapipe as mp
from gtts import gTTS
import io
import os
import math
import base64

# 1. Import your config
try:
    from config import (
        GUIDE_BOX_X, GUIDE_BOX_Y, GUIDE_BOX_W, GUIDE_BOX_H,
        MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
    )
except ImportError:
    # Fallback defaults if config is missing
    GUIDE_BOX_X, GUIDE_BOX_Y = 220, 140
    GUIDE_BOX_W, GUIDE_BOX_H = 200, 200
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.6


class HandDetector:
    def __init__(self, 
                 static_image_mode=True, 
                 max_num_hands=1, 
                 min_detection_confidence=MIN_DETECTION_CONFIDENCE, 
                 min_tracking_confidence=MIN_TRACKING_CONFIDENCE):
        """Initialize MediaPipe hand detection"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=True):
        """Detect hands in an image"""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        # Draw landmarks if hands detected and draw=True
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return img, self.results
    
    def extract_hand_roi(self, img, padding=30, target_size=(128, 128)):
        """
        Extract hand region with padding and resize to target size.
        If no hand is detected by MediaPipe, fallback to the guide box region.
        """
        h, w = img.shape[:2]
        
        # Convert to RGB and process with MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            # Take the first hand
            hand = self.results.multi_hand_landmarks[0]
            
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for landmark in hand.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            roi = img[y_min:y_max, x_min:x_max]
            
            if roi.size != 0:
                roi_resized = cv2.resize(roi, target_size)
                return roi_resized, (x_min, y_min, x_max, y_max), True
        
        # If we reach here, no hand was detected; fallback to guide box
        return self.extract_guide_box_roi(img, target_size)
    
    def extract_guide_box_roi(self, img, target_size=(128, 128)):
        """
        Extract ROI from a fixed guide box defined in config (GUIDE_BOX_X, GUIDE_BOX_Y, etc.).
        If that fails, returns a resized version of the entire image as a last resort.
        """
        h, w = img.shape[:2]
        
        # Coordinates for the guide box
        x1 = GUIDE_BOX_X
        y1 = GUIDE_BOX_Y
        x2 = x1 + GUIDE_BOX_W
        y2 = y1 + GUIDE_BOX_H
        
        # Ensure within image bounds
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > w: x2 = w
        if y2 > h: y2 = h
        
        roi = img[y1:y2, x1:x2]
        
        if roi.size != 0:
            roi_resized = cv2.resize(roi, target_size)
            return roi_resized, (x1, y1, x2, y2), False
        
        # Last resort: resize the whole image if guide box is invalid
        fallback_img = cv2.resize(img, target_size)
        return fallback_img, (0, 0, w, h), False
    
    def visualize_detection(self, img, bbox=None, guide_box=True):
        """
        Draw hand landmarks, bounding box, and optional guide box on the image
        """
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        
        # Draw guide box if requested
        if guide_box:
            # Use config-based box
            guide_x1 = GUIDE_BOX_X
            guide_y1 = GUIDE_BOX_Y
            guide_x2 = GUIDE_BOX_X + GUIDE_BOX_W
            guide_y2 = GUIDE_BOX_Y + GUIDE_BOX_H
            
            cv2.rectangle(
                img_copy,
                (guide_x1, guide_y1),
                (guide_x2, guide_y2),
                (0, 255, 0),  # Green for guide box
                2
            )
            cv2.putText(
                img_copy, 
                "Place hand here", 
                (guide_x1, guide_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
        
        # Process with MediaPipe
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img_copy,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                padding = 20
                
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Draw bounding box
                cv2.rectangle(
                    img_copy,
                    (x_min, y_min),
                    (x_max, y_max),
                    (255, 0, 0),  # Blue
                    2
                )
                cv2.putText(
                    img_copy, 
                    "Hand detected", 
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 0, 0), 
                    2
                )
        
        # Draw custom bounding box if provided
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(
                img_copy,
                (x_min, y_min),
                (x_max, y_max),
                (0, 0, 255),  # Red
                2
            )
        
        return img_copy
def preprocess_image(img, target_size=(128, 128), normalize=True, grayscale=False, add_dimensions=True):
    """Preprocess image for model input"""
    
    # Resize the image to target size
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size)
    
    # Convert to grayscale if required
    if grayscale and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)  # Add the channel dimension
    
    # Normalize pixel values to [0, 1]
    if normalize:
        img = img.astype(np.float32) / 255.0
    
    # Add batch dimension (i.e., (1, height, width, channels))
    if add_dimensions:
        img = np.expand_dims(img, axis=0)
    
    return img


def text_to_speech(text):
    """Convert text to speech audio data"""
    tts = gTTS(text=text, lang='en', slow=False)
    
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    audio_data = base64.b64encode(fp.read()).decode('utf-8')
    return audio_data

def create_directory_structure(base_dir):
    """Create directory structure if it doesn't exist"""
    os.makedirs(base_dir, exist_ok=True)
    
    alphabets_dir = os.path.join(base_dir, 'ALPHABETS')
    numbers_dir = os.path.join(base_dir, 'NUMBERS')
    
    os.makedirs(alphabets_dir, exist_ok=True)
    os.makedirs(numbers_dir, exist_ok=True)
    
    for i in range(10):
        os.makedirs(os.path.join(numbers_dir, str(i)), exist_ok=True)
    
    for i in range(26):
        letter = chr(65 + i)  # A-Z
        os.makedirs(os.path.join(alphabets_dir, letter), exist_ok=True)
        
    return True
