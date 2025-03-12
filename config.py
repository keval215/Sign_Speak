import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Dataset paths
CUSTOM_RAW_DIR = os.path.join(DATA_DIR, 'custom_raw')
CUSTOM_PROCESSED_DIR = os.path.join(DATA_DIR, 'custom_processed')
KAGGLE_DATASET_DIR = os.path.join(DATA_DIR, 'kaggle_dataset')

# Model checkpoint path
MODEL_CHECKPOINT_DIR = os.path.join(BASE_DIR, 'model_checkpoint')
FINAL_MODEL_PATH = os.path.join(MODEL_CHECKPOINT_DIR, 'final_model.h5')

# Create directories if they don't exist
os.makedirs(CUSTOM_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

# Image preprocessing settings
IMAGE_SIZE = (128, 128)
GRAYSCALE = False  # Use color images for better features

# Guide Box Coordinates
# For example, a 200x200 box at (220, 140) in a 640x480 frame
GUIDE_BOX_X = 170
GUIDE_BOX_Y = 90 
GUIDE_BOX_W = 300
GUIDE_BOX_H = 300

# Image Preprocessing
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# MediaPipe Confidence
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Model parameters
NUM_CLASSES = 36  # 26 letters (A-Z) + 10 digits (0-9)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
VALIDATION_SPLIT = 0.2

# LSTM parameters
SEQUENCE_LENGTH = 1  # For single images (not video)

# Map class indices to labels
# 0-9: Digits, 10-35: Letters A-Z
CLASS_MAPPING = {}
# Add digits 0-9
for i in range(10):
    CLASS_MAPPING[i] = str(i)
# Add letters A-Z (ASCII 65-90)
for i in range(26):
    CLASS_MAPPING[i + 10] = chr(i + 65)

# Reverse mapping (label to index)
LABEL_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}