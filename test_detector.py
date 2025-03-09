import cv2
import os
import sys

# Add parent directory to path to import project modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from utils import HandDetector, preprocess_image
from config import (
    CUSTOM_RAW_DIR,
    CUSTOM_PROCESSED_DIR,
    IMAGE_SIZE
)

def main():
    # Folders for raw input and processed output
    raw_folder = os.path.join(CUSTOM_RAW_DIR, "ALPHABETS", "a")
    processed_folder = os.path.join(CUSTOM_PROCESSED_DIR, "ALPHABETS", "a")
    
    # Create the processed output folder if it doesn't exist
    os.makedirs(processed_folder, exist_ok=True)
    
    if not os.path.exists(raw_folder):
        print(f"[ERROR] Folder does not exist: {raw_folder}")
        return
    
    # List all image files in the raw_folder
    files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"[INFO] No image files found in {raw_folder}")
        return
    
    print(f"[INFO] Found {len(files)} image(s) in {raw_folder}")
    
    # Create the hand detector
    detector = HandDetector(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )
    
    for img_file in files:
        img_path = os.path.join(raw_folder, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"[WARNING] Could not read image: {img_path}")
            continue
        
        # Perform hand detection
        hand_roi, bbox, hand_detected = detector.extract_hand_roi(
            img,
            padding=30,
            target_size=IMAGE_SIZE
        )
        
        print(f"\n[DEBUG] Processing: {img_file}")
        print(f"hand_detected: {hand_detected}, bbox: {bbox}")
        
        if hand_detected and hand_roi is not None:
            # Preprocess the ROI (no normalization for saving, you can enable if desired)
            processed_img = preprocess_image(
                hand_roi,
                target_size=IMAGE_SIZE,
                grayscale=False,   # or True if you want grayscale
                normalize=False
            )
            
            # Save the processed image to custom_processed
            save_path = os.path.join(processed_folder, img_file)
            cv2.imwrite(save_path, processed_img)
            print(f"[INFO] Saved processed image to: {save_path}")
            
            # Optionally display the processed ROI
            cv2.imshow("Hand ROI", processed_img)
            print("[INFO] Press any key to continue to next image...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # If no hand was detected, save a "nohand_" fallback
            nohand_path = os.path.join(processed_folder, f"nohand_{img_file}")
            cv2.imwrite(nohand_path, img)
            print(f"[INFO] No hand detected; saved fallback to: {nohand_path}")
            
            # Optionally display the original image
            cv2.imshow("No Hand Detected", img)
            print("[INFO] Press any key to continue to next image...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
