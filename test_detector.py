import cv2
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

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

def create_visualization(raw_path, processed_path, output_dir):
    """Create side-by-side comparison of raw and processed images"""
    raw_img = cv2.imread(raw_path)
    processed_img = cv2.imread(processed_path)
    
    if raw_img is None or processed_img is None:
        return False
    
    # Make images same height for comparison
    h1, w1 = raw_img.shape[:2]
    h2, w2 = processed_img.shape[:2]
    
    if h1 != h2:
        # Resize to match heights
        aspect = w1 / h1
        new_w = int(aspect * h2)
        raw_img = cv2.resize(raw_img, (new_w, h2))
    
    # Create side-by-side comparison
    comparison = np.hstack([raw_img, processed_img])
    
    # Add labels
    cv2.putText(comparison, "Raw Image", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Processed Image", (raw_img.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    vis_filename = f"vis_{os.path.basename(raw_path)}"
    vis_path = os.path.join(output_dir, vis_filename)
    cv2.imwrite(vis_path, comparison)
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test hand detection on sign language images")
    parser.add_argument("--category", type=str, default="ALPHABETS", choices=["ALPHABETS", "NUMBERS"],
                        help="Category to process (ALPHABETS or NUMBERS)")
    parser.add_argument("--class", dest="class_name", type=str, 
                        help="Class to process (e.g., 'a', 'b', or '1', '2')")
    parser.add_argument("--grayscale", action="store_true", default=False,
                        help="Convert images to grayscale")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Hand detection confidence threshold")
    parser.add_argument("--batch", action="store_true", 
                        help="Process images in batch mode without showing each image")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to process")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualization of raw vs processed images")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing processed files")
    args = parser.parse_args()
    
    # Folders for raw input and processed output
    raw_folder = os.path.join(CUSTOM_RAW_DIR, args.category, args.class_name)
    processed_folder = os.path.join(CUSTOM_PROCESSED_DIR, args.category, args.class_name)
    
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
    
    # Apply max files limit if specified
    if args.max_files and len(files) > args.max_files:
        files = files[:args.max_files]
    
    print(f"[INFO] Found {len(files)} image(s) in {raw_folder}")
    
    # Create the hand detector
    detector = HandDetector(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=args.confidence
    )
    
    # Statistics
    total_images = 0
    successful_detections = 0
    failed_detections = 0
    
    # Process images with tqdm progress bar
    for img_file in tqdm(files, desc="Processing images"):
        total_images += 1
        img_path = os.path.join(raw_folder, img_file)
        dst_path = os.path.join(processed_folder, img_file)
        
        # Skip if file exists and no overwrite flag
        if os.path.exists(dst_path) and not args.overwrite:
            print(f"[INFO] Skipping existing file: {dst_path}")
            successful_detections += 1
            continue
        
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"[WARNING] Could not read image: {img_path}")
            failed_detections += 1
            continue
        
        # Perform hand detection
        hand_roi, bbox, hand_detected = detector.extract_hand_roi(
            img,
            padding=30,
            target_size=IMAGE_SIZE
        )
        
        if not args.batch:
            print(f"\n[DEBUG] Processing: {img_file}")
            print(f"hand_detected: {hand_detected}, bbox: {bbox}")
        
        if hand_detected and hand_roi is not None:
            # Preprocess the ROI
            processed_img = preprocess_image(
                hand_roi,
                target_size=IMAGE_SIZE,
                grayscale=args.grayscale,
                normalize=False  # Don't normalize for saving
            )
            # Save the processed image
            cv2.imwrite(dst_path, processed_img)
            successful_detections += 1
            if not args.batch:
                print(f"[INFO] Saved processed image to: {dst_path}")
                # Optionally display the processed ROI
                cv2.imshow("Hand ROI", processed_img)
                print("[INFO] Press any key to continue to next image...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            # If no hand was detected, save a "nohand_" fallback
            nohand_path = os.path.join(processed_folder, f"nohand_{img_file}")
            cv2.imwrite(nohand_path, img)
            failed_detections += 1
            
            if not args.batch:
                print(f"[INFO] No hand detected; saved fallback to: {nohand_path}")
                # Optionally display the original image
                cv2.imshow("No Hand Detected", img)
                print("[INFO] Press any key to continue to next image...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    # Print statistics
    print("\n[INFO] Processing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Successful hand detections: {successful_detections}")
    print(f"Failed hand detections: {failed_detections}")
    if total_images > 0:
        print(f"Success rate: {successful_detections/total_images*100:.2f}%")
    
    # Create visualizations if requested
    if args.visualize:
        vis_dir = os.path.join(processed_folder, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        print("[INFO] Creating visualizations...")
        
        vis_count = 0
        for img_file in files:
            raw_path = os.path.join(raw_folder, img_file)
            processed_path = os.path.join(processed_folder, img_file)
            if os.path.exists(processed_path):
                success = create_visualization(raw_path, processed_path, vis_dir)
                if success:
                    vis_count += 1
        
        print(f"[INFO] Created {vis_count} visualizations in {vis_dir}")

if __name__ == "__main__":
    main()
