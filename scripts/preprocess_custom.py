import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import shutil
import sys

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils import HandDetector, preprocess_image
from config import (
    CUSTOM_RAW_DIR,
    CUSTOM_PROCESSED_DIR,
    IMAGE_SIZE,
    GRAYSCALE
)

def process_raw_custom_dataset(
    raw_dir=CUSTOM_RAW_DIR, 
    processed_dir=CUSTOM_PROCESSED_DIR,
    image_size=IMAGE_SIZE,
    grayscale=GRAYSCALE,
    hand_confidence=0.3,
    max_files_per_class=None,
    overwrite=False
):
    """
    Process raw custom dataset by detecting and extracting hand regions.
    """
    print(f"Processing raw custom dataset from '{raw_dir}' to '{processed_dir}'")
    print(f"Hand confidence set to: {hand_confidence}")
    print(f"Target image size: {image_size}, Grayscale: {grayscale}")
    
    # Create hand detector
    detector = HandDetector(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=hand_confidence
    )
    
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Keep track of statistics
    total_images = 0
    successful_detections = 0
    failed_detections = 0
    
    # Categories (ALPHABETS and NUMBERS)
    categories = ['ALPHABETS', 'NUMBERS']
    
    for category in categories:
        raw_category_dir = os.path.join(raw_dir, category)
        processed_category_dir = os.path.join(processed_dir, category)
        
        if not os.path.exists(raw_category_dir):
            print(f"[Warning] {raw_category_dir} does not exist. Skipping.")
            continue
        
        os.makedirs(processed_category_dir, exist_ok=True)
        
        for class_name in os.listdir(raw_category_dir):
            raw_class_dir = os.path.join(raw_category_dir, class_name)
            processed_class_dir = os.path.join(processed_category_dir, class_name)
            
            if not os.path.isdir(raw_class_dir):
                continue
            
            print(f"\n[INFO] Processing {category}/{class_name}")
            os.makedirs(processed_class_dir, exist_ok=True)
            
            image_files = [f for f in os.listdir(raw_class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if max_files_per_class and len(image_files) > max_files_per_class:
                image_files = image_files[:max_files_per_class]
            
            for img_file in tqdm(image_files, desc=f"{class_name}"):
                total_images += 1
                src_path = os.path.join(raw_class_dir, img_file)
                dst_path = os.path.join(processed_class_dir, img_file)
                
                if os.path.exists(dst_path) and not overwrite:
                    successful_detections += 1
                    continue
                
                print(f"[DEBUG] Processing file: {src_path}")
                
                try:
                    img = cv2.imread(src_path)
                    if img is None:
                        print(f"[Warning] Could not read {src_path}. Skipping.")
                        failed_detections += 1
                        continue
                    
                    # Extract hand region
                    hand_roi, bbox, hand_detected = detector.extract_hand_roi(
                        img, 
                        padding=30, 
                        target_size=image_size
                    )
                    
                    print(f"[DEBUG] hand_detected: {hand_detected}, bbox: {bbox}, hand_roi is None? {hand_roi is None}")
                    
                    if hand_detected and hand_roi is not None:
                        processed_img = preprocess_image(
                            hand_roi, 
                            target_size=image_size, 
                            normalize=False,  # Don't normalize for saving
                            grayscale=grayscale
                        )
                        # Save processed image
                        cv2.imwrite(dst_path, processed_img)
                        successful_detections += 1
                        print(f"[INFO] Saved processed image to: {dst_path}")
                    else:
                        # If hand not detected, check if ROI exists
                        if hand_roi is not None:
                            dst_path_nohand = os.path.join(processed_class_dir, f"nohand_{img_file}")
                            cv2.imwrite(dst_path_nohand, hand_roi)
                            failed_detections += 1
                            print(f"[INFO] No hand detected, saved fallback image to: {dst_path_nohand}")
                        else:
                            print(f"[Warning] No ROI returned for {src_path}. Skipping.")
                            failed_detections += 1
                
                except Exception as e:
                    print(f"[Error] Exception processing {src_path}: {e}")
                    failed_detections += 1
    
    print("\n[INFO] Processing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Successful hand detections: {successful_detections}")
    print(f"Failed hand detections: {failed_detections}")
    if total_images > 0:
        print(f"Success rate: {successful_detections/total_images*100:.2f}%")
    else:
        print("[WARNING] No images were processed.")

def copy_missing_structure(raw_dir=CUSTOM_RAW_DIR, processed_dir=CUSTOM_PROCESSED_DIR):
    """
    Copy directory structure from raw to processed if missing classes.
    """
    categories = ['ALPHABETS', 'NUMBERS']
    for category in categories:
        raw_category_dir = os.path.join(raw_dir, category)
        processed_category_dir = os.path.join(processed_dir, category)
        if not os.path.exists(raw_category_dir):
            continue
        os.makedirs(processed_category_dir, exist_ok=True)
        for class_name in os.listdir(raw_category_dir):
            raw_class_dir = os.path.join(raw_category_dir, class_name)
            processed_class_dir = os.path.join(processed_category_dir, class_name)
            if os.path.isdir(raw_class_dir) and not os.path.exists(processed_class_dir):
                os.makedirs(processed_class_dir, exist_ok=True)

def visualize_processing_results(raw_dir=CUSTOM_RAW_DIR,
                                 processed_dir=CUSTOM_PROCESSED_DIR,
                                 sample_count=5,
                                 output_dir=None):
    """
    Create visualization of raw vs processed images for inspection.
    """
    if output_dir is None:
        output_dir = os.path.join(processed_dir, "visualization")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple instance of HandDetector for visualization (if needed)
    detector = HandDetector()
    
    categories = ['ALPHABETS', 'NUMBERS']
    for category in categories:
        raw_category_dir = os.path.join(raw_dir, category)
        processed_category_dir = os.path.join(processed_dir, category)
        if not os.path.exists(raw_category_dir) or not os.path.exists(processed_category_dir):
            continue
        for class_name in os.listdir(raw_category_dir):
            raw_class_dir = os.path.join(raw_category_dir, class_name)
            processed_class_dir = os.path.join(processed_category_dir, class_name)
            if not os.path.isdir(raw_class_dir) or not os.path.exists(processed_class_dir):
                continue
            raw_files = [f for f in os.listdir(raw_class_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not raw_files:
                continue
            processed_files = [f for f in os.listdir(processed_class_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('nohand_')]
            common_files = set([f for f in raw_files if f in processed_files])
            if not common_files:
                continue
            sample_files = list(common_files)[:sample_count] if len(common_files) > sample_count else list(common_files)
            for img_file in sample_files:
                raw_path = os.path.join(raw_class_dir, img_file)
                processed_path = os.path.join(processed_class_dir, img_file)
                raw_img = cv2.imread(raw_path)
                processed_img = cv2.imread(processed_path)
                if raw_img is None or processed_img is None:
                    continue
                # Create side-by-side comparison
                comparison = np.hstack([raw_img, processed_img])
                cv2.putText(comparison, "Raw Image", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(comparison, "Processed Image", (raw_img.shape[1] + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                vis_filename = f"{category}_{class_name}_{img_file}"
                vis_path = os.path.join(output_dir, vis_filename)
                cv2.imwrite(vis_path, comparison)
                print(f"[INFO] Saved visualization: {vis_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess custom raw dataset for SignSpeak")
    parser.add_argument("--raw-dir", type=str, default=CUSTOM_RAW_DIR,
                        help="Directory containing raw custom dataset")
    parser.add_argument("--processed-dir", type=str, default=CUSTOM_PROCESSED_DIR,
                        help="Directory to save processed dataset")
    parser.add_argument("--hand-confidence", type=float, default=0.3,
                        help="Minimum confidence for hand detection")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to process per class")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing processed files")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualization of raw vs processed images")
    parser.add_argument("--grayscale", action="store_true", default=GRAYSCALE,
                        help="Convert images to grayscale")
    args = parser.parse_args()
    
    process_raw_custom_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        hand_confidence=args.hand_confidence,
        max_files_per_class=args.max_files,
        overwrite=args.overwrite,
        grayscale=args.grayscale
    )
    
    copy_missing_structure(args.raw_dir, args.processed_dir)
    
    if args.visualize:
        visualize_processing_results(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            sample_count=5
        )

if __name__ == "__main__":
    main()
