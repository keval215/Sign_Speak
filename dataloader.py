import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from config import IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT, CLASS_MAPPING
from sklearn.model_selection import train_test_split

def get_class_mapping(data_dir):
    """Get mapping from directory structure"""
    class_mapping = {}
    idx = 0
    
    # Check if the NUMBERS directory exists
    numbers_dir = os.path.join(data_dir, 'NUMBERS')
    if os.path.exists(numbers_dir):
        # Add digit classes
        for i in range(10):
            digit_dir = os.path.join(numbers_dir, str(i))
            if os.path.exists(digit_dir):
                class_mapping[str(i)] = idx
                idx += 1
    
    # Check if the ALPHABETS directory exists
    alphabets_dir = os.path.join(data_dir, 'ALPHABETS')
    if os.path.exists(alphabets_dir):
        # Add alphabet classes
        for i in range(26):
            letter = chr(65 + i)  # A-Z
            letter_dir = os.path.join(alphabets_dir, letter)
            if os.path.exists(letter_dir):
                class_mapping[letter] = idx
                idx += 1
    
    # Check if data directory has direct class folders (like Kaggle dataset)
    if idx == 0:
        # Try to find direct class folders (for Kaggle dataset)
        for item in sorted(os.listdir(data_dir)):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                if item.isdigit() or (len(item) == 1 and item.isalpha()):
                    class_mapping[item.upper()] = idx
                    idx += 1
    
    print(f"Found {idx} classes in {data_dir}")
    return class_mapping

def create_data_generators(data_dir, batch_size=BATCH_SIZE, img_size=IMAGE_SIZE, validation_split=VALIDATION_SPLIT):
    """Create training and validation data generators"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescale for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Check directory structure type
    has_categories = os.path.exists(os.path.join(data_dir, 'ALPHABETS')) or os.path.exists(os.path.join(data_dir, 'NUMBERS'))
    
    if has_categories:
        # For custom dataset with ALPHABETS and NUMBERS folders
        train_generator_list = []
        val_generator_list = []
        
        categories = ['ALPHABETS', 'NUMBERS']
        for category in categories:
            category_path = os.path.join(data_dir, category)
            if os.path.exists(category_path):
                # Create generators for this category
                train_gen = train_datagen.flow_from_directory(
                    category_path,
                    target_size=img_size,
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='training',
                    shuffle=True
                )
                
                val_gen = val_datagen.flow_from_directory(
                    category_path,
                    target_size=img_size,
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='validation',
                    shuffle=False
                )
                
                train_generator_list.append(train_gen)
                val_generator_list.append(val_gen)
        
        # Combine generators
        if len(train_generator_list) > 0:
            train_generator = CombinedGenerator(train_generator_list)
            val_generator = CombinedGenerator(val_generator_list)
            return train_generator, val_generator
    
    # For standard directory structure (like Kaggle dataset)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

class CombinedGenerator:
    """Combines multiple Keras generators into one"""
    def __init__(self, generators):
        self.generators = generators
        self.n = sum([len(g) for g in generators])
        self.class_indices = {}
        self.classes = []
        
        # Combine class indices
        offset = 0
        for gen in generators:
            for class_name, idx in gen.class_indices.items():
                self.class_indices[class_name] = idx + offset
            offset += len(gen.class_indices)
            
        # Set batch size attribute from first generator
        self.batch_size = generators[0].batch_size if generators else 32
    
    def __len__(self):
        return self.n
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Randomly select a generator based on their relative sizes
        weights = [len(g) for g in self.generators]
        total = sum(weights)
        weights = [w/total for w in weights]
        
        selected_gen_idx = np.random.choice(len(self.generators), p=weights)
        x_batch, y_batch = next(self.generators[selected_gen_idx])
        
        return x_batch, y_batch

def load_data_from_directory(data_dir, img_size=IMAGE_SIZE):
    """Load all images and labels from a directory structure into memory"""
    images = []
    labels = []
    class_mapping = {}
    
    # Use existing class mapping from config if available, otherwise create from directory
    if CLASS_MAPPING:
        class_mapping = {v: k for k, v in CLASS_MAPPING.items()}  # Reverse mapping (label to index)
    else:
        class_mapping = get_class_mapping(data_dir)
    
    # Check directory structure type
    has_categories = os.path.exists(os.path.join(data_dir, 'ALPHABETS')) or os.path.exists(os.path.join(data_dir, 'NUMBERS'))
    
    if has_categories:
        # For custom dataset with ALPHABETS and NUMBERS folders
        categories = ['ALPHABETS', 'NUMBERS']
        for category in categories:
            category_path = os.path.join(data_dir, category)
            if not os.path.exists(category_path):
                continue
                
            for class_name in os.listdir(category_path):
                class_path = os.path.join(category_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                    
                # Get class index from mapping or continue if not found
                class_idx = None
                if CLASS_MAPPING:
                    class_idx = class_mapping.get(class_name.upper())
                else:
                    class_idx = class_mapping.get(class_name.upper())
                
                if class_idx is None:
                    continue
                    
                # Load images from this class
                for img_file in os.listdir(class_path):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    img_path = os.path.join(class_path, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    # Preprocess image
                    img = cv2.resize(img, img_size)
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(class_idx)
    else:
        # For standard directory structure (like Kaggle dataset)
        for class_name in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            # Get class index from mapping
            class_idx = None
            if CLASS_MAPPING:
                # For digits 0-9
                if class_name.isdigit():
                    class_idx = int(class_name)
                # For letters A-Z
                elif len(class_name) == 1 and class_name.upper().isalpha():
                    letter_code = ord(class_name.upper()) - 65  # A=0, B=1, etc.
                    class_idx = letter_code + 10  # Offset by 10 for the digits
            else:
                class_idx = class_mapping.get(class_name.upper())
                
            if class_idx is None:
                continue
                
            # Load images from this class
            for img_file in os.listdir(class_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Preprocess image
                img = cv2.resize(img, img_size)
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(class_idx)
    
    # Convert to numpy arrays
    if not images:
        print(f"No images found in {data_dir}")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    X = np.array(images)
    y = np.array(labels)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, stratify=y, random_state=42)
    
    print(f"Loaded {len(X_train)} training images and {len(X_val)} validation images")
    return X_train, y_train, X_val, y_val

def preprocess_batch_for_lstm(X_batch, sequence_length=1):
    """
    Reshape batch data for LSTM model
    
    Args:
        X_batch: Batch of images with shape (batch_size, height, width, channels)
        sequence_length: Length of sequence dimension to add
        
    Returns:
        Reshaped batch with shape (batch_size, sequence_length, height, width, channels)
    """
    # Add sequence dimension
    return np.expand_dims(X_batch, axis=1)

def process_image_for_prediction(image_path, target_size=IMAGE_SIZE):
    """
    Load and preprocess a single image for prediction
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image ready for model prediction
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize to target size
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img