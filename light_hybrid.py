import os
import argparse
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Import the patched model
from model_patch import PatchedASLModel
from dataloader import create_data_generators, load_data_from_directory
from config import (
    CUSTOM_PROCESSED_DIR, 
    KAGGLE_DATASET_DIR, 
    MODEL_CHECKPOINT_DIR,
    NUM_CLASSES, 
    EPOCHS, 
    BATCH_SIZE, 
    LEARNING_RATE,
    IMAGE_SIZE,
    CLASS_MAPPING,
    FINAL_MODEL_PATH
)

def reshape_data_for_lstm(X, sequence_length=1):
    """Reshape input data for LSTM model"""
    if sequence_length == 1:
        return np.expand_dims(X, axis=1)
    else:
        return np.expand_dims(X, axis=1)

def train_lightweight_hybrid(epochs=EPOCHS, train_kaggle=True, train_custom=True, load_all_data=True, sequence_length=1):
    """Train the patched lightweight hybrid model"""
    print("Starting training with patched lightweight hybrid model")
    print(f"Training on Kaggle dataset: {train_kaggle}")
    print(f"Training on Custom dataset: {train_custom}")
    
    # Create patched model
    asl_model = PatchedASLModel(
        num_classes=NUM_CLASSES,
        input_shape=(*IMAGE_SIZE, 3),
        learning_rate=LEARNING_RATE
    )
    
    # Build the patched lightweight hybrid model
    model = asl_model.build_lightweight_hybrid_model(sequence_length)
    
    # Print model summary
    model.summary()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_CHECKPOINT_DIR, "patched_lightweight_hybrid_model.h5")
    
    # Get callbacks
    callbacks = asl_model.get_callbacks(checkpoint_path)
    
    # Load all data into memory
    if load_all_data:
        # Load and combine datasets
        X_all = []
        y_all = []
        
        if train_custom and os.path.exists(CUSTOM_PROCESSED_DIR):
            print("Loading custom dataset...")
            X_custom_train, y_custom_train, X_custom_val, y_custom_val = load_data_from_directory(CUSTOM_PROCESSED_DIR)
            
            # Print shapes for debugging
            print(f"Custom train X shape: {X_custom_train.shape}, y shape: {y_custom_train.shape}")
            print(f"Custom val X shape: {X_custom_val.shape}, y shape: {y_custom_val.shape}")
            
            # Check if labels are already one-hot encoded
            if len(y_custom_train.shape) > 1:
                print("Custom train data is already one-hot encoded")
                if y_custom_train.shape[1] != NUM_CLASSES:
                    print(f"WARNING: Custom train data has {y_custom_train.shape[1]} classes, but model expects {NUM_CLASSES}")
                    # Convert back to indices
                    y_custom_train_indices = np.argmax(y_custom_train, axis=1)
                    y_custom_train = to_categorical(y_custom_train_indices, num_classes=NUM_CLASSES)
                    
                    y_custom_val_indices = np.argmax(y_custom_val, axis=1)
                    y_custom_val = to_categorical(y_custom_val_indices, num_classes=NUM_CLASSES)
            else:
                # Convert to one-hot with the right number of classes
                y_custom_train = to_categorical(y_custom_train, num_classes=NUM_CLASSES)
                y_custom_val = to_categorical(y_custom_val, num_classes=NUM_CLASSES)
            
            X_all.extend([X_custom_train, X_custom_val])
            y_all.extend([y_custom_train, y_custom_val])
            
        if train_kaggle and os.path.exists(KAGGLE_DATASET_DIR):
            print("Loading Kaggle dataset...")
            X_kaggle_train, y_kaggle_train, X_kaggle_val, y_kaggle_val = load_data_from_directory(KAGGLE_DATASET_DIR)
            
            # Print shapes for debugging
            print(f"Kaggle train X shape: {X_kaggle_train.shape}, y shape: {y_kaggle_train.shape}")
            print(f"Kaggle val X shape: {X_kaggle_val.shape}, y shape: {y_kaggle_val.shape}")
            
            # Check if labels are already one-hot encoded
            if len(y_kaggle_train.shape) > 1:
                print("Kaggle train data is already one-hot encoded")
                if y_kaggle_train.shape[1] != NUM_CLASSES:
                    print(f"WARNING: Kaggle train data has {y_kaggle_train.shape[1]} classes, but model expects {NUM_CLASSES}")
                    # Convert back to indices
                    y_kaggle_train_indices = np.argmax(y_kaggle_train, axis=1)
                    y_kaggle_train = to_categorical(y_kaggle_train_indices, num_classes=NUM_CLASSES)
                    
                    y_kaggle_val_indices = np.argmax(y_kaggle_val, axis=1)
                    y_kaggle_val = to_categorical(y_kaggle_val_indices, num_classes=NUM_CLASSES)
            else:
                # Convert to one-hot with the right number of classes
                y_kaggle_train = to_categorical(y_kaggle_train, num_classes=NUM_CLASSES)
                y_kaggle_val = to_categorical(y_kaggle_val, num_classes=NUM_CLASSES)
            
            X_all.extend([X_kaggle_train, X_kaggle_val])
            y_all.extend([y_kaggle_train, y_kaggle_val])
        
        # Combine datasets
        X_train = np.concatenate([x for x in X_all if x.size > 0])
        y_train = np.concatenate([y for y in y_all if y.size > 0])
        
        # Print combined shapes for debugging
        print(f"Combined X_train shape: {X_train.shape}")
        print(f"Combined y_train shape: {y_train.shape}")
        
        # Double-check label dimensions
        if y_train.shape[1] != NUM_CLASSES:
            print(f"ERROR: Final y_train has {y_train.shape[1]} classes, but model expects {NUM_CLASSES}")
            # Fix the shape
            y_train_indices = np.argmax(y_train, axis=1)
            y_train = to_categorical(y_train_indices, num_classes=NUM_CLASSES)
            print(f"Fixed y_train shape: {y_train.shape}")
            
        # Reshape data for LSTM (add sequence dimension)
        X_train_lstm = reshape_data_for_lstm(X_train, sequence_length)
        print(f"X_train_lstm shape: {X_train_lstm.shape}")
        
        # Train the model
        model.fit(
            X_train_lstm, 
            y_train,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_path = os.path.join(MODEL_CHECKPOINT_DIR, "final_patched_lightweight_hybrid.h5")
        asl_model.save_model(final_path)
        print(f"Training complete. Final model saved to {final_path}")
    else:
        print("Use --all-data flag for this patched model")
    
    return model

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train patched lightweight hybrid ASL recognition model")
    parser.add_argument("--kaggle", action="store_true", help="Train on Kaggle dataset")
    parser.add_argument("--custom", action="store_true", help="Train on custom processed dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--no-all-data", action="store_true", help="Don't load all data into memory")
    parser.add_argument("--sequence-length", type=int, default=1, 
                        help="Sequence length for RNN (use 1 for single image classification)")
    
    args = parser.parse_args()
    
    # If neither dataset is specified, use both
    if not (args.kaggle or args.custom):
        args.kaggle = True
        args.custom = True
    
    # Train the model
    train_lightweight_hybrid(
        epochs=args.epochs,
        train_kaggle=args.kaggle,
        train_custom=args.custom,
        load_all_data=not args.no_all_data,
        sequence_length=args.sequence_length
    )