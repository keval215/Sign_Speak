import os
import argparse
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Local imports
from model import ASLModel
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
    """
    Reshape input data for LSTM model
    For single images, adds a sequence dimension with length 1
    
    Args:
        X: Input data with shape (batch, height, width, channels)
        sequence_length: Length of sequence dimension to add
    
    Returns:
        Reshaped data with shape (batch, sequence_length, height, width, channels)
    """
    # If sequence_length is 1, just add a dimension
    if sequence_length == 1:
        return np.expand_dims(X, axis=1)
    else:
        # For future implementation with longer sequences
        # This would require a different data preparation approach
        return np.expand_dims(X, axis=1)

def train_model(model_type="lightweight_hybrid", train_kaggle=True, train_custom=True, load_all_data=False, 
                sequence_length=1, epochs=EPOCHS):
    """Train the ASL recognition model
    
    Args:
        model_type: Type of model to train ('lightweight_hybrid', 'pure_rnn', or 'very_lightweight')
        train_kaggle: Whether to train on Kaggle dataset
        train_custom: Whether to train on custom processed dataset
        load_all_data: Whether to load all data into memory
        sequence_length: Length of sequence dimension for LSTM
        epochs: Number of training epochs
    """
    print(f"Starting training with model type: {model_type}")
    print(f"Training on Kaggle dataset: {train_kaggle}")
    print(f"Training on Custom dataset: {train_custom}")
    
    # Create model
    asl_model = ASLModel(
        num_classes=NUM_CLASSES,
        input_shape=(*IMAGE_SIZE, 3),
        learning_rate=LEARNING_RATE
    )
    
    # Build model based on type
    if model_type == "lightweight_hybrid":
        model = asl_model.build_lightweight_hybrid_model(sequence_length)
    elif model_type == "pure_rnn":
        model = asl_model.build_pure_rnn_model(sequence_length)
    elif model_type == "very_lightweight":
        model = asl_model.build_very_lightweight_hybrid_model(sequence_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Print model summary
    model.summary()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_CHECKPOINT_DIR, f"{model_type}_model.h5")
    
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
            X_all.extend([X_custom_train, X_custom_val])
            y_all.extend([y_custom_train, y_custom_val])
            
        if train_kaggle and os.path.exists(KAGGLE_DATASET_DIR):
            print("Loading Kaggle dataset...")
            X_kaggle_train, y_kaggle_train, X_kaggle_val, y_kaggle_val = load_data_from_directory(KAGGLE_DATASET_DIR)
            X_all.extend([X_kaggle_train, X_kaggle_val])
            y_all.extend([y_kaggle_train, y_kaggle_val])
        
        # Combine datasets
        X_train = np.concatenate([x for x in X_all if x.size > 0])
        y_train_raw = np.concatenate([y for y in y_all if y.size > 0])
        
        # Reshape data for LSTM (add sequence dimension)
        X_train_lstm = reshape_data_for_lstm(X_train, sequence_length)
        
        # Convert to one-hot encoding
        y_train = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
        
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
    else:
        # Use data generators with custom data processing
        if train_custom and os.path.exists(CUSTOM_PROCESSED_DIR):
            print("Training on custom processed dataset...")
            custom_train_gen, custom_val_gen = create_data_generators(
                CUSTOM_PROCESSED_DIR,
                batch_size=BATCH_SIZE,
                img_size=IMAGE_SIZE
            )
            
            # Custom generator wrapper for LSTM input
            def lstm_generator_wrapper(gen):
                for X_batch, y_batch in gen:
                    X_batch_lstm = np.expand_dims(X_batch, axis=1)
                    yield X_batch_lstm, y_batch
            
            # Train on custom dataset with LSTM generator
            custom_train_gen_lstm = lstm_generator_wrapper(custom_train_gen)
            custom_val_gen_lstm = lstm_generator_wrapper(custom_val_gen)
            
            model.fit(
                custom_train_gen_lstm,
                steps_per_epoch=len(custom_train_gen),
                validation_data=custom_val_gen_lstm,
                validation_steps=len(custom_val_gen),
                epochs=epochs // 2 if train_kaggle else epochs,  # Split epochs if using both datasets
                callbacks=callbacks,
                verbose=1
            )
            
        if train_kaggle and os.path.exists(KAGGLE_DATASET_DIR):
            print("Training on Kaggle dataset...")
            kaggle_train_gen, kaggle_val_gen = create_data_generators(
                KAGGLE_DATASET_DIR,
                batch_size=BATCH_SIZE,
                img_size=IMAGE_SIZE
            )
            
            # Custom generator wrapper for LSTM input
            def lstm_generator_wrapper(gen):
                for X_batch, y_batch in gen:
                    X_batch_lstm = np.expand_dims(X_batch, axis=1)
                    yield X_batch_lstm, y_batch
            
            # Train on Kaggle dataset with LSTM generator
            kaggle_train_gen_lstm = lstm_generator_wrapper(kaggle_train_gen)
            kaggle_val_gen_lstm = lstm_generator_wrapper(kaggle_val_gen)
            
            model.fit(
                kaggle_train_gen_lstm,
                steps_per_epoch=len(kaggle_train_gen),
                validation_data=kaggle_val_gen_lstm,
                validation_steps=len(kaggle_val_gen),
                epochs=epochs // 2 if train_custom else epochs,
                callbacks=callbacks,
                verbose=1
            )
    
    # Save final model
    asl_model.save_model(FINAL_MODEL_PATH)
    print(f"Training complete. Final model saved to {FINAL_MODEL_PATH}")
    
    return model

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train ASL recognition model")
    parser.add_argument("--model-type", type=str, default="lightweight_hybrid", 
                      choices=["lightweight_hybrid", "pure_rnn", "very_lightweight"],
                      help="Model architecture to use")
    parser.add_argument("--kaggle", action="store_true", help="Train on Kaggle dataset")
    parser.add_argument("--custom", action="store_true", help="Train on custom processed dataset")
    parser.add_argument("--all-data", action="store_true", help="Load all data into memory instead of using generators")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--sequence-length", type=int, default=1, 
                        help="Sequence length for RNN (use 1 for single image classification)")
    
    args = parser.parse_args()
    
    # If neither dataset is specified, use both
    if not (args.kaggle or args.custom):
        args.kaggle = True
        args.custom = True
    
    # Train the model
    train_model(
        model_type=args.model_type,
        train_kaggle=args.kaggle,
        train_custom=args.custom,
        load_all_data=args.all_data,
        sequence_length=args.sequence_length,
        epochs=args.epochs
    )   