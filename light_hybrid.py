import os
import argparse
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, GRU, Dropout, TimeDistributed, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from utils import preprocess_image
from dataloader import create_data_generators
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

# Optional: If you want to combine data from both datasets into a single
# generator, you can use a CombinedGenerator approach.
from tensorflow.keras.utils import Sequence

class CombinedGenerator(Sequence):
    """
    Merges two Keras generators into one, effectively concatenating them
    for the training loop. Inherits from tf.keras.utils.Sequence to be
    compatible with Keras' model.fit() method.
    """
    def __init__(self, gen_a, gen_b):
        self.gen_a = gen_a
        self.gen_b = gen_b
        self.len_a = len(gen_a)
        self.len_b = len(gen_b)
        self.n = self.len_a + self.len_b  # Total steps
        self.batch_size = gen_a.batch_size  # Assuming same batch size for both generators
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        # This logic will alternate between batches from gen_a and gen_b
        if index < self.len_a:
            return self.gen_a[index]
        else:
            return self.gen_b[index - self.len_a]
    
    def on_epoch_end(self):
        # This ensures that each generator starts a new epoch
        self.gen_a.on_epoch_end()
        self.gen_b.on_epoch_end()

class PatchedASLModel:
    def __init__(self, num_classes, input_shape, learning_rate):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = None

    def build_lightweight_hybrid_model(self, sequence_length=1):
        """Patched version of the lightweight hybrid model that ensures output compatibility"""
        print(f"Building patched lightweight_hybrid model with sequence_length={sequence_length}")
        print(f"Using num_classes={self.num_classes}")
        
        # Adjust input shape for sequence models
        if sequence_length == 1:
            # If sequence_length is 1, no need for sequence dimension (single image)
            sequence_input_shape = (*self.input_shape,)  # For single image classification
        else:
            # If sequence_length > 1, we need the sequence dimension
            sequence_input_shape = (sequence_length, *self.input_shape)
        
        # Define input
        inputs = Input(shape=sequence_input_shape)
        
        # If sequence_length > 1, use TimeDistributed to apply Conv2D to each frame in the sequence
        if sequence_length > 1:
            x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
            x = TimeDistributed(BatchNormalization())(x)
            x = TimeDistributed(MaxPooling2D((2, 2)))(x)
            
            x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
            x = TimeDistributed(BatchNormalization())(x)
            x = TimeDistributed(MaxPooling2D((2, 2)))(x)
            
            # Flatten spatial dimensions
            x = TimeDistributed(Flatten())(x)
        else:
            # For single image classification (no sequence), use Conv2D directly
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            
            # Flatten spatial dimensions
            x = Flatten()(x)
        
        # RNN layer (if using sequence, otherwise skip for single image)
        if sequence_length > 1:
            x = GRU(64, return_sequences=True)(x)
            x = Dropout(0.3)(x)
        
        # Output layer - ensuring we use the correct number of classes
        outputs = Dense(self.num_classes, activation='softmax', name='output_layer')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Print the model output shape for verification
        print(f"Model output shape: {model.output_shape}")
        
        # Compile model with explicit from_logits=False
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Store the model
        self.model = model
        return model


def train_lightweight_hybrid(epochs=EPOCHS, train_kaggle=True, train_custom=True, load_all_data=True):
    """
    Train a patched lightweight hybrid model on both custom_processed and kaggle_dataset.
    This script uses ImageDataGenerators to avoid loading everything into memory at once.
    """
    print("Starting training with patched lightweight hybrid model")
    print(f"Training on Kaggle dataset: {train_kaggle}")
    print(f"Training on Custom dataset: {train_custom}")
    
    # Create patched model
    asl_model = PatchedASLModel(
        num_classes=NUM_CLASSES,
        input_shape=(*IMAGE_SIZE, 3),
        learning_rate=LEARNING_RATE
    )
    
    # Build the patched lightweight hybrid model (single-image classification = seq_length=1)
    # If you want sequence data, adjust the sequence_length here.
    model = asl_model.build_lightweight_hybrid_model(sequence_length=1)
    
    # Print model summary
    model.summary()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_CHECKPOINT_DIR, "patched_lightweight_hybrid_model.h5")
    
    # Get callbacks
    callbacks = asl_model.get_callbacks(checkpoint_path)
    
    # Only proceed if load_all_data is True (we interpret this to mean: "train now")
    if not load_all_data:
        print("Use --all-data flag for this patched model")
        return model

    # Create empty references for custom and kaggle generators
    custom_train_gen = None
    custom_val_gen = None
    kaggle_train_gen = None
    kaggle_val_gen = None

    # If custom dataset is requested and folder exists, load its generator
    if train_custom and os.path.exists(CUSTOM_PROCESSED_DIR):
        print("Creating generators for custom_processed dataset...")
        custom_train_gen, custom_val_gen = create_data_generators(
            CUSTOM_PROCESSED_DIR,
            batch_size=BATCH_SIZE,
            img_size=IMAGE_SIZE
        )

    # If kaggle dataset is requested and folder exists, load its generator
    if train_kaggle and os.path.exists(KAGGLE_DATASET_DIR):
        print("Creating generators for kaggle_dataset...")
        kaggle_train_gen, kaggle_val_gen = create_data_generators(
            KAGGLE_DATASET_DIR,
            batch_size=BATCH_SIZE,
            img_size=IMAGE_SIZE
        )
    
    # Combine training generators if both exist
    if custom_train_gen and kaggle_train_gen:
        print("Combining custom and Kaggle training data...")
        train_generator = CombinedGenerator(custom_train_gen, kaggle_train_gen)
    elif custom_train_gen:
        train_generator = custom_train_gen
    elif kaggle_train_gen:
        train_generator = kaggle_train_gen
    else:
        print("No valid training dataset found. Exiting.")
        return model
    
    # Combine validation generators if both exist
    if custom_val_gen and kaggle_val_gen:
        print("Combining custom and Kaggle validation data...")
        val_generator = CombinedGenerator(custom_val_gen, kaggle_val_gen)
    elif custom_val_gen:
        val_generator = custom_val_gen
    elif kaggle_val_gen:
        val_generator = kaggle_val_gen
    else:
        print("No valid validation dataset found. Exiting.")
        return model

    # Train the model using the combined or single generator
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_path = os.path.join(MODEL_CHECKPOINT_DIR, "final_patched_lightweight_hybrid.h5")
    asl_model.save_model(final_path)
    print(f"Training complete. Final model saved to {final_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train patched lightweight hybrid ASL recognition model")
    parser.add_argument("--kaggle", action="store_true", help="Train on Kaggle dataset")
    parser.add_argument("--custom", action="store_true", help="Train on custom processed dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--no-all-data", action="store_true", help="Don't load all data into memory")
    
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
        load_all_data=not args.no_all_data
    )
