import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import LSTM, TimeDistributed, Input, Reshape, SimpleRNN, GRU, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

class ASLModel:
    def __init__(self, num_classes=36, input_shape=(128, 128, 3), learning_rate=0.001):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = None
    
    def build_pure_rnn_model(self, sequence_length=1):
        """Build a pure RNN model with NO CNN layers
        
        Args:
            sequence_length: Number of frames in the sequence
        """
        # Input shape: (sequence_length, height, width, channels)
        sequence_input_shape = (sequence_length, *self.input_shape)
        
        # Define input
        inputs = Input(shape=sequence_input_shape)
        
        # Reshape input for RNN
        # Flatten spatial dimensions (height × width × channels)
        input_size = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        x = TimeDistributed(Reshape((input_size,)))(inputs)
        
        # RNN layers
        x = SimpleRNN(128, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = GRU(64, return_sequences=(sequence_length > 1))(x)
        x = Dropout(0.3)(x)
        
        if sequence_length > 1:
            x = LSTM(32)(x)
            x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_lightweight_hybrid_model(self, sequence_length=1):
        """Build a lightweight hybrid CNN-RNN model
        
        Args:
            sequence_length: Number of frames in the sequence
        """
        # Input shape: (sequence_length, height, width, channels)
        sequence_input_shape = (sequence_length, *self.input_shape)
        
        # Define input
        inputs = Input(shape=sequence_input_shape)
        
        # Apply TimeDistributed CNN layers
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        
        # Flatten spatial dimensions
        x = TimeDistributed(Flatten())(x)
        
        # RNN layers
        x = GRU(64, return_sequences=(sequence_length > 1))(x)
        x = Dropout(0.3)(x)
        
        if sequence_length > 1:
            x = LSTM(32)(x)
            x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

    def build_very_lightweight_hybrid_model(self, sequence_length=1):
        """Build a very lightweight hybrid CNN-RNN model for resource-constrained environments
        
        Args:
            sequence_length: Number of frames in the sequence
        """
        # Input shape: (sequence_length, height, width, channels)
        sequence_input_shape = (sequence_length, *self.input_shape)
        
        # Define input
        inputs = Input(shape=sequence_input_shape)
        
        # Apply TimeDistributed CNN layers (fewer filters)
        x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(inputs)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        
        # Use AveragePooling instead of MaxPooling for the final reduction (less parameters)
        x = TimeDistributed(AveragePooling2D((2, 2)))(x)
        
        # Flatten spatial dimensions
        x = TimeDistributed(Flatten())(x)
        
        # Smaller RNN layers
        x = SimpleRNN(32, return_sequences=(sequence_length > 1))(x)
        x = Dropout(0.3)(x)
        
        if sequence_length > 1:
            x = SimpleRNN(16)(x)  # Use SimpleRNN instead of LSTM for fewer parameters
            x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self, checkpoint_path):
        """Get training callbacks"""
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            # Reduce learning rate when plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                mode='min',
                verbose=1
            )
        ]
        return callbacks
    
    def save_model(self, filepath):
        """Save the model to disk"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Build or load a model first.")
    
    def load_model(self, filepath):
        """Load model from disk"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def predict(self, image):
        """Predict class from a single image"""
        if self.model is None:
            raise ValueError("Model not built or loaded")
            
        # Ensure image is a 4D tensor (batch, height, width, channels)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # For LSTM model, add sequence dimension if missing
        if len(image.shape) == 4 and self.model.input_shape[1] != image.shape[0]:
            # Add sequence dimension (batch, seq, height, width, channels)
            image = np.expand_dims(image, axis=1)
            
        # Make prediction
        predictions = self.model.predict(image)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return class_idx, confidence

def main():
    # Create an instance of ASLModel
    model = ASLModel()
    
    print("=== Testing RNN Model ===")
    rnn_model = model.build_pure_rnn_model()
    print("RNN Model Summary:")
    rnn_model.summary()
    print("\n")
    
    print("=== Testing Lightweight Hybrid Model ===")
    hybrid_model = model.build_lightweight_hybrid_model()
    print("Lightweight Hybrid Model Summary:")
    hybrid_model.summary()
    print("\n")
    
    print("=== Testing Very Lightweight Hybrid Model ===")
    very_light_model = model.build_very_lightweight_hybrid_model()
    print("Very Lightweight Hybrid Model Summary:")
    very_light_model.summary()
    
    print("\nAll models built successfully!")
    
    # Optional: Create a small random input and test prediction
    try:
        import numpy as np
        # Create a random image (batch_size=1, height=128, width=128, channels=3)
        test_image = np.random.random((1, 128, 128, 3)).astype('float32')
        print("\nTesting prediction with random input:")
        class_idx, confidence = model.predict(test_image)
        print(f"Predicted class: {class_idx}, Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Prediction test failed: {str(e)}")

if __name__ == "__main__":
    main()