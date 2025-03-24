from model import ASLModel
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, GRU, Dropout, TimeDistributed, Flatten, Conv2D, MaxPooling2D, BatchNormalization

class PatchedASLModel(ASLModel):
    def build_lightweight_hybrid_model(self, sequence_length=1):
        """Patched version of the lightweight hybrid model that ensures output compatibility"""
        print(f"Building patched lightweight_hybrid model with sequence_length={sequence_length}")
        print(f"Using num_classes={self.num_classes}")
        
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
        
        # RNN layer
        x = GRU(64, return_sequences=(sequence_length > 1))(x)
        x = Dropout(0.3)(x)
        
        # Output layer - ensuring we use the correct number of classes
        outputs = Dense(self.num_classes, activation='softmax', name='output_layer')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Print the model output shape for verification
        print(f"Model output shape: {model.output_shape}")
        
        # Compile model with explicit from_logits=False
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Store the model
        self.model = model
        return model 