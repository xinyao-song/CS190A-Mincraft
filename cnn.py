import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import json

class GestureCNNTrainer:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the CNN trainer for hand gestures
        
        Args:
            data_dir: Path to dataset directory (should contain subdirectories for each gesture class)
            img_size: Size to resize images to (height, width)
            batch_size: Batch size for training
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load images and labels from directory structure"""
        images = []
        labels = []
        
        # Get class names from subdirectories
        self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        print(f"Found gesture classes: {self.class_names}")
        
        # Load images from each class directory
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            print(f"Loading {class_name} images...")
            
            # Look for various image formats
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(class_dir.glob(ext))
                image_files.extend(class_dir.glob(ext.upper()))
            
            class_count = 0
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"Could not load {img_path}")
                        continue
                        
                    # Convert grayscale to RGB if needed
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    img = cv2.resize(img, self.img_size)
                    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
                    
                    images.append(img)
                    labels.append(class_idx)
                    class_count += 1
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            print(f"  Loaded {class_count} images for {class_name}")
        
        self.X = np.array(images)
        self.y = np.array(labels)
        
        print(f"\nTotal: {len(self.X)} images across {len(self.class_names)} classes")
        print(f"Image shape: {self.X.shape}")
        return self.X, self.y
    
    def build_model(self):
        """Build CNN model without data augmentation layers (to avoid pickle issues)"""
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Build the CNN architecture directly without data augmentation
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(256, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(len(self.class_names), activation='softmax')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model

    def create_augmented_dataset(self, X, y):
        """Create augmented dataset using ImageDataGenerator (more stable approach)"""
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        return datagen
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model compiled successfully!")
        return self.model
    
    def train_model(self, epochs=50, validation_split=0.2, use_augmentation=True):
        """Train the model with optional data augmentation"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=validation_split, 
            stratify=self.y, random_state=42
        )
        
        # Simplified callbacks to avoid pickle issues
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=10, 
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        if use_augmentation:
            # Use ImageDataGenerator for data augmentation
            datagen = self.create_augmented_dataset(X_train, y_train)
            datagen.fit(X_train)
            
            # Calculate steps per epoch
            steps_per_epoch = len(X_train) // self.batch_size
            
            self.history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=self.batch_size),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without augmentation
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.history:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, test_size=0.2):
        """Evaluate model performance"""
        # Create test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, 
            stratify=self.y, random_state=42
        )
        
        # Predictions
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return y_test, y_pred
    
    def save_model(self, model_path='gesture_model.keras', config_path='model_config.json'):
        """Save the trained model and configuration"""
        if self.model:
            # Use the new Keras format to avoid HDF5 warnings
            self.model.save(model_path)
            
            # Save configuration
            config = {
                'class_names': self.class_names,
                'img_size': self.img_size,
                'num_classes': len(self.class_names)
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Model saved to {model_path}")
            print(f"Config saved to {config_path}")
        else:
            print("No model to save. Train the model first.")
    
    def predict_gesture(self, image):
        """Predict gesture from a single image"""
        if not self.model:
            print("No model loaded. Train or load a model first.")
            return None
        
        # Preprocess image
        if isinstance(image, str):  # If path to image
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:  # If numpy array
            img = image
        
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Predict
        predictions = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return {
            'gesture': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_predictions': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        }

# Example usage and training script
def main():
    # Initialize trainer
    trainer = GestureCNNTrainer(data_dir="leapGestRecog/00", img_size=(224, 224))
    
    # Load and preprocess data
    print("Loading dataset...")
    X, y = trainer.load_and_preprocess_data()
    
    # Build model
    print("Building model...")
    model = trainer.build_model()
    trainer.compile_model(learning_rate=0.001)
    
    # Print model summary
    model.summary()
    
    # Train model
    print("Starting training...")
    history = trainer.train_model(epochs=50, validation_split=0.2, use_augmentation=True)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    print("Evaluating model...")
    trainer.evaluate_model()
    
    # Save model
    trainer.save_model('minecraft_gesture_model.keras', 'minecraft_gesture_config.json')
    
    print("Training complete!")

if __name__ == "__main__":
    main()