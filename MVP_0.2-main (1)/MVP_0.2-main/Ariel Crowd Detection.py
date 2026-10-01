import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import json
import time
import threading
from collections import deque
import gc
import pandas as pd # Added for load_custom_dataset

# Configure GPU memory growth to prevent VRAM issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# For YOLO integration (optional)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install with: pip install ultralytics")

# For DETR integration (optional)
try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers")

class OptimizedAerialCrowdDetector:
    """
    An optimized class for aerial crowd detection, designed for efficient memory usage
    and performance on GPUs like the NVIDIA RTX 3050.
    """
    def __init__(self, img_height: int = 224, img_width: int = 224, use_mixed_precision: bool = True):
        """
        Initializes the OptimizedAerialCrowdDetector.

        Args:
            img_height (int): The target height for image resizing.
            img_width (int): The target width for image resizing.
            use_mixed_precision (bool): Whether to enable mixed precision training (float16).
        """
        self.img_height = img_height
        self.img_width = img_width
        self.presence_model: keras.Model = None
        self.density_model: keras.Model = None
        self.yolo_model: YOLO = None
        self.detr_model = None
        self.detr_processor = None
        self.scaler = StandardScaler() # Not explicitly used in current training flow, consider removing if not needed.
        self.real_time_buffer = deque(maxlen=30) # Not explicitly used in current training flow, consider removing if not needed.
        
        # Enable mixed precision for better performance on RTX 3050
        if use_mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled for better performance")
        
        # Set memory management
        self.batch_size = 8  # Smaller batch size for 8GB VRAM
        self.max_samples = 1000  # Reduced samples to fit in memory (for synthetic data)
        
    def preprocess_image(self, image_input: str | np.ndarray) -> np.ndarray:
        """
        Optimized preprocessing for input images, handling both file paths and numpy arrays.

        Args:
            image_input (str | np.ndarray): Path to the image file or a NumPy array representing the image.

        Returns:
            np.ndarray: The preprocessed and normalized image array.

        Raises:
            ValueError: If the image cannot be loaded.
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input
            
        if img is None:
            raise ValueError(f"Could not load image: {image_input}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize with optimized interpolation
        img_resized = cv2.resize(img_rgb, (self.img_width, self.img_height), 
                                interpolation=cv2.INTER_LINEAR)
        
        # Normalize efficiently
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    
    def build_efficient_presence_model(self) -> keras.Model:
        """
        Builds a memory-efficient CNN model for crowd presence detection,
        optimized for RTX 3050 with L2 regularization and Batch Normalization.

        Returns:
            keras.Model: The compiled presence detection model.
        """
        model = keras.Sequential([
            # Lightweight convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_height, self.img_width, 3),
                         kernel_regularizer=keras.regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(), # Replaced Flatten with GlobalAveragePooling for fewer parameters
            
            # Compact dense layers
            layers.Dense(64, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.0001)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid', name='crowd_presence',
                        dtype='float32'),  # Ensure float32 output
        ])
        
        return model
    
    def build_efficient_density_model(self) -> keras.Model:
        """
        Builds an efficient crowd density estimation model using MobileNetV2 as a backbone.

        Returns:
            keras.Model: The compiled density estimation model.
        """
        # Use MobileNetV2 instead of ResNet50 for better memory efficiency
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3),
            alpha=0.75  # Reduced width multiplier for lower memory usage
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear', name='crowd_density',
                        dtype='float32'),  # Ensure float32 output
        ])
        
        return model
    
    def create_memory_efficient_training_data(self, num_samples: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates synthetic training data with memory management by generating
        samples in smaller batches. This function is for demonstration and
        testing purposes. For real applications, use `load_custom_dataset`.

        Args:
            num_samples (int, optional): The total number of samples to generate.
                                         Defaults to self.max_samples.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing
                                                        (X, y_presence, y_density) arrays.
        """
        if num_samples is None:
            num_samples = self.max_samples
        
        print(f"Creating {num_samples} training samples (synthetic)...")
        
        batch_size = 100
        X_batches = []
        y_presence_batches = []
        y_density_batches = []
        
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_samples = batch_end - batch_start
            
            X_batch = []
            y_presence_batch = []
            y_density_batch = []
            
            for i in range(batch_samples):
                # Create diverse background
                background_type = np.random.choice(['grass', 'concrete', 'mixed'])
                
                if background_type == 'grass':
                    img = np.random.rand(self.img_height, self.img_width, 3) * 0.2 + 0.3
                    img[:, :, 1] *= 1.5  # More green
                elif background_type == 'concrete':
                    gray_val = np.random.uniform(0.4, 0.7)
                    img = np.ones((self.img_height, self.img_width, 3)) * gray_val
                    img += np.random.rand(self.img_height, self.img_width, 3) * 0.1
                else:
                    img = np.random.rand(self.img_height, self.img_width, 3) * 0.4 + 0.3
                
                # Add noise
                noise = np.random.randn(self.img_height, self.img_width, 3) * 0.05
                img = np.clip(img + noise, 0, 1)
                
                # Crowd simulation
                crowd_present = np.random.rand() > 0.3
                
                if crowd_present:
                    # Adjust density distribution for more realistic scenarios
                    density_tier = np.random.rand()
                    if density_tier > 0.8: # High density
                        crowd_density = np.random.uniform(0.6, 1.0)
                    elif density_tier > 0.4: # Medium density
                        crowd_density = np.random.uniform(0.3, 0.6)
                    else: # Low density
                        crowd_density = np.random.uniform(0.05, 0.3) # Min density slightly higher for 'present'
                    
                    num_people = int(crowd_density * 100) # Scale num_people based on density
                    
                    # Add people with different patterns
                    pattern = np.random.choice(['clustered', 'scattered', 'linear'])
                    
                    if pattern == 'clustered':
                        num_clusters = np.random.randint(2, 4)
                        for cluster in range(num_clusters):
                            cx = np.random.randint(20, self.img_width - 20)
                            cy = np.random.randint(20, self.img_height - 20)
                            cluster_size = num_people // num_clusters
                            
                            for _ in range(cluster_size):
                                angle = np.random.uniform(0, 2 * np.pi)
                                radius = np.random.exponential(15)
                                radius = min(radius, 25)
                                
                                x = int(cx + radius * np.cos(angle))
                                y = int(cy + radius * np.sin(angle))
                                x = max(0, min(x, self.img_width - 1))
                                y = max(0, min(y, self.img_height - 1))
                                
                                person_size = np.random.randint(1, 3)
                                person_color = np.random.rand(3) * 0.4 + 0.1
                                cv2.circle(img, (x, y), person_size, person_color, -1)
                    
                    elif pattern == 'scattered':
                        for _ in range(num_people):
                            x = np.random.randint(5, self.img_width - 5)
                            y = np.random.randint(5, self.img_height - 5)
                            person_size = np.random.randint(1, 3)
                            person_color = np.random.rand(3) * 0.4 + 0.1
                            cv2.circle(img, (x, y), person_size, person_color, -1)
                    
                    else:  # linear
                        start_x = np.random.randint(20, self.img_width - 20)
                        start_y = np.random.randint(20, self.img_height - 20)
                        direction = np.random.uniform(0, 2 * np.pi)
                        
                        for i in range(num_people):
                            x = int(start_x + i * 6 * np.cos(direction))
                            y = int(start_y + i * 6 * np.sin(direction))
                            
                            if 0 <= x < self.img_width and 0 <= y < self.img_height:
                                person_size = np.random.randint(1, 3)
                                person_color = np.random.rand(3) * 0.4 + 0.1
                                cv2.circle(img, (x, y), person_size, person_color, -1)
                    
                    presence = 1
                    density = crowd_density
                else:
                    presence = 0
                    density = 0
                
                X_batch.append(img.astype(np.float32))
                y_presence_batch.append(presence)
                y_density_batch.append(density)
            
            X_batches.append(np.array(X_batch))
            y_presence_batches.append(np.array(y_presence_batch))
            y_density_batches.append(np.array(y_density_batch))
            
            # Progress update
            if (batch_start + batch_size) % 500 == 0: # Adjusted frequency for cleaner output
                print(f"Generated {batch_start + batch_size}/{num_samples} samples")
        
        # Combine all batches
        X = np.concatenate(X_batches, axis=0)
        y_presence = np.concatenate(y_presence_batches, axis=0)
        y_density = np.concatenate(y_density_batches, axis=0)
        
        # Clean up
        del X_batches, y_presence_batches, y_density_batches
        gc.collect()
        
        return X, y_presence, y_density
    
    def train_models_optimized(self, X_train: np.ndarray, y_presence: np.ndarray, 
                             y_density: np.ndarray, validation_split: float = 0.2, 
                             epochs: int = 25) -> tuple[keras.callbacks.History, keras.callbacks.History]:
        """
        Trains the crowd presence and density estimation models with optimizations
        for the RTX 3050.

        Args:
            X_train (np.ndarray): The input image data for training.
            y_presence (np.ndarray): The target presence labels.
            y_density (np.ndarray): The target density values.
            validation_split (float): The proportion of the training data to be used for validation.
            epochs (int): The number of training epochs.

        Returns:
            tuple[keras.callbacks.History, keras.callbacks.History]: Training history for
                                                                   presence and density models.
        """
        print("Building optimized models...")
        
        # Build efficient models
        self.presence_model = self.build_efficient_presence_model()
        self.density_model = self.build_efficient_density_model()
        
        # Use mixed precision optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        # Compile models
        self.presence_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.density_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Memory-efficient callbacks
        presence_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5, 
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'best_presence_model.h5',
                monitor='val_accuracy',
                mode='max', # Added mode for accuracy monitoring
                save_best_only=True,
                save_weights_only=False,
                verbose=0 # Suppress verbose output for checkpoint
            )
        ]
        
        density_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5, 
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'best_density_model.h5',
                monitor='val_mae',
                mode='min', # Added mode for MAE monitoring
                save_best_only=True,
                save_weights_only=False,
                verbose=0 # Suppress verbose output for checkpoint
            )
        ]
        
        # Train presence model
        print("Training presence detection model...")
        with tf.device('/GPU:0'):
            presence_history = self.presence_model.fit(
                X_train, y_presence,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=self.batch_size,
                callbacks=presence_callbacks,
                verbose=1
            )
        
        # Clear session and collect garbage after a significant training phase
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Train density model
        print("Training density estimation model...")
        with tf.device('/GPU:0'):
            density_history = self.density_model.fit(
                X_train, y_density,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=self.batch_size,
                callbacks=density_callbacks,
                verbose=1
            )
        
        # Final cleanup
        tf.keras.backend.clear_session()
        gc.collect()
        
        return presence_history, density_history
    
    def initialize_yolo_lightweight(self) -> bool:
        """
        Initializes a lightweight YOLO model (YOLOv8n).

        Returns:
            bool: True if YOLO model was successfully initialized, False otherwise.
        """
        if not YOLO_AVAILABLE:
            print("YOLO not available. Please install 'ultralytics' library.")
            return False
        
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("Lightweight YOLO model (YOLOv8n) initialized.")
            return True
        except Exception as e:
            print(f"Error initializing YOLO: {e}")
            return False
            
    def initialize_detr(self) -> bool:
        """
        Initializes a DETR model. (Placeholder, requires more specific implementation)

        Returns:
            bool: True if DETR model was successfully initialized, False otherwise.
        """
        if not DETR_AVAILABLE:
            print("Transformers (DETR) not available. Please install 'transformers' library.")
            return False
        try:
            self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            print("DETR model initialized (using detr-resnet-50).")
            return True
        except Exception as e:
            print(f"Error initializing DETR: {e}")
            return False

    def predict_batch_optimized(self, images: np.ndarray) -> list[dict]:
        """
        Performs batch prediction for crowd presence and density with memory optimization.

        Args:
            images (np.ndarray): A batch of preprocessed images.

        Returns:
            list[dict]: A list of dictionaries, each containing 'presence_probability',
                        'presence_detected', and 'density_score' for an image.
        """
        results = []
        
        # Process in small batches to manage memory
        batch_size = 4 # Or self.batch_size, depending on desired inference batch size
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            current_batch_results = []
            
            # Presence prediction
            if self.presence_model:
                presence_probs = self.presence_model.predict(batch, verbose=0)
                for prob in presence_probs:
                    current_batch_results.append({
                        'presence_probability': float(prob[0]),
                        'presence_detected': prob[0] > 0.5
                    })
            
            # Density prediction
            if self.density_model:
                density_scores = self.density_model.predict(batch, verbose=0)
                for j, score in enumerate(density_scores):
                    if j < len(current_batch_results): # Ensure index is within bounds
                        current_batch_results[j]['density_score'] = float(score[0])
                    else:
                        # This case should ideally not happen if models are consistent
                        current_batch_results.append({'density_score': float(score[0])}) 
            results.extend(current_batch_results)
        
        return results
    
    def monitor_gpu_memory(self) -> tuple[float | None, float | None]:
        """
        Monitors and prints current and peak GPU memory usage for the first GPU.

        Returns:
            tuple[float | None, float | None]: A tuple of (current_memory_mb, peak_memory_mb)
                                                or (None, None) if no GPU is found or an error occurs.
        """
        if gpus:
            try:
                # Use experimental.get_memory_info for more detailed stats
                # Note: 'current' and 'peak' are in bytes
                gpu_details = tf.config.experimental.get_memory_info('GPU:0')
                current_mb = gpu_details['current'] / (1024 ** 2) # Convert bytes to MB
                peak_mb = gpu_details['peak'] / (1024 ** 2) # Convert bytes to MB
                print(f"GPU Memory (GPU:0) - Current: {current_mb:.2f}MB, Peak: {peak_mb:.2f}MB")
                return current_mb, peak_mb
            except Exception as e:
                print(f"Could not retrieve GPU memory info: {e}")
                return None, None
        print("No GPU detected for memory monitoring.")
        return None, None
    
    def save_models(self, base_path: str | Path):
        """
        Saves the trained presence and density models to the specified directory.

        Args:
            base_path (str | Path): The directory where models will be saved.
        """
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True, parents=True) # Ensure parent directories are created
        
        if self.presence_model:
            model_path = base_path / 'presence_model.h5'
            self.presence_model.save(model_path)
            print(f"Presence model saved to {model_path}")
        if self.density_model:
            model_path = base_path / 'density_model.h5'
            self.density_model.save(model_path)
            print(f"Density model saved to {model_path}")
        
    def load_models(self, base_path: str | Path):
        """
        Loads the presence and density models from the specified directory.

        Args:
            base_path (str | Path): The directory from which models will be loaded.
        """
        base_path = Path(base_path)
        
        presence_path = base_path / 'presence_model.h5'
        density_path = base_path / 'density_model.h5'
        
        if presence_path.exists():
            try:
                self.presence_model = keras.models.load_model(presence_path)
                print(f"Presence model loaded from {presence_path}")
            except Exception as e:
                print(f"Error loading presence model from {presence_path}: {e}")
        else:
            print(f"Presence model not found at {presence_path}")
        
        if density_path.exists():
            try:
                self.density_model = keras.models.load_model(density_path)
                print(f"Density model loaded from {density_path}")
            except Exception as e:
                print(f"Error loading density model from {density_path}: {e}")
        else:
            print(f"Density model not found at {density_path}")

def load_custom_dataset(image_dir: str, label_csv: str, detector: OptimizedAerialCrowdDetector) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a custom dataset from a directory of images and a CSV label file.

    Args:
        image_dir (str): Path to the directory containing image files.
        label_csv (str): Path to the CSV file with 'filename', 'presence', 'density' columns.
        detector (OptimizedAerialCrowdDetector): An instance of the detector for preprocessing.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing
                                                    (X_images, y_presence, y_density) arrays.
    """
    import pandas as pd

    if not Path(image_dir).is_dir():
        print(f"Warning: Image directory not found: {image_dir}. Using synthetic data if no custom data loaded.")
        return np.array([]), np.array([]), np.array([])
    if not Path(label_csv).is_file():
        print(f"Warning: Label CSV file not found: {label_csv}. Using synthetic data if no custom data loaded.")
        return np.array([]), np.array([]), np.array([])

    try:
        df = pd.read_csv(label_csv)  # CSV with columns: filename,presence,density
    except Exception as e:
        print(f"Error reading label CSV {label_csv}: {e}")
        return np.array([]), np.array([]), np.array([])

    X, y_presence, y_density = [], [], []
    skipped_images = 0

    print(f"Loading custom dataset from {image_dir} and {label_csv}...")
    for index, row in df.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        try:
            img = detector.preprocess_image(img_path)
            X.append(img)
            y_presence.append(float(row['presence']))
            y_density.append(float(row['density']))
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1} images...")
        except Exception as e:
            print(f"Error loading or preprocessing {img_path}: {e}")
            skipped_images += 1
            if skipped_images > 10 and skipped_images % 10 == 0:
                print(f"Warning: {skipped_images} images skipped so far due to errors.")

    if skipped_images > 0:
        print(f"Completed loading. {skipped_images} images were skipped due to errors.")
    else:
        print("All images processed successfully.")

    if not X: # Handle case where no images were loaded
        return np.array([]), np.array([]), np.array([])

    return np.array(X), np.array(y_presence), np.array(y_density)


def plot_training_history(presence_history: keras.callbacks.History, density_history: keras.callbacks.History, save_path: str = None):
    """
    Plots and optionally saves the training history for both presence and density models.

    Args:
        presence_history (keras.callbacks.History): History object for the presence model.
        density_history (keras.callbacks.History): History object for the density model.
        save_path (str, optional): Directory to save the plots. If None, plots are displayed.
    """
    plt.figure(figsize=(14, 10)) # Increased figure size

    # Presence Model Loss
    plt.subplot(2, 2, 1)
    plt.plot(presence_history.history['loss'], label='Training Loss')
    plt.plot(presence_history.history['val_loss'], label='Validation Loss')
    plt.title('Presence Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Presence Model Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(presence_history.history['accuracy'], label='Training Accuracy')
    plt.plot(presence_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Presence Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Density Model Loss
    plt.subplot(2, 2, 3)
    plt.plot(density_history.history['loss'], label='Training Loss')
    plt.plot(density_history.history['val_loss'], label='Validation Loss')
    plt.title('Density Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Density Model MAE
    plt.subplot(2, 2, 4)
    plt.plot(density_history.history['mae'], label='Training MAE')
    plt.plot(density_history.history['val_mae'], label='Validation MAE')
    plt.title('Density Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        Path(save_path).mkdir(exist_ok=True, parents=True)
        plt.savefig(Path(save_path) / 'training_history.png')
        print(f"Training history plot saved to {Path(save_path) / 'training_history.png'}")
    else:
        plt.show()

def main_optimized(use_synthetic_data: bool = True, epochs: int = 25):
    """
    Main function for running the optimized aerial crowd detection pipeline.

    Args:
        use_synthetic_data (bool): If True, synthetic data will be generated.
                                   If False, attempts to load from 'path/to/your/images' and 'path/to/your/labels.csv'.
        epochs (int): Number of epochs to train the models.
    """
    print("=== Optimized Aerial Crowd Detection for RTX 3050 ===")
    
    detector = OptimizedAerialCrowdDetector(use_mixed_precision=True)
    
    detector.monitor_gpu_memory()
    
    # Initialize lightweight YOLO if available
    if YOLO_AVAILABLE:
        detector.initialize_yolo_lightweight()
    
    # Optional: Initialize DETR
    if DETR_AVAILABLE:
        detector.initialize_detr()

    X_train, y_presence, y_density = np.array([]), np.array([]), np.array([])

    if use_synthetic_data:
        print("\nCreating synthetic training data...")
        X_train, y_presence, y_density = detector.create_memory_efficient_training_data(num_samples=detector.max_samples)
    else:
        print("\nAttempting to load custom dataset...")
        # !!! IMPORTANT: Replace these paths with your actual dataset paths !!!
        image_dir_path = 'path/to/your/images'
        label_csv_path = 'path/to/your/labels.csv'
        X_train, y_presence, y_density = load_custom_dataset(image_dir_path, label_csv_path, detector)
        
        if X_train.size == 0:
            print("No custom data loaded or an error occurred. Falling back to synthetic data generation.")
            X_train, y_presence, y_density = detector.create_memory_efficient_training_data(num_samples=detector.max_samples)
            use_synthetic_data = True # Update flag to reflect actual data source


    if X_train.size == 0:
        print("No training data available. Exiting.")
        return None

    print(f"\nTraining data statistics:")
    print(f"- Total samples: {len(X_train)}")
    print(f"- Positive (crowd present) samples: {int(np.sum(y_presence))}")
    print(f"- Negative (crowd absent) samples: {int(len(y_presence) - np.sum(y_presence))}")
    print(f"- Average density (for present crowds): {np.mean(y_density[y_presence == 1]):.3f}" if np.sum(y_presence) > 0 else "- Average density: N/A (no crowd present samples)")
    print(f"- Approximate memory usage of X_train: {X_train.nbytes / (1024 ** 2):.1f} MB")
    
    detector.monitor_gpu_memory()
    
    print("\nTraining optimized models...")
    try:
        presence_history, density_history = detector.train_models_optimized(
            X_train, y_presence, y_density, epochs=epochs
        )
        
        plot_training_history(presence_history, density_history, save_path='training_plots')
        
        print("\nTesting predictions...")
        # Ensure there are enough samples to pick from
        num_test_samples = min(3, len(X_train))
        if num_test_samples > 0:
            test_indices = np.random.choice(len(X_train), num_test_samples, replace=False)
            test_images = X_train[test_indices]
            
            predictions = detector.predict_batch_optimized(test_images)
            
            for i, (idx, pred) in enumerate(zip(test_indices, predictions)):
                true_presence = y_presence[idx]
                true_density = y_density[idx]
                
                print(f"\nTest {i+1} (Original index: {idx}):")
                print(f"  True: Presence={true_presence}, Density={true_density:.3f}")
                print(f"  Predicted: Presence={'True' if pred.get('presence_detected') else 'False'}, "
                      f"Density={pred.get('density_score', 0.0):.3f}")
        else:
            print("Not enough training samples to perform test predictions.")
        
        detector.save_models('optimized_models')
        
        detector.monitor_gpu_memory()
        
        print("\nOptimized training completed successfully!")
        print("\nOptimizations applied:")
        print("✓ Mixed precision training")
        print("✓ Memory-efficient data generation (for synthetic data)")
        print("✓ Smaller batch sizes for training and inference")
        print("✓ Lightweight model architectures (CNN, MobileNetV2)")
        print("✓ GPU memory monitoring")
        # Removed "Gradient accumulation support" as it's not explicitly implemented/demonstrated beyond batching.
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for better debugging
        print("Consider reducing batch size or number of samples if running out of memory.")
    
    return detector


if __name__ == "__main__":
    # Set memory allocation strategy (redundant with set_memory_growth but harmless)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Example usage: Run with synthetic data for demonstration
    detector_instance = main_optimized(use_synthetic_data=True, epochs=20) 
    
    # To run with custom data, set use_synthetic_data=False and update paths in main_optimized
    # detector_instance = main_optimized(use_synthetic_data=False, epochs=20)
