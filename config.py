"""
Configuration file for the Plant Disease Detection project.
Modify these settings according to your environment and requirements.
"""

import os

# ==============================
# DATA PATHS
# ==============================
# Base directory for the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories - modify these to match your local setup
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "color")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "sampled")

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, TRAIN_DIR, TEST_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==============================
# MODEL PATHS
# ==============================
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "plant_model_final.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "plant_model.tflite")
LABELS_PATH = os.path.join(PROJECT_ROOT, "labels.txt")

# ==============================
# TRAINING PARAMETERS
# ==============================
# Image configuration
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
IMG_CHANNELS = 3

# Training parameters
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-4

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Sampling configuration for balanced dataset
TARGET_TRAIN_COUNT = 500
TARGET_TEST_COUNT = 100

# ==============================
# MODEL ARCHITECTURE
# ==============================
# Base model
BASE_MODEL_NAME = "MobileNetV2"
BASE_MODEL_WEIGHTS = "imagenet"
INCLUDE_TOP = False

# Fine-tuning configuration
TRAINABLE_BASE = True
FREEZE_LAYERS = 50  # Number of layers to freeze from the beginning

# Classification head
DROPOUT_RATE = 0.3
ACTIVATION = "softmax"

# ==============================
# DATA AUGMENTATION
# ==============================
# Augmentation parameters
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
ROTATION_RANGE = 0.25
ZOOM_RANGE = 0.25
CONTRAST_RANGE = 0.25
BRIGHTNESS_RANGE = 0.25

# ==============================
# CALLBACKS
# ==============================
# Early stopping
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MONITOR = "val_loss"

# Model checkpoint
CHECKPOINT_MONITOR = "val_accuracy"
CHECKPOINT_MODE = "max"
CHECKPOINT_SAVE_BEST_ONLY = True

# ==============================
# INFERENCE
# ==============================
# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.5
TOP_K_PREDICTIONS = 3

# ==============================
# PLANT DISEASE CLASSES
# ==============================
# Classes to process (can be customized)
CLASSES_TO_PROCESS = {
    "Apple": [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___healthy"
    ],
    "Corn (Maize)": [
        "Corn_(maize)___Common_rust",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy"
    ],
    "Grape": [
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___healthy"
    ],
    "Potato": [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy"
    ],
    "Tomato": [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___healthy"
    ]
}

# ==============================
# LOGGING
# ==============================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(PROJECT_ROOT, "training.log")

# ==============================
# TENSORFLOW SETTINGS
# ==============================
# Enable mixed precision for faster training on compatible GPUs
MIXED_PRECISION = False

# TensorFlow autotune
AUTOTUNE = -1  # Use tf.data.AUTOTUNE

# Random seed for reproducibility
RANDOM_SEED = 42
