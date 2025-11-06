"""
Utility functions for the Plant Disease Detection project.
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Dict, Optional
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_gpu():
    """
    Configure GPU settings for optimal performance.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    else:
        logger.info("No GPU found. Training will use CPU.")


def count_images_in_directory(directory: str) -> Dict[str, int]:
    """
    Count the number of images in each subdirectory.
    
    Args:
        directory: Path to the parent directory containing class subdirectories
        
    Returns:
        Dictionary mapping class names to image counts
    """
    counts = {}
    
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return counts
    
    for entry in os.listdir(directory):
        folder_path = os.path.join(directory, entry)
        
        if os.path.isdir(folder_path):
            file_count = sum(
                1 for item in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, item))
            )
            counts[entry] = file_count
    
    return counts


def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Load and preprocess an image for model inference.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32)
        
        if normalize:
            img_array = img_array / 255.0
        
        return np.expand_dims(img_array, axis=0)
    
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise


def predict_image(
    model,
    image_path: str,
    labels: List[str],
    top_k: int = 3
) -> Dict[str, any]:
    """
    Predict the class of an image using the trained model.
    
    Args:
        model: Trained Keras model or TFLite interpreter
        image_path: Path to the image to classify
        labels: List of class labels
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary containing predictions and confidence scores
    """
    # Load and preprocess image
    img_array = load_and_preprocess_image(image_path)
    
    # Get predictions
    if isinstance(model, tf.keras.Model):
        predictions = model.predict(img_array)[0]
    else:
        # Handle TFLite interpreter
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], img_array)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])[0]
    
    # Get top k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    
    results = {
        "image_path": image_path,
        "predictions": [
            {
                "label": labels[idx],
                "confidence": float(predictions[idx])
            }
            for idx in top_indices
        ],
        "top_prediction": labels[top_indices[0]],
        "top_confidence": float(predictions[top_indices[0]])
    }
    
    return results


def plot_training_history(
    history,
    save_path: Optional[str] = None
):
    """
    Plot training history including accuracy and loss curves.
    
    Args:
        history: Keras training history object
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def save_labels(labels: List[str], filepath: str):
    """
    Save class labels to a text file.
    
    Args:
        labels: List of class labels
        filepath: Path to save the labels file
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for label in labels:
                f.write(label + '\n')
        logger.info(f"Labels saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving labels: {e}")
        raise


def load_labels(filepath: str) -> List[str]:
    """
    Load class labels from a text file.
    
    Args:
        filepath: Path to the labels file
        
    Returns:
        List of class labels
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(labels)} labels from {filepath}")
        return labels
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        raise


def convert_to_tflite(
    model,
    output_path: str,
    quantize: bool = False
):
    """
    Convert a Keras model to TensorFlow Lite format.
    
    Args:
        model: Keras model to convert
        output_path: Path to save the TFLite model
        quantize: Whether to apply post-training quantization
    """
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            logger.info("Applying post-training quantization")
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"TFLite model saved to {output_path} (Size: {model_size_mb:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Error converting model to TFLite: {e}")
        raise


def create_confusion_matrix(
    model,
    test_dataset,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Create and plot confusion matrix for model evaluation.
    
    Args:
        model: Trained Keras model
        test_dataset: Test dataset
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def get_model_summary_as_string(model) -> str:
    """
    Get model summary as a string.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary as string
    """
    from io import StringIO
    import sys
    
    stream = StringIO()
    old_stdout = sys.stdout
    sys.stdout = stream
    model.summary()
    sys.stdout = old_stdout
    
    return stream.getvalue()
