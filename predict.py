"""
Inference script for Plant Disease Detection model.
This script can be used to make predictions on individual images or batches.
"""

import os
import argparse
import logging
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

import config
from utils import (
    load_labels,
    predict_image,
    load_and_preprocess_image
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlantDiseasePredictor:
    """
    Plant Disease Detection predictor class.
    Supports both H5 and TFLite models.
    """
    
    def __init__(self, model_path: str, labels_path: str, model_type: str = "auto"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the model file (.h5 or .tflite)
            labels_path: Path to the labels file
            model_type: Type of model ('h5', 'tflite', or 'auto')
        """
        self.model_path = model_path
        self.labels = load_labels(labels_path)
        
        # Auto-detect model type
        if model_type == "auto":
            if model_path.endswith('.tflite'):
                model_type = "tflite"
            elif model_path.endswith('.h5'):
                model_type = "h5"
            else:
                raise ValueError("Cannot auto-detect model type. Please specify 'h5' or 'tflite'")
        
        self.model_type = model_type
        self.model = self._load_model()
        
        logger.info(f"Loaded {model_type.upper()} model from {model_path}")
        logger.info(f"Number of classes: {len(self.labels)}")
    
    def _load_model(self):
        """Load the model based on type."""
        if self.model_type == "tflite":
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        else:  # h5
            return load_model(self.model_path)
    
    def predict(self, image_path: str, top_k: int = 3) -> dict:
        """
        Predict the disease class for an image.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        return predict_image(self.model, image_path, self.labels, top_k=top_k)
    
    def predict_batch(self, image_paths: list, top_k: int = 3) -> list:
        """
        Predict disease classes for multiple images.
        
        Args:
            image_paths: List of image paths
            top_k: Number of top predictions to return per image
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, top_k=top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e)
                })
        return results
    
    def print_prediction(self, result: dict):
        """
        Pretty print a prediction result.
        
        Args:
            result: Prediction result dictionary
        """
        if "error" in result:
            print(f"\n❌ Error for {result['image_path']}: {result['error']}")
            return
        
        print(f"\n📷 Image: {result['image_path']}")
        print(f"🔍 Top Prediction: {result['top_prediction']}")
        print(f"✅ Confidence: {result['top_confidence']:.4f} ({result['top_confidence']*100:.2f}%)")
        
        if len(result['predictions']) > 1:
            print(f"\n🔝 Top {len(result['predictions'])} Predictions:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"   {i}. {pred['label']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")


def main():
    """
    Main inference function.
    """
    parser = argparse.ArgumentParser(description='Plant Disease Detection Inference')
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image file'
    )
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Paths to multiple image files'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images to process'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=config.TFLITE_MODEL_PATH,
        help='Path to model file (.h5 or .tflite)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=config.LABELS_PATH,
        help='Path to labels file'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='auto',
        choices=['auto', 'h5', 'tflite'],
        help='Type of model file'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=config.TOP_K_PREDICTIONS,
        help='Number of top predictions to show'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save predictions as JSON'
    )
    
    args = parser.parse_args()
    
    # Check if model and labels exist
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.labels):
        logger.error(f"Labels file not found: {args.labels}")
        return
    
    # Initialize predictor
    predictor = PlantDiseasePredictor(
        model_path=args.model,
        labels_path=args.labels,
        model_type=args.model_type
    )
    
    # Collect image paths
    image_paths = []
    
    if args.image:
        image_paths.append(args.image)
    
    if args.images:
        image_paths.extend(args.images)
    
    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            logger.error(f"Directory not found: {args.image_dir}")
            return
        
        # Get all image files from directory
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        for filename in os.listdir(args.image_dir):
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(args.image_dir, filename))
    
    if not image_paths:
        logger.error("No images provided. Use --image, --images, or --image-dir")
        parser.print_help()
        return
    
    logger.info(f"Processing {len(image_paths)} image(s)...")
    
    # Make predictions
    if len(image_paths) == 1:
        result = predictor.predict(image_paths[0], top_k=args.top_k)
        predictor.print_prediction(result)
        results = [result]
    else:
        results = predictor.predict_batch(image_paths, top_k=args.top_k)
        for result in results:
            predictor.print_prediction(result)
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")
    
    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
