"""
Example script demonstrating the usage of the Plant Disease Detection system.
This script shows how to use the prediction API programmatically.
"""

import os
import sys
from predict import PlantDiseasePredictor


def example_single_prediction():
    """
    Example: Predict disease from a single image.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Image Prediction")
    print("="*60)
    
    # Initialize predictor with TFLite model
    predictor = PlantDiseasePredictor(
        model_path="plant_model.tflite",
        labels_path="labels.txt",
        model_type="tflite"
    )
    
    # Make prediction
    # Note: Replace with actual image path
    image_path = "path/to/your/test_image.jpg"
    
    if os.path.exists(image_path):
        result = predictor.predict(image_path, top_k=3)
        predictor.print_prediction(result)
    else:
        print(f"⚠️  Image not found: {image_path}")
        print("Please update the image_path variable with a valid image path.")


def example_batch_prediction():
    """
    Example: Predict diseases from multiple images.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Prediction")
    print("="*60)
    
    # Initialize predictor with H5 model
    predictor = PlantDiseasePredictor(
        model_path="plant_model_final.h5",
        labels_path="labels.txt",
        model_type="h5"
    )
    
    # List of images to process
    # Note: Replace with actual image paths
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg"
    ]
    
    # Filter existing images
    existing_images = [img for img in image_paths if os.path.exists(img)]
    
    if existing_images:
        results = predictor.predict_batch(existing_images, top_k=3)
        
        for result in results:
            predictor.print_prediction(result)
    else:
        print("⚠️  No valid images found.")
        print("Please update the image_paths list with valid image paths.")


def example_programmatic_usage():
    """
    Example: Using predictions programmatically in your code.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Programmatic Usage")
    print("="*60)
    
    try:
        predictor = PlantDiseasePredictor(
            model_path="plant_model.tflite",
            labels_path="labels.txt"
        )
        
        image_path = "path/to/test_image.jpg"
        
        if os.path.exists(image_path):
            # Get prediction result
            result = predictor.predict(image_path, top_k=3)
            
            # Access prediction data programmatically
            top_class = result['top_prediction']
            confidence = result['top_confidence']
            
            print(f"\n📊 Programmatic Access:")
            print(f"   Top class: {top_class}")
            print(f"   Confidence: {confidence:.4f}")
            
            # Make decisions based on confidence
            if confidence > 0.9:
                print("   ✅ High confidence - Disease detected with high certainty")
            elif confidence > 0.7:
                print("   ⚠️  Medium confidence - Further inspection recommended")
            else:
                print("   ❌ Low confidence - Uncertain prediction")
            
            # Access all predictions
            print(f"\n   All predictions:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"      {i}. {pred['label']}: {pred['confidence']:.4f}")
        else:
            print(f"⚠️  Image not found: {image_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the model and labels files exist.")


def example_with_custom_threshold():
    """
    Example: Using custom confidence threshold for filtering predictions.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Confidence Threshold")
    print("="*60)
    
    try:
        predictor = PlantDiseasePredictor(
            model_path="plant_model.tflite",
            labels_path="labels.txt"
        )
        
        # Custom threshold
        CONFIDENCE_THRESHOLD = 0.8
        
        image_path = "path/to/test_image.jpg"
        
        if os.path.exists(image_path):
            result = predictor.predict(image_path, top_k=1)
            
            print(f"\n📊 Prediction with threshold {CONFIDENCE_THRESHOLD}:")
            print(f"   Image: {image_path}")
            
            if result['top_confidence'] >= CONFIDENCE_THRESHOLD:
                print(f"   ✅ Accepted: {result['top_prediction']}")
                print(f"   Confidence: {result['top_confidence']:.4f}")
            else:
                print(f"   ❌ Rejected: Confidence too low")
                print(f"   Top prediction: {result['top_prediction']}")
                print(f"   Confidence: {result['top_confidence']:.4f}")
                print(f"   Below threshold: {CONFIDENCE_THRESHOLD}")
        else:
            print(f"⚠️  Image not found: {image_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def print_menu():
    """
    Print example menu.
    """
    print("\n" + "="*60)
    print("Plant Disease Detection - Examples")
    print("="*60)
    print("\nAvailable examples:")
    print("  1. Single image prediction")
    print("  2. Batch prediction")
    print("  3. Programmatic usage")
    print("  4. Custom confidence threshold")
    print("  5. Run all examples")
    print("  0. Exit")
    print("\nNote: Update image paths in the example functions before running.")


def main():
    """
    Main function to run examples.
    """
    print("\n🌿 Plant Disease Detection - Example Script")
    print("="*60)
    print("\nThis script demonstrates how to use the prediction API.")
    print("Please update the image paths in the example functions before running.")
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '0':
            print("\nExiting. Goodbye!")
            break
        elif choice == '1':
            example_single_prediction()
        elif choice == '2':
            example_batch_prediction()
        elif choice == '3':
            example_programmatic_usage()
        elif choice == '4':
            example_with_custom_threshold()
        elif choice == '5':
            example_single_prediction()
            example_batch_prediction()
            example_programmatic_usage()
            example_with_custom_threshold()
        else:
            print("❌ Invalid choice. Please enter a number between 0 and 5.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
