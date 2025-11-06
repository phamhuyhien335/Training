"""
Model evaluation script with comprehensive metrics and visualizations.
"""

import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import tensorflow as tf

import config
from utils import load_labels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_dataset(test_dir: str, batch_size: int = 32):
    """
    Load test dataset for evaluation.
    
    Args:
        test_dir: Path to test data directory
        batch_size: Batch size for loading data
        
    Returns:
        Test dataset and class names
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=config.IMG_SHAPE,
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = test_ds.class_names
    return test_ds, class_names


def evaluate_model(model, test_ds):
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained Keras model
        test_ds: Test dataset
        
    Returns:
        Loss and accuracy values
    """
    logger.info("Evaluating model on test set...")
    loss, accuracy = model.evaluate(test_ds)
    
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return loss, accuracy


def get_predictions_and_labels(model, test_ds):
    """
    Get all predictions and true labels from test dataset.
    
    Args:
        model: Trained Keras model
        test_ds: Test dataset
        
    Returns:
        Tuple of (y_true, y_pred, y_pred_probs)
    """
    logger.info("Generating predictions...")
    
    y_true = []
    y_pred = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
        y_pred_probs.extend(predictions)
    
    return np.array(y_true), np.array(y_pred), np.array(y_pred_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """
    Plot per-class accuracy bar chart.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import accuracy_score
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0)
    
    # Create bar plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(class_names)), class_accuracies, color='steelblue')
    
    # Color bars based on performance
    for i, bar in enumerate(bars):
        if class_accuracies[i] >= 0.9:
            bar.set_color('green')
        elif class_accuracies[i] >= 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Per-class accuracy plot saved to {save_path}")
    
    plt.show()
    
    return class_accuracies


def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Calculate and print overall metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    print("\n" + "="*60)
    print("OVERALL METRICS (Weighted Average)")
    print("="*60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60 + "\n")


def plot_confidence_distribution(y_pred_probs, y_true, y_pred, save_path=None):
    """
    Plot distribution of prediction confidences.
    
    Args:
        y_pred_probs: Prediction probabilities
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    # Get max probabilities (confidence scores)
    confidences = np.max(y_pred_probs, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = y_true == y_pred
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=50, alpha=0.7, color='green', label='Correct')
    plt.hist(incorrect_confidences, bins=50, alpha=0.7, color='red', label='Incorrect')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(
        [correct_confidences, incorrect_confidences],
        labels=['Correct', 'Incorrect'],
        patch_artist=True
    )
    plt.ylabel('Confidence Score', fontsize=12)
    plt.title('Confidence Box Plot', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confidence distribution plot saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("CONFIDENCE STATISTICS")
    print("="*60)
    print(f"Correct Predictions:")
    print(f"  Mean confidence: {np.mean(correct_confidences):.4f}")
    print(f"  Median confidence: {np.median(correct_confidences):.4f}")
    print(f"  Min confidence: {np.min(correct_confidences):.4f}")
    print(f"\nIncorrect Predictions:")
    print(f"  Mean confidence: {np.mean(incorrect_confidences):.4f}")
    print(f"  Median confidence: {np.median(incorrect_confidences):.4f}")
    print(f"  Max confidence: {np.max(incorrect_confidences):.4f}")
    print("="*60 + "\n")


def main():
    """
    Main evaluation function.
    """
    parser = argparse.ArgumentParser(description='Evaluate Plant Disease Detection Model')
    parser.add_argument(
        '--model',
        type=str,
        default=config.FINAL_MODEL_PATH,
        help='Path to trained model (.h5 file)'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default=config.TEST_DIR,
        help='Path to test data directory'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=config.LABELS_PATH,
        help='Path to labels file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model)
    
    # Load test dataset
    logger.info(f"Loading test dataset from {args.test_dir}")
    test_ds, class_names = load_test_dataset(args.test_dir, args.batch_size)
    
    logger.info(f"Number of classes: {len(class_names)}")
    logger.info(f"Class names: {class_names}")
    
    # Evaluate model
    loss, accuracy = evaluate_model(model, test_ds)
    
    # Get predictions
    y_true, y_pred, y_pred_probs = get_predictions_and_labels(model, test_ds)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)
    
    # Plot per-class accuracy
    acc_path = os.path.join(args.output_dir, 'per_class_accuracy.png')
    class_accuracies = plot_per_class_accuracy(y_true, y_pred, class_names, save_path=acc_path)
    
    # Print classification report
    print_classification_report(y_true, y_pred, class_names)
    
    # Plot confidence distribution
    conf_path = os.path.join(args.output_dir, 'confidence_distribution.png')
    plot_confidence_distribution(y_pred_probs, y_true, y_pred, save_path=conf_path)
    
    # Save summary to text file
    summary_path = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Directory: {args.test_dir}\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Total Test Samples: {len(y_true)}\n\n")
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Per-Class Accuracy:\n")
        for i, (name, acc) in enumerate(zip(class_names, class_accuracies)):
            f.write(f"  {name}: {acc:.4f} ({acc*100:.2f}%)\n")
    
    logger.info(f"Evaluation summary saved to {summary_path}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
