"""
Training script for Plant Disease Detection model.
This script can be run from the command line for automated training.
"""

import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from utils import (
    setup_gpu,
    plot_training_history,
    save_labels,
    convert_to_tflite,
    get_model_summary_as_string
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_data_augmentation():
    """
    Create data augmentation pipeline.
    
    Returns:
        Keras Sequential model with augmentation layers
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(config.ROTATION_RANGE),
        layers.RandomZoom(config.ZOOM_RANGE),
        layers.RandomContrast(config.CONTRAST_RANGE),
        layers.RandomBrightness(config.BRIGHTNESS_RANGE),
    ], name="data_augmentation")


def load_datasets(train_dir: str, test_dir: str):
    """
    Load training and validation datasets.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test/validation data directory
        
    Returns:
        Tuple of (train_ds, val_ds, class_names)
    """
    logger.info("Loading datasets...")
    
    # Load training dataset
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=config.IMG_SHAPE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    # Load validation dataset
    val_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=config.IMG_SHAPE,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    
    return train_ds, val_ds, class_names


def build_model(num_classes: int):
    """
    Build the plant disease detection model.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    logger.info("Building model...")
    
    # Load base model
    base_model = keras.applications.MobileNetV2(
        input_shape=config.IMG_SHAPE + (config.IMG_CHANNELS,),
        include_top=config.INCLUDE_TOP,
        weights=config.BASE_MODEL_WEIGHTS
    )
    
    # Configure fine-tuning
    base_model.trainable = config.TRAINABLE_BASE
    
    if config.TRAINABLE_BASE and config.FREEZE_LAYERS > 0:
        for layer in base_model.layers[:config.FREEZE_LAYERS]:
            layer.trainable = False
        logger.info(f"Frozen first {config.FREEZE_LAYERS} layers of base model")
    
    # Build full model
    inputs = keras.Input(shape=config.IMG_SHAPE + (config.IMG_CHANNELS,))
    x = layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=config.TRAINABLE_BASE)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation=config.ACTIVATION)(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    logger.info("Model built successfully")
    logger.info(f"\n{get_model_summary_as_string(model)}")
    
    return model


def train_model(
    model,
    train_ds,
    val_ds,
    epochs: int = None
):
    """
    Train the model with callbacks.
    
    Args:
        model: Compiled Keras model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of training epochs (uses config value if None)
        
    Returns:
        Training history object
    """
    if epochs is None:
        epochs = config.EPOCHS
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    # Configure callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.CHECKPOINT_PATH,
            monitor=config.CHECKPOINT_MONITOR,
            save_best_only=config.CHECKPOINT_SAVE_BEST_ONLY,
            mode=config.CHECKPOINT_MODE,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training completed!")
    
    return history


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument(
        '--train-dir',
        type=str,
        default=config.TRAIN_DIR,
        help='Path to training data directory'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default=config.TEST_DIR,
        help='Path to test/validation data directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.BATCH_SIZE,
        help='Batch size for training'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=config.MODEL_DIR,
        help='Directory to save trained models'
    )
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.BATCH_SIZE = args.batch_size
    
    # Setup GPU
    setup_gpu()
    
    # Load datasets
    train_ds, val_ds, class_names = load_datasets(args.train_dir, args.test_dir)
    
    # Apply data augmentation to training set
    data_augmentation = create_data_augmentation()
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    # Optimize datasets
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Build model
    num_classes = len(class_names)
    model = build_model(num_classes)
    
    # Train model
    history = train_model(model, train_ds, val_ds, epochs=args.epochs)
    
    # Save final model
    logger.info(f"Saving final model to {config.FINAL_MODEL_PATH}")
    model.save(config.FINAL_MODEL_PATH)
    
    # Convert to TFLite
    logger.info("Converting model to TensorFlow Lite format...")
    convert_to_tflite(model, config.TFLITE_MODEL_PATH)
    
    # Save labels
    logger.info(f"Saving labels to {config.LABELS_PATH}")
    save_labels(class_names, config.LABELS_PATH)
    
    # Plot training history
    logger.info("Plotting training history...")
    plot_path = os.path.join(args.output_dir, "training_history.png")
    plot_training_history(history, save_path=plot_path)
    
    # Evaluate model
    logger.info("Evaluating model on validation set...")
    loss, accuracy = model.evaluate(val_ds)
    logger.info(f"Final validation loss: {loss:.4f}")
    logger.info(f"Final validation accuracy: {accuracy:.4f}")
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
