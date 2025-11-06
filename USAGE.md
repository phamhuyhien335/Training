# Usage Guide

This guide provides detailed instructions on how to use the Plant Disease Detection scripts.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Inference](#inference)
6. [Using Jupyter Notebooks](#using-jupyter-notebooks)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/phamhuyhien335/Training.git
cd Training
```

### 2. Create a virtual environment (recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

### Organize Your Dataset

Before training, your dataset should be organized with this structure:

```
data/
└── color/
    ├── Apple___Apple_scab/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Apple___Black_rot/
    │   └── ...
    └── ...
```

### Run Data Preparation Script

The `prepare_data.py` script will:
- Sample images from each class
- Split data into train and test sets
- Create proper directory structure

```bash
python prepare_data.py \
    --source-dir /path/to/your/data/color \
    --output-dir data/sampled \
    --train-count 500 \
    --test-count 100
```

**Arguments:**
- `--source-dir`: Directory containing class subdirectories with images
- `--output-dir`: Where to save the organized data (default: data/sampled)
- `--train-count`: Number of images per class for training (default: 500)
- `--test-count`: Number of images per class for testing (default: 100)
- `--classes`: Specific classes to process (optional)
- `--random-seed`: Random seed for reproducibility (default: 42)

### Verify Data Structure

To verify your data without processing:

```bash
python prepare_data.py --verify-only --output-dir data/sampled
```

## Training

### Quick Start Training

Train the model with default parameters:

```bash
python train.py \
    --train-dir data/train \
    --test-dir data/test \
    --epochs 40
```

### Advanced Training Options

```bash
python train.py \
    --train-dir data/train \
    --test-dir data/test \
    --epochs 50 \
    --batch-size 32 \
    --output-dir models
```

**Arguments:**
- `--train-dir`: Path to training data directory
- `--test-dir`: Path to test/validation data directory
- `--epochs`: Number of training epochs (default: 40)
- `--batch-size`: Batch size for training (default: 32)
- `--output-dir`: Directory to save trained models (default: models)

### What Happens During Training

1. **GPU Setup**: Automatically detects and configures GPU if available
2. **Data Loading**: Loads and preprocesses training and validation data
3. **Data Augmentation**: Applies random transformations to training images
4. **Model Building**: Creates MobileNetV2-based model
5. **Training**: Trains with callbacks (early stopping, model checkpoint)
6. **Model Saving**: Saves models in H5 and TFLite formats
7. **Visualization**: Generates training history plots

### Training Output

After training, you'll find:
- `models/best_model.h5` - Best model checkpoint
- `models/plant_model_final.h5` - Final trained model
- `models/plant_model.tflite` - TensorFlow Lite model
- `labels.txt` - Class labels
- `models/training_history.png` - Training curves
- `training.log` - Detailed training log

## Evaluation

### Evaluate Trained Model

After training, evaluate your model on the test set:

```bash
python evaluate.py \
    --model models/plant_model_final.h5 \
    --test-dir data/test \
    --output-dir evaluation_results
```

**Arguments:**
- `--model`: Path to trained model (.h5 file)
- `--test-dir`: Path to test data directory
- `--labels`: Path to labels file (default: labels.txt)
- `--output-dir`: Directory to save evaluation results
- `--batch-size`: Batch size for evaluation (default: 32)

### Evaluation Output

The evaluation script generates:
- **Confusion Matrix** (`confusion_matrix.png`)
- **Per-Class Accuracy** (`per_class_accuracy.png`)
- **Confidence Distribution** (`confidence_distribution.png`)
- **Classification Report** (printed to console)
- **Summary Text File** (`evaluation_summary.txt`)

## Inference

### Predict Single Image

```bash
python predict.py \
    --image path/to/image.jpg \
    --model models/plant_model.tflite \
    --labels labels.txt
```

### Predict Multiple Images

```bash
python predict.py \
    --images image1.jpg image2.jpg image3.jpg \
    --model models/plant_model.tflite
```

### Predict All Images in Directory

```bash
python predict.py \
    --image-dir path/to/images/ \
    --model models/plant_model.tflite \
    --output predictions.json
```

### Using H5 Model Instead of TFLite

```bash
python predict.py \
    --image path/to/image.jpg \
    --model models/plant_model_final.h5 \
    --model-type h5
```

**Arguments:**
- `--image`: Path to a single image
- `--images`: Paths to multiple images
- `--image-dir`: Directory containing images
- `--model`: Path to model file (.h5 or .tflite)
- `--labels`: Path to labels file
- `--model-type`: Type of model ('auto', 'h5', or 'tflite')
- `--top-k`: Number of top predictions to show (default: 3)
- `--output`: Save predictions to JSON file

### Example Prediction Output

```
📷 Image: test_image.jpg
🔍 Top Prediction: Apple___Apple_scab
✅ Confidence: 0.9542 (95.42%)

🔝 Top 3 Predictions:
   1. Apple___Apple_scab: 0.9542 (95.42%)
   2. Apple___Black_rot: 0.0321 (3.21%)
   3. Apple___healthy: 0.0089 (0.89%)
```

## Using Jupyter Notebooks

### Original Notebooks

The project includes three Jupyter notebooks:

1. **Train.ipynb** - Local training notebook
2. **Train_Colab.ipynb** - Google Colab training notebook
3. **Test.ipynb** - Testing and inference notebook

### Start Jupyter

```bash
jupyter notebook
```

### Using Train.ipynb

1. Open `Train.ipynb`
2. Update data paths to match your setup
3. Run cells sequentially
4. Monitor training progress
5. Model will be saved automatically

### Using Train_Colab.ipynb

1. Upload to Google Colab
2. Mount Google Drive
3. Update paths to your Drive folders
4. Enable GPU runtime
5. Run all cells

### Using Test.ipynb

1. Open `Test.ipynb`
2. Update model and image paths
3. Run cells to test predictions
4. View results and confidence scores

## Configuration

### Customizing Parameters

Edit `config.py` to customize:

- Image size and batch size
- Learning rate and epochs
- Model architecture settings
- Data augmentation parameters
- Callbacks configuration

### Example Configuration Changes

```python
# In config.py

# Change image size
IMG_SIZE = 299  # For InceptionV3

# Increase epochs
EPOCHS = 60

# Adjust learning rate
LEARNING_RATE = 5e-5

# Change dropout rate
DROPOUT_RATE = 0.5
```

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
```bash
# Solution: Reduce batch size
python train.py --batch-size 16
```

**Issue**: Model not found
```bash
# Solution: Check model path
python predict.py --model models/plant_model_final.h5
```

**Issue**: CUDA out of memory
```bash
# Solution: Enable memory growth (automatic) or use CPU
# Set GPU memory growth in utils.py:setup_gpu()
```

### Getting Help

If you encounter issues:
1. Check the log files (`training.log`)
2. Verify data directory structure
3. Ensure all dependencies are installed
4. Check Python version (3.8+)
5. Verify TensorFlow installation

## Best Practices

1. **Use Virtual Environment**: Always use a virtual environment
2. **Data Backup**: Keep backup of your original dataset
3. **Version Control**: Track changes to config.py
4. **Monitor Training**: Watch training curves for overfitting
5. **Validate Often**: Use validation set to check performance
6. **Document Changes**: Keep notes on experiments

## Advanced Usage

### Custom Data Augmentation

Modify `train.py` to add custom augmentation:

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    # Add your custom layers here
])
```

### Transfer Learning with Different Base Models

Modify `train.py` to use a different base model:

```python
# Instead of MobileNetV2, use:
base_model = keras.applications.InceptionV3(
    input_shape=config.IMG_SHAPE + (config.IMG_CHANNELS,),
    include_top=False,
    weights="imagenet"
)
```

### Batch Prediction

For processing many images efficiently:

```python
from predict import PlantDiseasePredictor

predictor = PlantDiseasePredictor(
    model_path="models/plant_model.tflite",
    labels_path="labels.txt"
)

image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = predictor.predict_batch(image_paths)

for result in results:
    predictor.print_prediction(result)
```

## Next Steps

After successfully running the scripts:

1. Experiment with different hyperparameters
2. Try different base models
3. Collect more training data
4. Deploy the model to mobile or web
5. Create a REST API for the model
6. Integrate with a web application

For more information, see the main [README.md](README.md).
