# Quick Start Guide

Get started with Plant Disease Detection in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA-compatible GPU for faster training

## Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/phamhuyhien335/Training.git
cd Training

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Your Data

Organize your plant disease images:

```
your_data/
└── color/
    ├── Apple___Apple_scab/
    │   └── *.jpg
    ├── Apple___Black_rot/
    │   └── *.jpg
    └── ...
```

Run data preparation:

```bash
python prepare_data.py \
    --source-dir path/to/your_data/color \
    --train-count 500 \
    --test-count 100
```

This will create organized `data/train` and `data/test` directories.

## Step 3: Train the Model

```bash
python train.py \
    --train-dir data/train \
    --test-dir data/test \
    --epochs 40
```

Training will take 30-60 minutes depending on your hardware. The script will:
- Load and preprocess data
- Build the model
- Train with data augmentation
- Save the best model
- Generate training visualizations

## Step 4: Evaluate the Model

```bash
python evaluate.py \
    --model models/plant_model_final.h5 \
    --test-dir data/test
```

This generates:
- Confusion matrix
- Per-class accuracy charts
- Confidence distribution plots
- Detailed classification report

## Step 5: Make Predictions

Predict a single image:

```bash
python predict.py \
    --image path/to/test_image.jpg \
    --model plant_model.tflite
```

Output:
```
📷 Image: test_image.jpg
🔍 Top Prediction: Tomato___Early_blight
✅ Confidence: 0.9542 (95.42%)

🔝 Top 3 Predictions:
   1. Tomato___Early_blight: 0.9542 (95.42%)
   2. Tomato___Late_blight: 0.0321 (3.21%)
   3. Tomato___healthy: 0.0089 (0.89%)
```

## Alternative: Using Jupyter Notebooks

If you prefer Jupyter notebooks:

```bash
# Start Jupyter
jupyter notebook

# Open and run:
# - Train.ipynb for local training
# - Train_Colab.ipynb for Google Colab
# - Test.ipynb for testing
```

## What's Next?

### Learn More
- Read [USAGE.md](USAGE.md) for detailed documentation
- Check [example.py](example.py) for programmatic usage
- See [README.md](README.md) for full documentation

### Customize
- Edit [config.py](config.py) to adjust parameters
- Modify training hyperparameters
- Try different model architectures

### Contribute
- Read [CONTRIBUTING.md](CONTRIBUTING.md)
- Report bugs or suggest features
- Share your improvements

## Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError`
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Problem**: Out of memory during training
```bash
# Solution: Reduce batch size
python train.py --batch-size 16
```

**Problem**: No GPU detected
```bash
# Check TensorFlow GPU installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Problem**: Model file not found
```bash
# Ensure you've completed training first
# Or use existing model files in the repository
```

## Quick Command Reference

```bash
# Data preparation
python prepare_data.py --source-dir DATA_DIR

# Training
python train.py --train-dir data/train --test-dir data/test

# Evaluation
python evaluate.py --model models/plant_model_final.h5

# Prediction (single image)
python predict.py --image IMAGE_PATH

# Prediction (multiple images)
python predict.py --images img1.jpg img2.jpg img3.jpg

# Prediction (directory)
python predict.py --image-dir images/ --output results.json

# Run examples
python example.py
```

## Need Help?

- 📖 Check the [documentation](README.md)
- 🐛 [Report a bug](https://github.com/phamhuyhien335/Training/issues)
- 💡 [Request a feature](https://github.com/phamhuyhien335/Training/issues)
- 📧 Contact the maintainer

Happy training! 🌿
