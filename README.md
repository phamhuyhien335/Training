# Plant Disease Detection

A deep learning project for detecting plant diseases using TensorFlow and MobileNetV2. This project can classify 15 different types of plant diseases across multiple plant species including Apple, Corn, Grape, Potato, and Tomato.

## 🌿 Overview

This project uses transfer learning with MobileNetV2 to detect plant diseases from leaf images. The model is trained on the PlantVillage dataset and can be deployed on mobile devices using TensorFlow Lite.

## 📋 Features

- **15 Plant Disease Classes**: Detects diseases in Apple, Corn, Grape, Potato, and Tomato plants
- **Transfer Learning**: Uses pre-trained MobileNetV2 for efficient training
- **Data Augmentation**: Includes random flips, rotations, zoom, contrast, and brightness adjustments
- **Model Export**: Supports both H5 and TensorFlow Lite formats for deployment
- **Cross-platform**: Works on Windows, Linux, and macOS

## 🗂️ Project Structure

```
Training/
├── Train.ipynb              # Main training notebook
├── Train_Colab.ipynb        # Google Colab training notebook
├── Test.ipynb               # Model testing and inference
├── best_model.h5            # Best model checkpoint during training
├── plant_model_final.h5     # Final trained model (H5 format)
├── plant_model.tflite       # TensorFlow Lite model for mobile deployment
├── labels.txt               # Class labels
├── colab results/           # Training results from Colab
└── README.md                # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- CUDA-compatible GPU (optional, but recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/phamhuyhien335/Training.git
cd Training
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses the PlantVillage dataset with the following disease classes:

1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___healthy
4. Corn_(maize)___Common_rust
5. Corn_(maize)___Northern_Leaf_Blight
6. Corn_(maize)___healthy
7. Grape___Black_rot
8. Grape___Esca_(Black_Measles)
9. Grape___healthy
10. Potato___Early_blight
11. Potato___Late_blight
12. Potato___healthy
13. Tomato___Bacterial_spot
14. Tomato___Early_blight
15. Tomato___healthy

### Data Preparation

The notebooks include code to:
- Count images in each class
- Split data into training and testing sets
- Apply data augmentation techniques
- Handle class imbalance through sampling

## 🏋️ Training

### Local Training

1. Prepare your dataset in the following structure:
```
data/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ...
└── test/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    └── ...
```

2. Open and run `Train.ipynb` in Jupyter:
```bash
jupyter notebook Train.ipynb
```

3. Update the data paths in the notebook to match your local setup.

### Google Colab Training

For training on Google Colab with GPU support:

1. Open `Train_Colab.ipynb` in Google Colab
2. Mount your Google Drive
3. Update the paths to your dataset in Drive
4. Run all cells to train the model

### Training Parameters

- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Epochs**: 40 (with early stopping)
- **Optimizer**: Adam (learning rate: 1e-4)
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Fine-tuning**: All layers unfrozen except first 50

## 🧪 Testing

### Using TensorFlow Lite Model

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
interpreter = tf.lite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Prepare image
img = Image.open("path/to/image.jpg").convert("RGB").resize((224, 224))
input_data = np.expand_dims(np.float32(img) / 255.0, axis=0)

# Predict
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# Get results
pred_idx = np.argmax(output_data)
confidence = output_data[pred_idx]
print(f"Predicted: {labels[pred_idx]} (confidence: {confidence:.3f})")
```

### Using H5 Model

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model("plant_model_final.h5")

# Prepare and predict
img = Image.open("path/to/image.jpg").convert("RGB").resize((224, 224))
arr = np.expand_dims(np.float32(img) / 255.0, axis=0)
pred = model.predict(arr)

idx = np.argmax(pred)
confidence = np.max(pred)
print(f"Predicted class: {idx}, Confidence: {confidence:.4f}")
```

## 📈 Model Performance

The model uses:
- **Data Augmentation**: Random flip, rotation, zoom, contrast, and brightness
- **Regularization**: Dropout (0.3) and early stopping
- **Transfer Learning**: Fine-tuned MobileNetV2
- **Callbacks**: ModelCheckpoint and EarlyStopping

## 🔧 Model Architecture

```
Input (224x224x3)
    ↓
Rescaling (1./255)
    ↓
MobileNetV2 (pre-trained, partially frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (num_classes, softmax)
```

## 📱 Deployment

The TensorFlow Lite model (`plant_model.tflite`) can be deployed on:
- Android apps
- iOS apps
- Edge devices
- Web applications (using TensorFlow.js)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👥 Authors

- phamhuyhien335

## 🙏 Acknowledgments

- PlantVillage dataset
- TensorFlow team
- MobileNetV2 architecture
