# Contributing to Plant Disease Detection

Thank you for considering contributing to this project! We welcome contributions from everyone.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Submitting Changes](#submitting-changes)
6. [Reporting Bugs](#reporting-bugs)
7. [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Report bugs you encounter
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit bug fixes or new features
4. **Documentation**: Improve or add documentation
5. **Dataset Contributions**: Share additional plant disease images
6. **Model Improvements**: Experiment with different architectures
7. **Testing**: Add or improve tests

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/Training.git
cd Training
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/your-bugfix-name
```

### 3. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Make Your Changes

- Write clear, commented code
- Follow the existing code style
- Add tests if applicable
- Update documentation as needed

## Coding Standards

### Python Style Guide

We follow PEP 8 guidelines with some modifications:

- **Line Length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Group imports (standard library, third-party, local)
- **Docstrings**: Use triple quotes for all docstrings
- **Comments**: Write clear, concise comments

### Example Code Style

```python
"""
Module docstring explaining purpose.
"""

import os
import sys

import numpy as np
import tensorflow as tf

from utils import load_model


def example_function(param1: str, param2: int) -> bool:
    """
    Function docstring explaining what it does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
    # Implementation
    pass
```

### Code Quality Tools

Before submitting, run these tools:

```bash
# Format code with black (if available)
black *.py

# Check with flake8 (if available)
flake8 *.py

# Type checking with mypy (if available)
mypy *.py
```

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

```bash
# Good commit messages:
git commit -m "Add data augmentation for training"
git commit -m "Fix bug in prediction confidence calculation"
git commit -m "Update README with installation instructions"

# Bad commit messages:
git commit -m "Update"
git commit -m "Fix stuff"
git commit -m "Changes"
```

### Pull Request Process

1. **Update Your Fork**

```bash
git fetch upstream
git rebase upstream/main
```

2. **Push Your Changes**

```bash
git push origin feature/your-feature-name
```

3. **Create Pull Request**

- Go to the original repository on GitHub
- Click "New Pull Request"
- Select your fork and branch
- Fill in the PR template with:
  - Clear description of changes
  - Related issue numbers
  - Testing performed
  - Screenshots (for UI changes)

4. **PR Review Process**

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows the style guidelines
- [ ] Comments and docstrings are added/updated
- [ ] Documentation is updated if needed
- [ ] Tests pass (if applicable)
- [ ] No unnecessary files are included
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes

## Reporting Bugs

### Before Submitting a Bug Report

- Check if the bug has already been reported
- Try to reproduce the bug with the latest version
- Collect relevant information (error messages, logs, etc.)

### How to Submit a Bug Report

Create an issue with the following information:

**Title**: Brief, descriptive title

**Description**:
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Steps to Reproduce**: Detailed steps to reproduce the bug
- **Environment**:
  - OS and version
  - Python version
  - TensorFlow version
  - Other relevant package versions
- **Error Messages**: Full error messages and stack traces
- **Screenshots**: If applicable

**Example Bug Report**:

```markdown
## Bug: Model fails to load TFLite file

**Expected Behavior**: 
Model should load successfully from TFLite file.

**Actual Behavior**: 
Program crashes with "Invalid model file" error.

**Steps to Reproduce**:
1. Run `python predict.py --image test.jpg --model plant_model.tflite`
2. Error occurs immediately

**Environment**:
- OS: Windows 10
- Python: 3.9.7
- TensorFlow: 2.10.0

**Error Message**:
```
ValueError: Invalid model file format
```

**Additional Context**:
This happens with the TFLite model but not with the H5 model.
```

## Suggesting Enhancements

### Before Submitting an Enhancement

- Check if the enhancement has already been suggested
- Consider if it fits the project scope
- Think about how it would benefit users

### How to Submit an Enhancement

Create an issue with the following:

**Title**: Clear, descriptive title

**Description**:
- **Problem**: What problem does this solve?
- **Proposed Solution**: Your suggested implementation
- **Alternatives**: Other approaches you considered
- **Benefits**: Why this would be valuable
- **Implementation Details**: Technical details if applicable

**Example Enhancement Request**:

```markdown
## Enhancement: Add support for video prediction

**Problem**: 
Currently, the model only processes individual images. Users want to process video files.

**Proposed Solution**:
Add a `predict_video.py` script that:
- Extracts frames from video
- Processes each frame
- Outputs results with timestamps

**Alternatives**:
- Real-time webcam prediction
- Batch image processing from video

**Benefits**:
- Process multiple images efficiently
- Monitor plant health over time
- Real-time disease detection

**Implementation**:
- Use OpenCV for video processing
- Add progress bar for feedback
- Save results to CSV or JSON
```

## Dataset Contributions

If you have plant disease images to contribute:

1. **Image Requirements**:
   - High quality (minimum 224x224 pixels)
   - Clear focus on affected plant parts
   - Good lighting conditions
   - Properly labeled

2. **Format**:
   - JPEG or PNG
   - Organized by disease class
   - Include metadata if available

3. **Licensing**:
   - Ensure you have rights to share images
   - Specify license for your contributions

4. **Submission**:
   - Create an issue describing the dataset
   - Provide download link or contact for large datasets
   - Include description and source information

## Model Improvements

Contributions for model improvements:

1. **Architecture Changes**:
   - Document the architecture modifications
   - Provide training results comparison
   - Include code for the new architecture

2. **Hyperparameter Tuning**:
   - Share your findings
   - Include training curves and metrics
   - Document the parameter search process

3. **New Features**:
   - Explain the feature engineering
   - Show performance improvements
   - Provide implementation code

## Questions?

If you have questions about contributing:

- Open an issue with the "question" label
- Provide context about what you're trying to do
- Check existing documentation first

## Recognition

All contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Plant Disease Detection! 🌿
