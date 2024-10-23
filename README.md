# Fire and Smoke Detection

## Overview

This project implements a fire and smoke detection system using deep learning techniques. The goal is to classify images as either "fire" or "smoke" using a convolutional neural network (CNN).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Pillow
- Scikit-learn

## Installation

Install the required Python packages using `pip`:

```bash
pip install tensorflow keras numpy matplotlib seaborn pillow scikit-learn

git clone https://github.com/Arunkarthi123/fire-smoke-detection.git
cd fire-smoke-detection
pip install -r requirements.txt
```
Usage
Data Cleaning

Before training the model, you should clean the dataset to remove any corrupted or truncated images:
```bash
python clean_data.py

```
Model Training
```bash
python train.py

```

Visualization
```bash
python train.py

```


To view the training history (accuracy and loss) and confusion matrix, run:```bash
python train.py

```

To train the CNN model on your dataset, run the following command:

Evaluation

After training, evaluate the model on the test data and visualize the results:
Visualization

To view the training history (accuracy and loss) and confusion matrix, run:
