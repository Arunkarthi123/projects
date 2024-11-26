# Fire and Smoke Detection Using Convolutional Neural Networks (CNN)

This project implements a deep learning model using Convolutional Neural Networks (CNN) to detect fire and smoke from images. The model is trained to classify images as either containing fire or smoke. This can be useful in real-time fire detection systems for public safety and security applications.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Cleaning](#data-cleaning)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
  - [Classify a Single Image](#classify-a-single-image)
  - [Visualization](#visualization)
- [Results](#results)
- [Contributing](#conResults
Training and Validation Accuracy

Training and Validation Accuracy
Confusion Matrix

Confusion Matrix

The CNN model achieved a high accuracy on both training and validation sets, and the confusion matrix shows the performance on the test data.
tributing)
- [License](#license)

---

## Overview
This project focuses on detecting fire and smoke in images using a machine learning approach. A CNN model is trained on a dataset of fire and smoke images. The model is capable of classifying images into two categories: `fire` and `smoke`. It also includes handling of corrupted images, data cleaning, and visualization of results, such as the training history and confusion matrix.

---

## Features
- **Data Preprocessing**: Augmentation and normalization of images.
- **Custom Data Generator**: Skips corrupted or truncated images during training.
- **Model Training**: CNN architecture for binary image classification.
- **Model Evaluation**: Confusion matrix and accuracy/loss plots.
- **Single Image Classification**: Classify any user-provided image.
- **Visualization**: View sample predictions and model performance.

---

## Dataset
The dataset used consists of two main classes:
1. **Fire**: Images containing fire.
2. **Smoke**: Images containing visible smoke.

You can download a similar dataset from Kaggle: [Fire and Smoke Detection Dataset](https://www.kaggle.com/datasets).

Ensure that your dataset is structured as follows:


---

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

To set up the project and install all required dependencies, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/fire-smoke-detection.git
   cd fire-smoke-detection
Install the required Python packages:
```bash
pip install tensorflow keras numpy matplotlib seaborn pillow scikit-learn
```
##data-cleaning
Before training the model, you should clean the dataset to remove any corrupted or truncated images:
```bash
python clean_data.py
```
##model-training
To train the CNN model on your dataset, run the following command:

```bash
python train.py
```
##evaluation

After training, evaluate the model on the test data and visualize the results:
```bash
python evaluate.py
```
##classify-a-single-image
You can classify any user-provided image to predict whether it contains fire or smoke:

```bash
python classify.py --image path_to_image.jpg
```
##visualization
To view the training history (accuracy and loss) and confusion matrix, run:
```bash
python visualize.py
```
##results
Training and Validation Accuracy

Training and Validation Accuracy
Confusion Matrix

Confusion Matrix

The CNN model achieved a high accuracy on both training and validation sets, and the confusion matrix shows the performance on the test data.
##contributing
Contributions are welcome! If you have any suggestions or improvements, please feel free to submit a pull request.

    Fork the project.
    Create a feature branch (git checkout -b feature-name).
    Commit your changes (git commit -m 'Add some feature').
    Push to the branch (git push origin feature-name).
    Open a pull request.

License

This project is licensed under the MIT License - see the LICENSE file for details.
```markdown

Instructions for Use:
1. Replace Placeholder URLs: Ensure to replace the placeholder URLs (like `https://github.com/your-username/fire-smoke-detection.git`) with your actual repository URL.
2. Images: Place output images (like accuracy plots and confusion matrices) in the `/images/` folder and reference them correctly in the README.
3. Python Scripts: Make sure to create the corresponding Python scripts (`clean_data.py`, `train.py`, `evaluate.py`, `classify.py`, `visualize.py`) as indicated.

This README should provide a comprehensive overview of your project, making it easier for others to understand and contribute. Let me know if you need further changes!
```
