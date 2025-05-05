# Facial Recognition with MLP - Olivetti Faces

This project implements a facial recognition system using a Multi-Layer Perceptron (MLP) neural network on the Olivetti faces dataset, with an interactive Streamlit interface for exploration and visualization.

## Overview

The Olivetti faces dataset consists of 400 grayscale images of 40 different individuals (10 images per person). Despite the limited examples per class, we apply MLP-based classification to identify subjects from their facial images.

## Features

- **Dataset Exploration**: View sample images from the dataset and explore individual subjects
- **Interactive Model Training**: Configure and train an MLP with customizable parameters
- **Model Evaluation**: Visualize model performance with accuracy metrics, confusion matrices, and more
- **Prediction Visualization**: See model predictions with correct and misclassified samples highlighted

## Getting Started

### Prerequisites

The application requires the following Python packages:
```
streamlit
scikit-learn
numpy
matplotlib
pandas
seaborn
```

### Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```
   pip install streamlit scikit-learn numpy matplotlib pandas seaborn
   ```

### Running the Application

Run the Streamlit app with:
```
streamlit run olivetti_mlp_streamlit.py
```

## Usage Guide

1. **Explore the Dataset**: 
   - Scroll through sample faces
   - Select specific subjects to view all their images

2. **Configure the Model**:
   - Adjust test/train split ratio
   - Enable/disable PCA dimensionality reduction
   - Configure MLP architecture (hidden layers, neurons)
   - Set regularization strength and learning parameters

3. **Train and Evaluate**:
   - Click "Train MLP Classifier" to begin training
   - View performance metrics, learning curves, and confusion matrix
   - Examine model predictions on test images

4. **Analyze Results**:
   - Toggle between correct and misclassified predictions
   - Identify patterns in model errors

## Machine Learning Concepts

This application demonstrates several important machine learning concepts:

1. **Facial Recognition Fundamentals**:
   - Pixel-based feature representation
   - Subject identification through supervised learning

2. **MLP Architecture**:
   - Multi-layer neural networks for image classification
   - Hidden layer configurations and their impact on performance

3. **Dimensionality Reduction**:
   - PCA for efficient feature extraction
   - Variance retention in facial recognition context

4. **Model Evaluation**:
   - Classification metrics interpretation
   - Visual analysis of model predictions

5. **Challenges of Limited Data**:
   - Working with small datasets (10 samples per class)
   - Techniques to mitigate overfitting

## Notes on Limited Data

As noted in the scikit-learn documentation, with only 10 examples per class, this dataset presents challenges for supervised learning approaches. The application includes several techniques to address these limitations:

- **PCA for Dimensionality Reduction**: Reduces feature space to essential components
- **Regularization**: Uses L2 regularization (alpha parameter) to prevent overfitting
- **Early Stopping**: Prevents overtraining on limited samples

## Extensions

Potential enhancements for this project:

1. **Data Augmentation**: Implement image transformations to expand the training set
2. **Transfer Learning**: Compare with pretrained face recognition models
3. **Cross-Validation**: Add k-fold cross-validation for robust performance estimation
4. **Alternative Models**: Compare MLP with SVMs, decision trees, and other classifiers
5. **Feature Engineering**: Explore manual feature extraction techniques for faces
