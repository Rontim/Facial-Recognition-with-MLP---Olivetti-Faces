import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import pandas as pd
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Facial Recognition with MLP",
    page_icon="👤",
    layout="wide"
)


# Helper functions
def load_data():
    olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = olivetti.data
    y = olivetti.target
    return X, y, olivetti.images


def show_faces(X, y, images, n_row=2, n_col=5, titles=None):
    fig, axes = plt.subplots(n_row, n_col, figsize=(2 * n_col, 2 * n_row))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        if titles is not None:
            ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    return fig


def train_model(X_train, y_train, X_test, y_test, hidden_layers, alpha, learning_rate_init, max_iter, use_pca,
                pca_components):
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA if selected
    if use_pca:
        pca = PCA(n_components=pca_components)
        X_train_processed = pca.fit_transform(X_train_scaled)
        X_test_processed = pca.transform(X_test_scaled)
        explained_variance = sum(pca.explained_variance_ratio_)
        st.write(f"PCA: Reduced features from {X_train_scaled.shape[1]} to {X_train_processed.shape[1]} dimensions")
        st.write(f"Explained variance ratio: {explained_variance:.2f}")
    else:
        X_train_processed = X_train_scaled
        X_test_processed = X_test_scaled

    # Train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=alpha,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
        random_state=42
    )

    # Track training time
    start_time = time.time()
    mlp.fit(X_train_processed, y_train)
    training_time = time.time() - start_time

    # Evaluate
    y_pred = mlp.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)

    return mlp, y_pred, accuracy, training_time, X_test_processed


def visualize_predictions(X_test, y_test, y_pred, test_images, indices=None, n_samples=10):
    if indices is None:
        # Get a mix of correct and incorrect predictions
        correct = np.where(y_test == y_pred)[0]
        incorrect = np.where(y_test != y_pred)[0]

        # Prioritize showing incorrect samples if they exist
        n_incorrect = min(len(incorrect), n_samples // 2)
        n_correct = min(len(correct), n_samples - n_incorrect)

        incorrect_indices = np.random.choice(incorrect, size=n_incorrect, replace=False) if n_incorrect > 0 else []
        correct_indices = np.random.choice(correct, size=n_correct, replace=False) if n_correct > 0 else []

        indices = np.concatenate([incorrect_indices, correct_indices])

    n_samples = min(len(indices), n_samples)
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(indices[:n_samples]):
        axes[i].imshow(test_images[idx], cmap='gray')

        true_label = y_test[idx]
        pred_label = y_pred[idx]

        title = f"True: {true_label}\nPred: {pred_label}"
        color = "green" if true_label == pred_label else "red"
        axes[i].set_title(title, color=color)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return fig


def plot_learning_curve(mlp):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mlp.loss_curve_)
    ax.set_title('MLP Learning Curve')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.grid(True)
    return fig


# App title and introduction
st.title("👤 Facial Recognition using MLP with Olivetti Faces Dataset")

st.markdown("""
### About the Dataset
The Olivetti faces dataset consists of:
- 400 grayscale images of 40 different people (10 images per person)
- Each image is 64×64 pixels
- Variations include lighting, facial expressions, and details like glasses
- All images are against a dark background with subjects in frontal position
""")

# Initialize or load the data
if 'data_loaded' not in st.session_state:
    with st.spinner('Loading Olivetti faces dataset...'):
        X, y, images = load_data()
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['images'] = images
        st.session_state['data_loaded'] = True
else:
    X = st.session_state['X']
    y = st.session_state['y']
    images = st.session_state['images']

# Display dataset information
st.header("Dataset Overview")
n_samples, n_features = X.shape
n_classes = len(np.unique(y))

col1, col2, col3 = st.columns(3)
col1.metric("Total Images", n_samples)
col2.metric("Unique Subjects", n_classes)
col3.metric("Images per Subject", n_samples // n_classes)

# Sidebar for configuration
st.sidebar.header("Model Configuration")

# Test-train split
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 30, 5) / 100

# PCA configuration
use_pca = st.sidebar.checkbox("Use PCA for Dimensionality Reduction", True)
pca_components = st.sidebar.slider("PCA Components (% variance to retain)", 70, 99, 95, 1) / 100

# MLP configuration
st.sidebar.subheader("MLP Parameters")
n_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 3, 2)
hidden_layer_sizes = []
for i in range(n_hidden_layers):
    layer_size = st.sidebar.slider(f"Neurons in Layer {i + 1}", 10, 200, 100, 10)
    hidden_layer_sizes.append(layer_size)

alpha = st.sidebar.select_slider(
    "Regularization Strength (alpha)",
    options=[0.0001, 0.001, 0.01, 0.1, 1.0],
    value=0.0001
)

learning_rate = st.sidebar.select_slider(
    "Initial Learning Rate",
    options=[0.0001, 0.001, 0.01, 0.1],
    value=0.001
)

max_iterations = st.sidebar.slider("Maximum Iterations", 100, 2000, 1000, 100)

# Display sample faces
st.header("Sample Faces")
st.write("Here are some random samples from the dataset:")

# Group samples by subject
st.subheader("Samples by Subject")
subject_to_show = st.slider("Select Subject ID", 0, n_classes - 1, 0)
subject_indices = np.where(y == subject_to_show)[0]
subject_samples = [images[i] for i in subject_indices]
subject_titles = [f"Sample {i + 1}" for i in range(len(subject_samples))]

col1, col2 = st.columns([3, 1])
with col1:
    fig = show_faces(X, y, subject_samples, n_row=2, n_col=5, titles=subject_titles)
    st.pyplot(fig)
with col2:
    st.write(f"**Subject #{subject_to_show}**")
    st.write(f"10 different images showing variations in lighting, expressions, and details")

# Train model button
st.header("Train and Evaluate Model")
if st.button("Train MLP Classifier"):

    # Split data
    X_train, X_test, y_train, y_test, train_images, test_images = train_test_split(
        X, y, images, test_size=test_size, stratify=y, random_state=42
    )

    st.write(f"Training set: {X_train.shape[0]} images")
    st.write(f"Testing set: {X_test.shape[0]} images")

    # Show progress
    with st.spinner("Training MLP classifier..."):
        mlp, y_pred, accuracy, training_time, X_test_processed = train_model(
            X_train, y_train, X_test, y_test,
            tuple(hidden_layer_sizes), alpha, learning_rate, max_iterations,
            use_pca, pca_components
        )

    # Display results
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Training Time", f"{training_time:.2f} seconds")

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df.style.highlight_max(axis=0))

    # Learning curve
    st.subheader("Learning Curve")
    fig = plot_learning_curve(mlp)
    st.pyplot(fig)

    # Confusion matrix (with option to toggle)
    if st.checkbox("Show Confusion Matrix", False):
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(y_test, y_pred)
        st.pyplot(fig)

    # Prediction visualization
    st.subheader("Model Predictions")
    st.write("Green title = correct prediction, Red title = incorrect prediction")

    # Option to view different types of predictions
    view_option = st.radio(
        "Choose predictions to view:",
        ["Mixed Predictions", "Misclassified Only", "Correctly Classified Only"]
    )

    if view_option == "Misclassified Only":
        misclassified = np.where(y_test != y_pred)[0]
        if len(misclassified) > 0:
            fig = visualize_predictions(X_test_processed, y_test, y_pred, test_images,
                                        indices=misclassified[:10])
            st.pyplot(fig)
        else:
            st.write("No misclassified samples found!")

    elif view_option == "Correctly Classified Only":
        correct = np.where(y_test == y_pred)[0]
        if len(correct) > 0:
            fig = visualize_predictions(X_test_processed, y_test, y_pred, test_images,
                                        indices=correct[:10])
            st.pyplot(fig)
    else:
        fig = visualize_predictions(X_test_processed, y_test, y_pred, test_images)
        st.pyplot(fig)

# Additional information
st.header("About MLP for Facial Recognition")
st.markdown("""
### Challenges with Limited Data
As noted in the dataset description, with only 10 examples per class, this dataset is challenging for supervised learning approaches like MLPs. Some challenges include:

1. **Overfitting**: With limited training data, the model may memorize the training set rather than learning general patterns
2. **Generalization**: The model may struggle to recognize new images of the same people under different conditions
3. **Feature Extraction**: Without enough examples, it's difficult to learn which facial features are most important

### Techniques Used to Address These Challenges
- **Dimensionality Reduction with PCA**: Reduces the number of features to focus on the most important ones
- **Regularization (alpha parameter)**: Prevents overfitting by penalizing large weights
- **Early Stopping**: Halts training when validation performance stops improving
- **Standardization**: Normalizes input features to improve convergence

### Alternative Approaches
For a dataset with limited samples like this, alternative approaches might include:
- Transfer learning with pre-trained face recognition models
- Data augmentation to artificially increase the training set size
- Semi-supervised learning methods that can leverage unlabeled data
- Contrastive learning or other approaches that learn embeddings 
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Created as a demonstration of facial recognition with MLPs using the Olivetti faces dataset")