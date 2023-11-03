# Handwritten Alphabet Recognizer Model

This repository contains a handwritten alphabet recognizer model implemented using Keras and TensorFlow. The model is designed to recognize handwritten alphabet characters. It's a Convolutional Neural Network (CNN) that has been trained on a dataset of handwritten alphabet images.

## Dataset
The dataset used for training and validation is stored in the `Dataset/` directory. The dataset is organized into subdirectories, where each subdirectory corresponds to a different alphabet character. The dataset contains a variety of handwritten characters from A to Z.
```
Dataset/
|-- A/
|   |-- 1.png
|   |-- 2.png
|   |-- ...
|-- B/
|   |-- 1.png
|   |-- 2.png
|   |-- ...
|-- ...
|-- Z/
|   |-- 1.png
|   |-- 2.png
|   |-- ...
```

### Dataset Details
- Number of classes: 26 classes
- Number of images: 130 images
- Number of images/class: 130/26 = 5
- smol dataset :)
  
## Getting Started

### Prerequisites
To run this code, you need to have the following libraries installed:
- `numpy`
- `matplotlib`
- `keras`
- `scikit-learn`

You can install these libraries using `pip`:
```bash
pip install numpy matplotlib keras scikit-learn
```

### Code Overview
The code is structured as follows:

- Import necessary libraries and modules.
- Load and preprocess the dataset.
- Split the data into training and validation sets.
- Define a data generator for preprocessing and data augmentation.
- Build the CNN model for alphabet recognition.
- Compile the model with the appropriate loss and optimizer.
- Convert string labels to numerical labels and one-hot encode them.
- Train the model on the training data.
- Evaluate the model on the validation data.
- Visualize feature maps for a sample image.

## Model Architecture
The CNN model consists of the following layers:
1. Convolutional layer with 32 filters (5x5) and ReLU activation.
2. Max-pooling layer (3x3).
3. Convolutional layer with 32 filters (3x3) and ReLU activation.
4. Max-pooling layer (2x2).
5. Convolutional layer with 64 filters (3x3) and ReLU activation.
6. Max-pooling layer (2x2).
7. Flatten layer.
8. Fully connected layer with 512 units and ReLU activation.
9. Dropout layer with a dropout rate of 0.25.
10. Fully connected layer with 128 units and ReLU activation.
11. Output layer with softmax activation (for {num_classes} classes).

## Training
The model is trained for 10 epochs using the training data.

## Evaluation
The model is evaluated on the validation data, and the test accuracy is printed.

## Layers Visualization
The code also includes a section for visualizing feature maps for a sample image. It demonstrates how to extract and visualize the features from different layers of the CNN.

## How to Use
You can use this code as a starting point for your own handwritten alphabet recognition tasks. You can modify the model architecture, data preprocessing, and data augmentation techniques to suit your specific use case.

To recognize alphabet characters, you can pass an image of a handwritten character through the trained model and obtain predictions.

Feel free to explore and adapt this code for your own handwriting recognition projects. Happy coding!
