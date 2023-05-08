# GAYOGA

# Yoga Pose Image Classification using DenseNet

This repository contains the code for a deep learning model that classifies 107 different yoga poses based on input images. The model is based on the DenseNet architecture, with a few additional layers added on top for fine-tuning.

# Dataset

The dataset used to train and test this model consists of a collection of images of individuals performing 107 different yoga poses. The images were carefully curated and preprocessed to ensure accuracy and consistency, and were split into training and validation sets for model training.


![download](https://user-images.githubusercontent.com/102585626/236802689-6aa1ffe4-97c0-4e63-b4f5-378baa8f69a9.png)
(THE IMAGES IN THE DATASET)


# Model Architecture

The model is based on the DenseNet architecture, which has shown strong performance in a variety of computer vision tasks. The model uses a pre-trained DenseNet model as a base, and adds several additional layers on top for fine-tuning. The model includes dropout layers for regularization and to prevent overfitting, and uses a softmax activation function in the final layer for multiclass classification.

# Training

The model was trained using the Adam optimizer and a batch size of 32. The training process involved multiple epochs of training on the training set, with validation performed after each epoch to monitor performance and prevent overfitting. The model was fine-tuned using transfer learning techniques, with the pre-trained DenseNet model serving as a starting point for training.

# Evaluation

The model was evaluated on a separate test set, which was not used during training or validation. The model's performance was compared to existing models for yoga pose classification, and demonstrated strong performance across a range of metrics.

# Usage

To use this model for yoga pose classification, simply clone this repository and run the yoga_pose_classifier.py file. The file takes an input image and returns a predicted yoga pose label.

# Some potential areas for future work on this project include:

Adding additional layers to the model for further fine-tuning and optimization
Exploring alternative model architectures and training techniques to improve performance
Developing an application or platform for real-time yoga pose classification using this model.
