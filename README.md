## Prediction-of-Oral-Cancer-Using-Different-Convolutional-Neural-Networks-and-comparing-their-metrics

This project aims to predict oral cancer using various Convolutional Neural Network (CNN) architectures and compare their performance metrics. By leveraging deep learning techniques and image analysis, we aim to develop accurate and efficient tools for the early detection of oral cancer.

# Table of Contents
1. Introduction
2. Installation
3. Dataset
4. Methodology
5. Results and Analysis
6. Usage
7. Contributing


# [A] Introduction

Oral cancer is a significant global health concern, and its early detection plays a crucial role in improving patient outcomes. This project explores the prediction of oral cancer using different CNN architectures, namely VGG16, ResNet, InceptionV3, and MobileNetV2. By comparing their performance metrics, we aim to identify the most effective architecture for oral cancer detection.

# [B] Installation

To run the code and reproduce the results, follow these steps:

1. Clone the repository:

	https://github.com/Khorrlybobo/Prediction-of-Oral-Cancer-Using-Different-Convolutional-Neural-Networks-and-Comparing-Their-Metrics.git

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

# [C] Dataset

The dataset used for training and evaluation consists of oral cancer images. The dataset is divided into two classes: "cancer" and "non-cancer". These images are expertly curated and labeled by medical professionals, ensuring the accuracy and reliability of the dataset.

# [D] Methodology

1. Data Preprocessing:
   - Data augmentation techniques, such as rotation, shear, zoom, horizontal flip, and brightness adjustments, are applied to enhance the model's performance and prevent overfitting.
   - The dataset is split into training, validation, and testing sets to train and evaluate the CNN models.

2. Model Architecture:
   - Four different CNN architectures, namely VGG16, ResNet, InceptionV3, and MobileNetV2, are utilized for oral cancer prediction.
   - Each architecture is fine-tuned and customized specifically for oral cancer detection by adding additional layers.

3. Model Training and Evaluation:
   - The models are trained using the training set and evaluated on the validation set.
   - Evaluation metrics such as accuracy, precision, recall, and area under the curve (AUC) are monitored to assess the models' performance.
   - Comparative analysis of the metrics is performed to identify the most effective architecture for oral cancer prediction.

# [E] Results and Analysis

Based on the evaluation metrics, a comprehensive analysis of the performance of each CNN architecture is conducted. The results include accuracy, precision, recall, and AUC scores for each architecture. This analysis provides insights into the strengths and weaknesses of each model, facilitating the selection of the most appropriate architecture for oral cancer prediction.

# [F] Usage

To use the trained models for oral cancer prediction, follow these steps:

1. Load the desired model:

   ```python
   from tensorflow.keras.models import load_model

   model = load_model('model.h5')
   ```

2. Preprocess the input image.

3. Perform prediction:

   ```python
   prediction = model.predict(input_image)
   ```

4. Analyze the prediction results based on the model's performance metrics.

# [G] Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

