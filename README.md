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
	```shell
	https://github.com/Khorrlybobo/Prediction-of-Oral-Cancer-Using-Different-Convolutional-Neural-Networks-and-Comparing-Their-Metrics.git
 ```

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

### VGG16
The VGG16 model achieved the following performance metrics:

- Loss: 0.5319547653198242
- Accuracy: 0.8583333492279053
- Precision: 0.8865979313850403
- Recall: 0.9347826242446899
- AUC: 0.8544255495071411

### ResNet
The ResNet model achieved the following performance metrics:

- Loss: 0.6444424390792847
- Accuracy: 0.8333333134651184
- Precision: 0.8913043737411499
- Recall: 0.8913043737411499
- AUC: 0.8761645555496216

### InceptionV3
The InceptionV3 model achieved the following performance metrics:

- Loss: 0.7425212264060974
- Accuracy: 0.6499999761581421
- Precision: 0.837837815284729
- Recall: 0.6739130616188049
- AUC: 0.6308229565620422

### MobileNetV2
The MobileNetV2 model achieved the following performance metrics:

- Loss: 0.7906744480133057
- Accuracy: 0.7749999761581421
- Precision: 0.782608687877655
- Recall: 0.97826087474823
- AUC: 0.69972825050354

These results highlight the performance of each CNN architecture in predicting oral cancer. Based on the metrics, we observe that the VGG16 and ResNet models achieved higher accuracy, precision, recall, and AUC scores compared to InceptionV3 and MobileNetV2. These findings suggest that VGG16 and ResNet architectures may be more suitable for oral cancer detection tasks. However, further analysis and experiments are necessary to determine the most optimal architecture for this specific application.

Please note that these results are based on the specific dataset and experimental setup used in this project. Further research and evaluation may be required to generalize the findings to different datasets and scenarios.

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

