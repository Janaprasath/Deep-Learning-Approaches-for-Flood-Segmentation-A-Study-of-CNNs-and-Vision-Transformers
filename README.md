# Deep Learning Approaches for Flood Segmentation A Study of CNNs and Vision Transformers

**Project Overview**:
- This project investigates and compares the performance of Convolutional Neural Networks (CNNs) and Vision Transformer (ViT)-based architectures for semantic segmentation, with a specific focus on flood detection using satellite or aerial imagery.
- The study evaluates multiple state-of-the-art models-including U-Net, ResUNet, DeepLabV3+, UNetR, SwinUNet, and ViT-across a range of metrics such as accuracy, F1-score, precision, recall, Intersection over Union (IoU)and Dice Coefficient.
- The goal is to identify the strengths and limitations of each architecture and provide recommendations for their deployment in real-world, resource-constrained scenarios.

**Key Features**:
1. Implementation of both CNN-based and Vision Transformer-based segmentation models.
2. Standardized dataset preprocessing (resizing, normalization, mask alignment).
3. Rigorous training and evaluation using consistent data splits and metrics.
4. Detailed performance analysis, including computational efficiency and model convergence.
5. Application focus: Flood-affected area segmentation in remote sensing imagery.

**Models Implemented**:
1. CNN-based: U-Net, ResUNet, DeepLabV3+
2. Vision Transformer-based: Vision Transformer (ViT), UNetR, SwinUNet

**Performance Metrics**:
1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. Intersection over Union (IoU)
6. Dice Coefficient

**Project Structure & Model Pipeline**:
1. Import required libraries
2. [Data Loader](https://github.com/Janaprasath/Deep-Learning-Approaches-for-Flood-Segmentation-A-Study-of-CNNs-and-Vision-Transformers/blob/main/src/Utils/Data_Loader.py)
3. [Data Generator](https://github.com/Janaprasath/Deep-Learning-Approaches-for-Flood-Segmentation-A-Study-of-CNNs-and-Vision-Transformers/blob/main/src/Utils/Data_generator.py)
4. [Data Splits](https://github.com/Janaprasath/Deep-Learning-Approaches-for-Flood-Segmentation-A-Study-of-CNNs-and-Vision-Transformers/blob/main/src/Utils/Data_Split.py)
5. [Sample Visualization](https://github.com/Janaprasath/Deep-Learning-Approaches-for-Flood-Segmentation-A-Study-of-CNNs-and-Vision-Transformers/blob/main/src/Utils/Sample_Visualization.py)
6. [Model Architecture](https://github.com/Janaprasath/Deep-Learning-Approaches-for-Flood-Segmentation-A-Study-of-CNNs-and-Vision-Transformers/tree/main/src/Models)(choose the architecture from Models folder)
7. [Performance Metrics](https://github.com/Janaprasath/Deep-Learning-Approaches-for-Flood-Segmentation-A-Study-of-CNNs-and-Vision-Transformers/blob/main/src/Utils/Performance_Metrics.py
8. [Performance Metrics Visualization](src/Utils/Performance_Metrics_Visualization.py)
9. [Test Dataset Prediction and Performance Metrics](src/Utils/Test_Dataset_Prediction_and_Performance_Metrics.py)
10. [Test Dataset Prediction Visualization](src/Utils/Test_Dataset_Prediction_Visualization.py)

