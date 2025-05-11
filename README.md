# Deep Learning Approaches for Flood Segmentation A Study of CNNs and Vision Transformers

Project Overview:
- This project investigates and compares the performance of Convolutional Neural Networks (CNNs) and Vision Transformer (ViT)-based architectures for semantic segmentation, with a specific focus on flood detection using satellite or aerial imagery. - - The study evaluates multiple state-of-the-art models-including U-Net, ResUNet, DeepLabV3+, UNetR, SwinUNet, and ViT-across a range of metrics such as accuracy, F1-score, precision, recall, and Intersection over Union (IoU).
- The goal is to identify the strengths and limitations of each architecture and provide recommendations for their deployment in real-world, resource-constrained scenarios.

Key Features:
1. Implementation of both CNN-based and Vision Transformer-based segmentation models.
2. Standardized dataset preprocessing (resizing, normalization, mask alignment).
3. Rigorous training and evaluation using consistent data splits and metrics.
4. Detailed performance analysis, including computational efficiency and model convergence.
5. Application focus: Flood-affected area segmentation in remote sensing imagery.

Models Implemented:
1. CNN-based: U-Net, ResUNet, DeepLabV3+
2. Vision Transformer-based: Vision Transformer (ViT), UNetR, SwinUNet

Performance Metrics:
1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. Intersection over Union (IoU)

Summary of Results:
1. DeepLabV3+ and ViT achieved the highest overall accuracy and F1-score.
2. ResUNet, UNetR, and SwinUNet demonstrated superior IoU, making them suitable for tasks requiring precise boundary delineation.
3. The choice of model depends on whether the application prioritizes overall segmentation accuracy or precise spatial localization

How to Run:
1. Prepare the dataset of satellite/aerial images and corresponding segmentation masks.
2. Preprocess images (resize, normalize, mask alignment).
3. Select the desired model architecture and configure training parameters.
4. Train the model using the provided scripts (TensorFlow/Keras).
5. Evaluate performance on validation and test sets using the specified metrics.
6. Visualize segmentation results and analyze metric trends to inform model selection.

Technical Requirements:
1. Python 3.x
2. TensorFlow and Keras
3. OpenCV, NumPy, scikit-learn
4. GPU-enabled environment recommended for training transformer-based models
