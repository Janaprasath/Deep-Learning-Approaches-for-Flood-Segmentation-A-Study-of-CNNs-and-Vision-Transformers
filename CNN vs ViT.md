**Why CNNs Outperformed Vision Transformers?**
1. **Data Efficiency**
- CNNs like U-Net and ResUNet are highly effective on small datasets due to their strong inductive biases — such as local connectivity and translation invariance.
- These properties allow them to learn meaningful spatial patterns even when data is limited.
- On the other hand, Vision Transformers (ViTs) generally require large datasets to perform well, as they rely more on learning relationships from data rather than hardcoded spatial priors.

2. **Training Stability on Small Data**
- ViTs tend to overfit on small datasets or fail to converge well without large-scale pretraining, strong regularization, or data augmentation.
- In your case, despite using combined loss functions, models like ViT for Segmentation and UNetR showed lower test accuracy and generalization, likely due to insufficient training signals from the limited dataset.

3. **Model Complexity vs. Dataset Size**
- Transformer-based models often have more parameters and higher complexity.
-  When the dataset is small (300 images), this complexity becomes a liability, making them prone to memorizing rather than generalizing. CNNs, being more compact and structurally constrained, are less prone to this issue.

4. **Combined Loss Can Favor CNNs**
- Combined loss functions like Dice + BCE are more effective with models that already handle class imbalance and spatial structure well — which CNNs do.
- ViTs might require further tuning or architecture adjustments to benefit equally from such losses.
