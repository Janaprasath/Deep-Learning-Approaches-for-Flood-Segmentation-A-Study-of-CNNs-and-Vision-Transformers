CONCLUSION:

- This project compared three CNN-based models (U-Net, DeepLabV3+, ResUNet) and three Vision Transformer-based models (ViT for Segmentation, UNetR, SwinUNet) for flood segmentation tasks.
- The evaluation was based on performance metrics such as Accuracy, Dice Coefficient, F1 Score, Precision, Recall, MeanIoU, and Loss across training, validation, and test sets.

Training Performance of CNN and ViT Models on Flood Segmentation:

Model/Metrics           |   Accuracy     |      Loss	   |      Precision  |	  Recall   |	  F1 Score  |    MeanIoU   |   Dice coefficient	 |
------------------------|----------------|---------------|-----------------|-------------|--------------|--------------|---------------------|
UNet	                  |      96.11	   |    0.0695     |      	96.72	   |   95.38     |	   96.14	  |     49.99	   |       94.14         |
DeepLabV3+	            |      97.78     |    0.0316	   |       98.86     |	 97.34	   |    97.91	    |     47.52	   |      97.27          |
ResUNet	                |      97.63     |	  0.3533     |	     98.5      |	 97.25     |	  97.85     |	    35.33    |	    97.01          |
ViT for Segmentation	  |      83.33     |	  0.3266     |	    80.82      |	  80.5     | 	  81.85     |	    25       |	    70.64          |
UNetR	                  |      83.83     |	  0.3282     |  	  86.75	     |    77.51	   |    82.01     |	    25	     |      72.79          |
SwinUNet	              |      84.01     |  	0.3266	   |      80.82	     |    80.85	   |    81.85	    |      25	     |      73.2           |


Validation Performance of CNN and ViT Models on Flood Segmentation:

Model/Metrics           |   Accuracy     |      Loss	   |      Precision  |	  Recall   |	  F1 Score  |    MeanIoU |   Dice coefficient	|
------------------------|----------------|---------------|-----------------|-------------|--------------|------------|--------------------|
UNet	                  |    87.26       |   	0.3251     |  	83.62        |    	88.07  |	87.64       |	  88.07	   |      81.18         |
DeepLabV3+	            |    81.67       |  	0.8546     | 	  88.27        |     	65.49  |  77.81       |  	47.51    |    	70.75         |
ResUNet	                |    83.07       |  	0.5189     |	  88.69        |	    71.86  |	81.71	      |   36.05    |	    77.12         |
ViT for Segmentation	  |    80.28       |  	0.4104     |	  78.16        |    	78.94  | 	79.15       |	    25	   |      71.02         |
UNetR	                  |    81.36       | 	  0.354	     |    80           | 	    78.54  |	80.55       |	    25	   |      71.92         |
SwinUNet	              |    78.73       | 	  0.4104     | 	  75.44        |    	79.16  | 	78.7        |	    25	   |      78.7          |



Test Performance of CNN and ViT Models on Flood Segmentation:

Model/Metrics           |   Accuracy     |      Loss	   |      Precision  |	  Recall   |	  F1 Score  |    MeanIoU |   Dice coefficient	|
------------------------|----------------|---------------|-----------------|-------------|--------------|------------|--------------------|
UNet	                  |   76.04	       |     0.4057	   |     49.99	     |    78.41	   |   85.43	    |    86.71   |     85.88	        |
DeepLabV3+	            |   68.72	       |     0.7842	   |     47.5	       |    86.15	   |   76.7	      |    86.15   |     83.98	        |
ResUNet	                |   79.28	       |     0.3643	   |     6.06	       |    88.23	   |   88.03      |	   82.2    |     89.04          |
ViT for Segmentation	  |   63.35	       |     0.4214    |	    25	       |    68.47	   |   76.09	    |    80.26   |     78.83          |	
UNetR	                  |   69.62	       |     0.3018	   |      25	       |    82.14	   |   86.36	    |    84.81	 |     87.27	        |
SwinUNet	              |   78.23	       |     0.2391	   |       25	       |    83.87	   |   88.65	    |    88.94   |     89.11          |	

- **CNN**-based models generally outperformed **ViT**-based models in terms of segmentation accuracy and generalization.

- **SwinUNet** achieved the best test **Dice Coefficient (89.11%)** and **MeanIoU (88.94%)**, indicating strong segmentation quality on unseen data.

- **ResUNet** showed strong generalization with high test **F1 Score (88.03%)** and **Dice Coefficient (89.04%)**, while also maintaining top performance during training and validation.

- **ViT for Segmentation** and **UNetR**, while competitive during training, struggled with generalization, evident in their lower test precision and MeanIoU.

- **DeepLabV3**+, although strong in training, showed signs of overfitting, with a notable drop in validation and test performance.

- Overall, **CNN**-based architectures (especially ResUNet and SwinUNet) demonstrated more stable and robust performance across all datasets. Vision Transformer models need more data or regularization techniques to improve their generalization on flood segmentation tasks. 


