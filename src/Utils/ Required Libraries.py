import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, UpSampling2D,
    AveragePooling2D, Conv2DTranspose, Concatenate, Lambda
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import MeanIoU, Precision, Recall

from sklearn.model_selection import train_test_split

