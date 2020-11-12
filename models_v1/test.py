import numpy as np
import cv2
import os

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = [100, 100]

gen = ImageDataGenerator(
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
)

train_generator = gen.flow_from_directory(
  "./data",
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=1,
  color_mode="grayscale"
)

class_labels = train_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print(class_labels)