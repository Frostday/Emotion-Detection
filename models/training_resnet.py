import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

IMAGE_SIZE = [100, 100]
epochs = 100
batch_size = 5
PATH = './data/'

# useful for getting number of files
image_files = glob(PATH + '/*/*.jp*g')
# useful for getting number of classes
folders = glob(PATH + '/*')
print("Number of images:", len(image_files))
print("Number of classes: ", len(folders))

resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in resnet.layers:
  layer.trainable = False
x = Flatten()(resnet.output)
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=prediction)
# model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

gen = ImageDataGenerator(
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)

train_generator = gen.flow_from_directory(
  PATH,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)

r = model.fit(
  train_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
)

model.save("models/saved_models_resnet/model100x100.hdf5")
print("Saved Resnet model")