import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

IMAGE_SIZE = [100, 100]
epochs = 200
batch_size = 5
PATH = './data/'

# useful for getting number of files
image_files = glob(PATH + '/*/*.jp*g')
# useful for getting number of classes
folders = glob(PATH + '/*')
print("Number of images:", len(image_files))
print("Number of classes: ", len(folders))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001), input_shape=IMAGE_SIZE+[1]))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(7, kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Activation("softmax"))
model.add(Dense(526, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(folders), activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

gen = ImageDataGenerator(
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
)

train_generator = gen.flow_from_directory(
  PATH,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
  color_mode="grayscale"
)

r = model.fit(
  train_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
)

model.save("models/saved_models_custom/model100x100.hdf5")
print("Saved model")