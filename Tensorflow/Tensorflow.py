import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

import sys
import cv2 as cv 
import numpy as np


# dimensions of our images.
img_width, img_height = 120, 120

datadir = 'data'
#trainDir = 'data'
#validationDir = 'data'
nTrainSamples = 18474
nValidationSamples = 4618
epochs = 50
batchSize = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(6, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

Datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    validation_split= 0.2)

trainGenerator = Datagen.flow_from_directory(
    #trainDir,
    datadir,
    subset="training",
    target_size=(img_width, img_height),
    batch_size=batchSize,
    class_mode='categorical')

validationGenerator = Datagen.flow_from_directory(
    #validationDir,
    datadir,
    subset="validation",
    target_size=(img_width, img_height),
    batch_size=batchSize,
    class_mode='categorical')

model.fit_generator(
    trainGenerator,
    steps_per_epoch=nTrainSamples // batchSize,
    epochs=epochs,
    validation_data=validationGenerator,
    validation_steps=nValidationSamples // batchSize)

model.save_weights('model.h5')