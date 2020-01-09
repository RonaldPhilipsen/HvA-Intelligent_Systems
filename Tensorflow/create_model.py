import os, sys, time, cv2
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# dimensions of our images.
img_width, img_height = 300, 300

trainDir = 'data\\train'
validationDir = 'data\\test'
nTrainSamples = 9081
nValidationSamples = 3632
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
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
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
    horizontal_flip=True)

trainGenerator = Datagen.flow_from_directory(
    trainDir,
    target_size=(img_width, img_height),
    batch_size=batchSize,
    class_mode='categorical')

validationGenerator = Datagen.flow_from_directory(
    validationDir,
    target_size=(img_width, img_height),
    batch_size=batchSize,
    class_mode='categorical')

es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

startTime = time.time();

model.fit_generator(
    trainGenerator,
    steps_per_epoch=nTrainSamples // batchSize,
    epochs=epochs,
    validation_data=validationGenerator,
    validation_steps=nValidationSamples // batchSize,
    callbacks=[es, mc],
    verbose=1)

endTime = time.time();

print("Time spent training: " + str(endTime-startTime) + "seconds") 