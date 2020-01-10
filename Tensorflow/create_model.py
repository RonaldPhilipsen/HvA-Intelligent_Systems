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
imgWidth, imgHeight = 300, 300

trainDir = 'data\\train'
validationDir = 'data\\test'
nTrainSamples = 9081
nValidationSamples = 3632
epochs = 60
nbatch = 128

input_shape = (imgWidth, imgHeight, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(6, activation="softmax"))

model.compile(loss="categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


trainDatagen = ImageDataGenerator( rescale=1./255,
                                    rotation_range=10.,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True
                                  )

testDatagen  = ImageDataGenerator( rescale=1./255 )

trainGen = trainDatagen.flow_from_directory(
        trainDir,
        target_size=(imgWidth, imgHeight),
        color_mode='grayscale',
        batch_size=nbatch,
        classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
        class_mode='categorical'
    )

validationGen = testDatagen.flow_from_directory(
        validationDir,
        target_size=(imgWidth, imgHeight),
        color_mode='grayscale',
        batch_size=nbatch,
        classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
        class_mode='categorical'
    )


es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

startTime = time.time();

model.fit_generator(
    trainGen,
    steps_per_epoch=71,
    epochs=epochs,
    validation_data=validationGen,
    validation_steps=28,
    callbacks=[es, mc],
    verbose=1)

endTime = time.time();

print("Time spent training: " + str(endTime-startTime) + "seconds") 