import os, sys, time, cv2, matplotlib
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.metrics import confusion_matrix

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

model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=['accuracy'])

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

history = model.fit_generator(
    trainGen,
    steps_per_epoch=71,
    epochs=epochs,
    validation_data=validationGen,
    validation_steps=28,
    callbacks=[es, mc],
    verbose=1)

endTime = time.time();

#plot stuff

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
nepochs=len(history.history['loss'])
plt.plot(range(nepochs), history.history['loss'],     'r-', label='train')
plt.plot(range(nepochs), history.history['val_loss'], 'b-', label='test')
plt.legend(prop={'size': 20})
plt.ylabel('loss')
plt.xlabel('# of epochs')
plt.subplot(1,2,2)
plt.plot(range(nepochs), history.history['acc'],     'r-', label='train')
plt.plot(range(nepochs), history.history['val_acc'], 'b-', label='test')
plt.legend(prop={'size': 20})
plt.ylabel('accuracy')
plt.xlabel('# of epochs')


print("Time spent training: " + str(endTime-startTime) + "seconds") 



# test stuff
xTest, yTest = [], []
for ibatch, (X, y) in enumerate(validationGen):
    xTest.append(X)
    yTest.append(y)
    ibatch += 1
    if (ibatch == 5*28): break

# Concatenate everything together
xTest = np.concatenate(xTest)
yTest = np.concatenate(yTest)
yTest = np.int32([np.argmax(r) for r in yTest])

# Get the predictions from the model and calculate the accuracy
yPred = np.int32([np.argmax(r) for r in model.predict(xTest)])
match = (yTest == yPred)
print('Testing Accuracy = %.2f%%' % (np.sum(match)*100/match.shape[0]))

nomatch = (yTest != yPred)
bad_pred = yPred[nomatch]
bad_true = yTest[nomatch]
bad_img = xTest[nomatch]
print('%d examples of bad predictions' % bad_pred.size)

plt.figure(figsize=(9,8))
cm = confusion_matrix(y_test, y_pred)
cm = cm / cm.sum(axis=1)
sn.heatmap(cm, annot=True);

