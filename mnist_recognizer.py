import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

#upload dataset
x = np.load('x.npy')
y = np.load('y.npy')
y = keras.utils.to_categorical(y)
dimensions = x.shape[1:]

#print a random picture 


#augument some data 
datagen = ImageDataGenerator(width_shift_range=0.2,
                             height_shift_range=0.2,
                             rotation_range=40,
                             brightness_range=[0.3,1.45],
                             zoom_range=[0.6,1.3],
                             shear_range=25,
                             validation_split=0.2)

#build a model
model = models.Sequential()

model.add(layers.Conv2D(10, (7, 7), activation='relu', input_shape=(28,28,1)))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(10, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(20, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(20, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())
# Fully connected layer

model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dropout(0.4))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation = 'softmax'))

#print to analyze the number of trainable parameters
model.summary()

#apply optimization
opt = Adam(lr=0.05,clipnorm=1)


model.compile(loss='categorical_crossentropy',
      optimizer=opt,
      metrics=['accuracy'])

#initialize some training parameters
batch_size = 64
epochs = 70
early_stopping = EarlyStopping(monitor='loss',patience=15,restore_best_weights=True)
LR_schedule = ReduceLROnPlateau(monitor='accuracy',factor=0.6, patience=5)


# fit the model on batches with real-time data augmentation:
model.fit(datagen.flow(x, y, batch_size=64),
          steps_per_epoch=len(x) / batch_size, 
          epochs=epochs,verbose=1, 
          batch_size=batch_size,
          callbacks = [early_stopping,LR_schedule],
          shuffle=1)


model.save("model.h5")
