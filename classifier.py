# -*- coding: utf-8 -*-
#!/user/bin/env python3
import os
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam, SGD
from keras import initializers
#%% Parameters
train_data_dir = '/home/winter/Keras/SceneClassify/data/train/train_data'
validation_data_dir = '/home/winter/Keras/SceneClassify/data/validation/validation_data'
path_weights = 'params/first_try.h5'

img_width, img_height = 500, 500
batch_size = 32
nb_train_samples = 53879
nb_validation_samples = 7210
epochs = 50

#%% bulid model
#model = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=None,
#              pooling=None, classes = 80)

################################sequential to build model#################################
model = Sequential()
model.add(Conv2D(32, (3, 3), activation= 'relu', input_shape=(500, 500, 3)))
model.add(Conv2D(32, (3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation= 'relu'))
model.add(Conv2D(64, (3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='softmax'))


#adam = Adam(lr=1e-4,  beta_1=0.9, beta_2=0.999, epsilon=1e-08)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#%% train and validaton
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.0,
    zoom_range=0.0,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

if os.path.exists(path_weights):
    model.load_weights(path_weights)
    print('Starting train with %s ' % path_weights)
else:
    print('Strating train with no pre_weights')
    
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    callbacks = [ModelCheckpoint(path_weights)],
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights(path_weights)