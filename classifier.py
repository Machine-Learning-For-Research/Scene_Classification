# -*- coding: utf-8 -*-
#!/user/bin/env python3
import os
import h5py
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras import initializers
from keras.layers.normalization import BatchNormalization
#from keras.utils import multi_gpu_model
#%% Parameters
train_data_dir = '/home/winter/Keras/SceneClassify_Winter/data/train/train_data'
validation_data_dir = '/home/winter/Keras/SceneClassify_Winter/data/validation/validation_data'
path_weights = 'params/inception_resnet_v2/{epoch:05d}-{val_loss:.4f}-{val_acc:.4f}.h5'
path_summary = 'logs/'
img_width, img_height = 299, 299
batch_size = 32
nb_train_samples = 53879
nb_validation_samples = 7210
epochs = 50
nb_classes =80
#%%
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
nadam = Nadam(lr=0.002, beta_1 = 0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004)
#  add new layers
def add_new_last_layer(base_model, nb_classes):
  """
  input: base_model and clasess
  ouput: new keras model
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)

  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, output=predictions)
  model.summary()
  return model

# free the base_model layers
def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
      layer.trainable = False
  model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])
    
 
#%% bulid model
#base_model = VGG16(include_top=False, weights='imagenet', input_tensor=None) # download no_top model
#base_model = Xception(include_top=False, weights='imagenet')
base_model = InceptionResNetV2(include_top=False, weights='imagenet')
model = add_new_last_layer(base_model, nb_classes)
setup_to_transfer_learn(model, base_model)

#base_model = ResNet50(include_top=False, weights=path_weights, input_tensor=None, input_shape=None, pooling=None, 
#                 classes=80)
#
#base_model = InceptionResNetv2(include_top=False, weights=path_weights, input_tensor=None, input_shape=None, pooling=None, 
#                 classes=80)


################################sequential to build model#################################
#with tf.device('/cpu:0'):
#model = Sequential()
##################################################################################
#adam = Adam(lr=1e-4,  beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#parallel_model = multi_gpu_model(model, gpus=2)
#rmsprop = RMSprop(lr = 0.01, rho=0.9, epsilon=1e-8)
#model.compile(loss='categorical_crossentropy',
#              optimizer='sgd',
#              metrics=['accuracy'])

#%% train and validaton
train_datagen = ImageDataGenerator(#rescale=1. / 255,
                                   channel_shift_range=0.1,
                                   rotation_range=15,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='reflect',
                                   preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(#rescale=1. / 255,
                                  fill_mode='reflect',
                                  preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
#%%
if os.path.exists(path_weights):
    model.load_weights(path_weights)
    print('Starting train with %s ' % path_weights)
else:
    print('Strating train with no pre_weights')
    
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    callbacks = [ModelCheckpoint(path_weights,),
                 TensorBoard(log_dir= path_summary)], #every epoch save model
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights(path_weights)