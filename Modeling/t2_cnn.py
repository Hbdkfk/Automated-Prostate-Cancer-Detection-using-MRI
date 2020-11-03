"""
Author: Alex Hamilton - https://github.com/alexhamiltonRN
Created: 2018-11-18
Description: CNN training using t2 images from ProstateX challenge. 
"""

#import keras
#from keras import layers
#from keras import models
#from keras import regularizers
#from keras.utils import plot_model

import training_plots

import numpy as np
from pathlib import Path

import tensorflow
from tensorflow.keras.utils import plot_model
from tensorflow.keras import models 
from tensorflow.keras import layers
from tensorflow.keras import regularizers
#from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard


# LOADING THE DATA
t2_samples = np.load('/project2/msca/projects/ProstateMRI/data/PROSTATEx-team1/ProstateX/Train/DATAPREP/numpy-train/t2/X_train.npy')
t2_labels = np.load('/project2/msca/projects/ProstateMRI/data/PROSTATEx-team1/ProstateX/Train/DATAPREP/numpy-train/t2/Y_train.npy')

# CONVERT IMAGE SAMPLES TO FLOAT32 (REDUCE PRECISION FROM FLOAT64)
t2_samples_flt32 = np.array(t2_samples, dtype=np.float32, copy = True)

# RESHAPE IMAGE SAMPLES TO INCLUDE A SINGLE CHANNEL
x_train = t2_samples_flt32.reshape((1869,32,32,1))
y_train = t2_labels

# MODEL SPECIFICATION
model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3,3), padding = 'same', activation='relu', input_shape=(32,32,1))) 
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64, (3,3), padding = 'same', activation='relu'))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))
training_plots
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(128, kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001), activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# COMPILATION
opt = tensorflow.keras.optimizers.Adadelta()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# ask Keras to save best weights (in terms of validation loss) into file:
model_checkpoint = ModelCheckpoint(filepath='t2_cnn_weight_base.hdf5', monitor='val_loss', save_best_only=True)
# ask Keras to log each epoch loss:
csv_logger = CSVLogger('t2_base_log.csv', append=True, separator=';')
# ask Keras to log info in TensorBoard format:
tensorboard = TensorBoard(log_dir='t2_base/', write_graph=True, write_images=True)

# FIT
history = model.fit(x_train, y_train, epochs=100, validation_split=0.25, class_weight={0:1,1:2}, batch_size=80, shuffle=True)

# PLOT ACCURACY/VALIDATION CURVES
plot_model(model, to_file='t2_model.png', show_shapes = True)
training_plots.plot_metrics(history, 'T2')
