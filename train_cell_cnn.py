# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:06:20 2019

@author: jbrowning
"""
import numpy as np
from create_cell_cnn_vgg16 import build_cnn
import importlib
import matplotlib.pyplot as plt
from keras import optimizers
import plot_prediction
import scipy.io as sio
#Create cnn and display summary

cell_net = build_cnn()
cell_net.summary()

#load image and target data from .mat MATLAB files

images = sio.loadmat(r'C:\Users\VAMS_2\Dropbox\ML\cell_cnn\training_data\coins\trainImages')
images = images['trainImages']
images = np.transpose(images,(3,0,1,2))
images = np.float32(images/255)
targets = sio.loadmat(r'C:\Users\VAMS_2\Dropbox\ML\cell_cnn\training_data\coins\outVector')
targets = np.float32(targets['outVector'])

#split data into training and validation 
train_images = images[0:1801,:,:,:]
train_targets = targets[0:1801,:]

test_images = images[1801:,:,:,:]
test_targets = targets[1801:,:]

#Compile and train network
epochs = 400
learning_rate = 0.01
decay_rate = learning_rate / epochs
momentum = 0.9
batch_size = 8

cell_net.compile(optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False),
#cell_net.compile(optimizer=optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=decay_rate),
              loss='mse',
              metrics = ['mae'])

history = cell_net.fit(train_images,train_targets,
                         epochs = epochs,
                         batch_size = batch_size,
                         validation_data = (test_images, test_targets))

prediction = cell_net.predict(test_images)

#Plot an image with prediction
importlib.reload(plot_prediction)
image_num = 3;
plot_prediction.plot_prediction(test_images[image_num,:,:,1], prediction[image_num,:], 7, 0.5)

#plot_prediction.plot_prediction(test_images[image_num,:,:,1], test_targets[image_num,:], 7, 0.5)
