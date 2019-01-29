# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:06:20 2019

@author: dan
"""
import numpy as np
import create_cell_cnn
import importlib
import matplotlib.pyplot as plt
from keras import optimizers
#Create cnn and display summary
importlib.reload(create_cell_cnn)
cell_net = create_cell_cnn.create_cell_cnn()
cell_net.summary()

#load image and target data from .m MATLAB files
import scipy.io as sio
images = sio.loadmat(r'C:\Users\VAMS_2\Dropbox\ML\cell_cnn\training_data\coins\trainImages')
images = images['trainImages']
images = np.transpose(images,(3,0,1,2))
images = np.float32(images/255)
targets = sio.loadmat(r'C:\Users\VAMS_2\Dropbox\ML\cell_cnn\training_data\coins\outVector')
targets = np.float32(targets['outVector'])

train_images = images[0:1801,:,:,:]
train_targets = targets[0:1801,:]

test_images = images[1801:,:,:,:]
test_targets = targets[1801:,:]

cell_net.compile(optimizer=optimizers.SGD(lr=0.05, momentum=0.9, decay=0.0, nesterov=False),
              loss='mse',
              metrics = ['mae'])

history = cell_net.fit(train_images,train_targets,
                         epochs = 1800,
                         batch_size = 32,
                         validation_data = (test_images, test_targets))

prediction = cell_net.predict(test_images)

import plot_prediction
importlib.reload(plot_prediction)
image_num = 1;
plot_prediction.plot_prediction(test_images[image_num,:,:,1], prediction[image_num,:], 7, 0.5)

#plot_prediction.plot_prediction(test_images[image_num,:,:,1], test_targets[image_num,:], 7, 0.5)
