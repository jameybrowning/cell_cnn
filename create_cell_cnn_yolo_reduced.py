# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:40:41 2019
Creates a regression CNN absed on YOLO for detection of 
cells in microscopy images. 
@author jbrowning
"""
def build_cnn():
    from keras import layers
    from keras import models
    from keras import regularizers
    regularizer = regularizers.l2(0.0)
    #regularizer = 'none'
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5,5), activation = 'relu', kernel_regularizer = regularizer,
                            input_shape = (224,224,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(96, (3,3), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (1,1), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    model.add(layers.Conv2D(128, (3,3), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    model.add(layers.Conv2D(128, (1,1), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    model.add(layers.Conv2D(256, (3,3), activation = 'relu',kernel_regularizer = regularizer,
                            padding = 'same'))
    model.add(layers.MaxPooling2D((2,2)))
    #model.add(layers.Conv2D(112, (1,1), activation = 'relu', padding = 'same'))
    #model.add(layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
   # model.add(layers.Conv2D(112, (1,1), activation = 'relu', padding = 'same'))
    #model.add(layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.Conv2D(112, (1,1), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    model.add(layers.Conv2D(256, (3,3), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    #model.add(layers.Conv2D(112, (1,1), activation = 'relu', padding = 'same'))
    #model.add(layers.Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.Conv2D(256, (1,1), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    model.add(layers.Conv2D(512, (3,3), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    model.add(layers.MaxPooling2D((2,2)))
    #model.add(layers.Conv2D(256, (1,1), activation = 'relu', padding = 'same'))
    #model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.Conv2D(256, (1,1), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    #model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.Conv2D(512, (3,3), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    model.add(layers.Conv2D(512, (3,3), strides = (2,2), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same')) # needs to be stride 2
    model.add(layers.Conv2D(512, (3,3), activation = 'relu', kernel_regularizer = regularizer,
                            padding = 'same'))
    #model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation= 'relu', kernel_regularizer = regularizer))
    model.add(layers.Dense(147, activation='relu', kernel_regularizer = regularizer))
    
    #model.summary()
    return model


 

    
    
    
    