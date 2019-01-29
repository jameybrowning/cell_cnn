# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:40:41 2019

@author 
"""
def create_cell_cnn_inceptionv3():
    from keras.applications.inception_v3 import InceptionV3
    from keras import layers
    from keras import models
    
    input_tensor = layers.Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
    base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(147, activation='relu')(x)
    
    head_model = models.Model(input=base_model.input, output = x)
    
    head_model.summary()
    return head_model



 

    
    
    
    