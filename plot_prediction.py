# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:35:04 2019

@author: dan

takes an mxm image,  1xk output vector, number of cells c,
and probability threshold p, and returns a list of [x,y] cell coordinates
relative to the upper left corner of an mxm image.
"""
import matplotlib.pyplot as plt
def plot_prediction(image, out_vector, c, p):
    plt.imshow(image)    
    
    cell_width = image.shape[0]/c
    for i in range(out_vector.size//3):
        if out_vector[i] > p:            
            xidx = i%c
            yidx = (i)//c
            x_cell_rel = out_vector[i+c**2]
            y_cell_rel = out_vector[i+2*c**2]
            x_coord = cell_width*(xidx+x_cell_rel)
            y_coord = cell_width*(yidx+y_cell_rel)
            plt.scatter(x_coord, y_coord, c='r', alpha=0.5)
    
    
    
          
        
 
    
    