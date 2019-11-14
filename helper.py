'''
  File name: helper.py
  Author: Barrows, Fisher, Woc
  Date created: Nov 13, 2019
'''

'''
  Function clarification: ignore_edge_pts()
    Deletes points from near edges of image from anms output.
    Inputs:
        x: N x 1 array of x-coordinates of corners
        y: N x 1 array of y-coordinates of corners
        w: width
        h: height
        border: the distance in pixels from edge within which corners will be deleted
    Outputs:
        X: n x 1 array of x-coordinates ignoring edge points with n<=N
        Y: n x 1 array of y-coordinates ignoring edge points with n<=N
'''

import numpy as np

def ignore_edge_pts(x_in,y_in,h,w,border): 
    
    x_out,y_out = [],[]
    
    for x,y in zip(x_in,y_in):
        if (x > border) & (x < w-border) & (y > border) & (y < h-border):
            x_out.append(x)
            y_out.append(y)
    
    X,Y = np.array(x_out),np.array(y_out)
    
    return X,Y