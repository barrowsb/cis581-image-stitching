'''
File name: mymosaic.py
Author: Barrows, Fisher, Woc
Date created: Nov 7, 2019
'''

'''
File clarification:
Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
As a bonus, you can implement smooth image blending of the final mosaic.
- Input img_input: M elements numpy array or list, each element is a input image.
- Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''

import numpy as np
from scipy.ndimage import geometric_transform
from scipy.ndimage import convolve
import cv2
import matplotlib.pyplot as plt
from math import floor,ceil

def mymosaic(img_l,img_m,img_r,H12,H32):

    # Find inverse homography matrices
    H21 = np.linalg.inv(H12)
    H21 /= H21[2,2]
    H23 = np.linalg.inv(H32)
    H23 /= H23[2,2]
    
    # Create Callable to Satisfy geometric_transform input criteria
    def transformR(p):
        p2 = np.dot(H32, [p[0], p[1], 1])
        homog_out_R = (p2[0]/p2[2], p2[1]/p2[2])
        return homog_out_R
    
    # Create Callable to Satisfy geometric_transform input criteria
    def transformL(p):
        p2 = np.dot(H12, [p[0], p[1], 1])
        homog_out_L = (p2[0]/p2[2], p2[1]/p2[2])
        return homog_out_L
    
    # Create Callable to Satisfy geometric_transform input criteria
    def inv_transformR(p):
        p2 = np.dot(H23, [p[0], p[1], 1])
        homog_out_R = (p2[0]/p2[2], p2[1]/p2[2])
        return homog_out_R
    
    # Create Callable to Satisfy geometric_transform input criteria
    def inv_transformL(p):
        p2 = np.dot(H21, [p[0], p[1], 1])
        homog_out_L = (p2[0]/p2[2], p2[1]/p2[2])
        return homog_out_L
    
    def show(img):
        plt.imshow(img.astype(np.int))
        plt.show()
        
    # Create blank canvas with H/2 and W/2 padding each way
    h,w,nCh =img_m.shape
    zeros = np.zeros((h,w,nCh),dtype=np.uint8)
    canvas = np.hstack((np.vstack((zeros,zeros)),np.vstack((zeros,zeros))))
    canvas[int(h/2):int(h/2+h),int(w/2):int(w/2+w),:] = img_m
    nrows,ncols,nCh = canvas.shape
    
    #%% LEFT
    
    # Transform corners of left image
    xTL_l,yTL_l = transformL((0,0))
    xTR_l,yTR_l = transformL((w,0))
    xBL_l,yBL_l = transformL((0,h))
    xBR_l,yBR_l = transformL((w,h))
    
    # Find patch containing transformed left image
    xMax_l = ceil(max(xTL_l,xBL_l,xTR_l,xBR_l) + w/2)
    xMin_l = floor(min(xTL_l,xBL_l,xTR_l,xBR_l) + w/2)
    yMax_l = ceil(max(yTL_l,yBL_l,yTR_l,yBR_l) + h/2)
    yMin_l = floor(min(yTL_l,yBL_l,yTR_l,yBR_l) + h/2)
    patch_l = np.zeros(shape=canvas.shape[0:2])
    patch_l[yMin_l:yMax_l,xMin_l:xMax_l] = 1
    
    # Loop over pixels in padded target
    for r in range(0,nrows):
        for c in range(0,ncols):
            if patch_l[r,c] > 0:
                # Reverse transform to find corresponding source pixel in left image
                x,y = inv_transformL((c-w/2,r-h/2))
                x = int(x)
                y = int(y)
                
                # Add pixel to canvas
                if (x>0) & (y>0) & (x<w) & (y<h):
                    pixel = img_l[y,x,:]
                    canvas[r,c,:] = pixel
        
    #%% RIGHT
    
    # Transform corners of right image
    xTL_r,yTL_r = transformR((0,0))
    xTR_r,yTR_r = transformR((w,0))
    xBL_r,yBL_r = transformR((0,h))
    xBR_r,yBR_r = transformR((w,h))
    
    # Find patch containing transformed right image
    xMax_r = ceil(max(xTL_r,xBL_r,xTR_r,xBR_r) + w/2)
    xMin_r = floor(min(xTL_r,xBL_r,xTR_r,xBR_r) + w/2)
    yMax_r = ceil(max(yTL_r,yBL_r,yTR_r,yBR_r) + h/2)
    yMin_r = floor(min(yTL_r,yBL_r,yTR_r,yBR_r) + h/2)
    patch_r = np.zeros(shape=canvas.shape[0:2])
    patch_r[yMin_r:yMax_r,xMin_r:xMax_r] = 1
    
    # Loop over pixels in padded target
    for r in range(0,nrows):
        for c in range(0,ncols):
            if patch_r[r,c] > 0:
                # Reverse transform to find corresponding source pixel in right image
                x,y = inv_transformR((c-w/2,r-h/2))
                x = int(x)
                y = int(y)
                
                # Add pixel to canvas
                if (x>0) & (y>0) & (x<w) & (y<h):
                    pixel = img_r[y,x,:]
                    canvas[r,c,:] = pixel
                    
    return canvas
    
    #%% OLD METHOD
#    #%% Stitching From Right Image to Middle 
#    
#    # Dimensions
#    nRows, nCols, nCh = img_middle.shape
#    padding = nCols
#    
#    # Zero Pad Middle Image on its Right Side 
#    img_middle_new = np.hstack((img_middle, np.zeros((nRows, padding, nCh))))
#    show(img_middle_new)
#    
#    # Initialize Warped Right Image Size to Stitch onto Middle Image
#    img_right_warp = np.zeros(shape=(nRows, nCols+padding, nCh))
#    
#    # Loop Through Color Channels and Warp the Right Side Image
#    for c in range(nCh):
#        img_right_warp[:,:,c] = geometric_transform(img_right[:,:,c], transformR, (nRows, nCols+padding))
#    show(img_right_warp)
#    
#    # Calculate Alpha for Blending then Blend Warped Right Image with Middle Image
#    alpha_right = ((img_right_warp[:,:,0] * img_right_warp[:,:,1] * img_right_warp[:,:,2]) > 0)
#    for c in range(nCh):
#        # Foreground Img is Multiplied with Alpha, Background with (1-alpha)
#        img_middle_new[:,:,c] = img_right_warp[:,:,c] * (alpha_right) + img_middle_new[:,:,c] * (1 - alpha_right)    
#    show(img_middle_new)
#    
#    #%% Stitching From Left Image to Updated Middle
#    
#    # Current Stitched Image Dimensions
#    stitch_rows, stitch_cols, nCh = img_middle_new.shape
#    
#    # Zero Pad Middle Image on its Left Side 
#    img_mosaic = np.hstack((np.zeros((nRows, padding, nCh)), img_middle_new))
#    show(img_mosaic)
#    
#    # Initialize Warped Left Image Size to Stitch onto Middle Image
#    img_left_warp = np.zeros((stitch_rows, stitch_cols+padding, nCh))
#    
#    # Warp the Left Side Image
#    H_delta = np.array([[1,0,0], [0,1,-padding], [0,0,1]]) # Affine Translation
#    H12 = np.dot(H12,H_delta)
#    #nRows_left, nCols_left, nCh = img_left.shape
#    #img_left_warp = np.zeros(shape=(nRows_left, nCols_left+(padding*2), nCh))
#    for c in range(nCh):
#        img_left_warp[:,:,c] = geometric_transform(img_left[:,:,c], transformL, (stitch_rows, stitch_cols+padding))
#    show(img_left_warp)
#    
#    # Calculate Alpha for Blending then Blend Warped Left Image with Middle Image
#    alpha_left = ((img_left_warp[:,:,0] * img_left_warp[:,:,1] * img_left_warp[:,:,2]) > 0)
#    for c in range(nCh):
#        # Foreground Img is Multiplied with Alpha, Background with (1-alpha)
#        img_mosaic[:,:,c] = img_left_warp[:,:,c] * alpha_left + img_mosaic[:,:,c] * (1 - alpha_left)  
#    show(img_mosaic)
#    
#    return img_mosaic