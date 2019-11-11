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
from interp2 import *


def mymosaic(img_left,img_middle,img_right,H12,H32):
  
  #H12 = np.linalg.inv(H12)
  #H32 = np.linalg.inv(H32)  
    
  #H12 = H12/H12[2,2]
  #H32 = H32/H32[2,2]
    
  # Stitching From Right Image to Middle 
  # Dimensions of Each Image are Equal
  nRows, nCols, nCh = img_middle.shape
  padding = 2400 # or maybe nCols
  
  # Zero Pad Middle Image on its Right Side 
  img_middle_new = np.hstack((img_middle, np.zeros((nRows, padding, nCh))))
  
  # Initialize Warped Right Image Size to Stitch onto Middle Image
  img_right_warp = np.zeros((nRows, nCols+padding, nCh))

  # Loop Through Color Channels and Warp the Right Side Image
  nRows_right, nCols_right, nCh = img_right.shape
  img_right_warp = np.zeros(shape=(nRows_right, nCols_right, nCh))
  
  # removed +padding from nCols_right above
  
  dim = nRows_right*nCols_right
  u_x = np.zeros(shape=(dim,1))
  u_y = u_x
  u_z = u_x
  for r in range(nRows_right):
      for c in range(nCols_right):
          index = nCols_right*r+c
          u_x[index] = H32[0,0]*r + H32[0,1]*c + H32[0,2]
          u_y[index] = H32[1,0]*r + H32[1,1]*c + H32[1,2]
          u_z[index] = H32[2,0]*r + H32[2,1]*c + H32[2,2]
          
  u_x = np.divide(u_x, u_z)
  u_y = np.divide(u_y, u_z)
  print(u_x.shape)
  print(nRows_right)
  print(nCols_right)
  print(padding)
  
  for ch in range(nCh):
    img_right_warp[:,:,ch] = np.reshape(interp2(img_right[:,:,ch], u_x, u_y), (nRows_right, nCols_right))
    # removed +padding from nCols_right above
  img_right_warp = img_right_warp.astype(np.uint8)
    
  return  img_right_warp
'''
  #img_right_warp = np.hstack((np.zeros((nRows, padding, nCh)), img_right_warp))  
  
  # Calculate Alpha for Blending then Blend Warped Right Image with Middle Image
  alpha_right = ((img_right_warp[:,:,0] * img_right_warp[:,:,1] * img_right_warp[:,:,2]) > 0)
  for c in range(nCh):
    # Foreground Img is Multiplied with Alpha, Background with (1-alpha)
    img_middle_new[:,:,c] = img_right_warp[:,:,c] * (alpha_right) + img_middle_new[:,:,c] * (1 - alpha_right)    


  # Stitching From Left Image to Updated Middle
  # Current Stitched Image Dimensions
  stitch_rows, stitch_cols, nCh = img_middle_new.shape
  
  # Affine Translation
  H_delta = np.array([[1,0,0], [0,1,-stitch_cols], [0,0,1]])
  H12 = np.dot(H12,H_delta)
  
  # Zero Pad Middle Image on its Left Side 
  img_mosaic = np.hstack((np.zeros((nRows, padding, nCh)), img_middle_new))
  
  # Initialize Warped Left Image Size to Stitch onto Middle Image
  img_left_warp = np.zeros((stitch_rows, stitch_cols+padding, nCh))
    
  # Create Callable to Satisfy geometric_transform input criteria
  def transformL(p): 
    p2 = np.dot(H12, [p[0], p[1], 1])
    homog_out_L = (p2[0]/p2[2], p2[1]/p2[2])
    return homog_out_L
  
  # Warp the Left Side Image
  nRows_left, nCols_left, nCh = img_left.shape
  img_left_warp = np.zeros(shape=(nRows_left, nCols_left+(padding*2), nCh))
  for c in range(nCh):
    #img_left_warp[:,:,c] = convolve(img_left[:,:,c], H12, mode='constant')
    img_left_warp[:,:,c] = geometric_transform(img_left[:,:,c], transformL, (stitch_rows, stitch_cols+padding))
  
  #img_left_warp = np.hstack((img_left_warp, np.zeros((nRows, padding*2, nCh))))

  # Calculate Alpha for Blending then Blend Warped Left Image with Middle Image
  alpha_left = ((img_left_warp[:,:,0] * img_left_warp[:,:,1] * img_left_warp[:,:,2]) > 0)
  for c in range(nCh):
    # Foreground Img is Multiplied with Alpha, Background with (1-alpha)
    img_mosaic[:,:,c] = img_left_warp[:,:,c] * alpha_left + img_mosaic[:,:,c] * (1 - alpha_left)  
    
    
  return img_mosaic
'''