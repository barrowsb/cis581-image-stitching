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

def mymosaic(img_left,img_middle,img_right,H12,H32):
  
  # Stitching From Right Image to Middle 
  # Dimensions of Each Image are Equal
  nRows, nCols, nCh = img_middle.shape
  padding = nCols
  
  # Zero Pad Middle Image on its Right Side 
  img_middle_new = np.hstack((img_middle, np.zeros((nRows, padding, nCh))))
  
  # Initialize Warped Right Image Size to Stitch onto Middle Image
  img_right_warp = np.zeros((nRows, nCols+padding, nCh))
  
  # Homography
  #pR_out = dot(H32, [pR_in[0], pR_in[1], 1])
  #homog_out_R = np.array(pR_out[0]/pR_out[2], pR_out[1]/pR_out[2])
  homog_out_R = np.convolve(H32, img_right)
  
  # Loop Through Color Channels and Warp the Right Side Image
  for c in range(nCh):
    img_right_warp[:,:,c] = geometric_transform(img_right[:,:,c], homog_out_R, (nRows, nCols+padding))
  
  # Calculate Alpha for Blending then Blend Warped Right Image with Middle Image
  alpha_right = ((img_right_warp[:,:,0] * img_right_warp[:,:,1] * img_right_warp[:,:,2]) > 0)
  for c in range(3):
    # Foreground Img is Multiplied with Alpha, Background with (1-alpha)
    img_middle_new[:,:,c] = img_right_warp[:,:,c] * alpha_right + img_middle_new[:,:,c] * (1 - alpha_right)
    
    
  # Stitching From Left Image to Updated Middle
  # Current Stitched Image Dimensions
  stitch_rows, stitch_cols, nCh = img_middle_new.shape
  
  # Affine Translation
  #H_delta = np.array([[1,0,0], [0,1,-nCols], [0,0,1]])
  #H12 = dot(H12,H_delta)
  
  # Zero Pad Middle Image on its Left Side 
  img_mosaic = np.hstack((zeros((nRows, padding, nCh)), img_middle_new))
  
  # Initialize Warped Left Image Size to Stitch onto Middle Image
  img_left_warp = np.zeros((stitch_rows, stitch_cols+padding, nCh))
  
  # Homography
  #pL_out = dot(H12, [pL_in[0], pL_in[1], 1])
  #homog_out_L = np.array(p_out[0]/p_out[2], p_out[1]/p_out[2])
  homog_out_L = np.convolve(H12, img_left)
  
  # Warp the Left Side Image
  for c in range(3):
    img_left_warp[:,:,c] = geometric_transform(img_left[:,:,c], homog_out_L, (stitch_rows, stitch_cols+padding))
    
  # Calculate Alpha for Blending then Blend Warped Left Image with Middle Image
  alpha_left = ((img_left_warp[:,:,0] * img_left_warp[:,:,1] * img_left_warp[:,:,2]) > 0)
  for c in range(3):
    # Foreground Img is Multiplied with Alpha, Background with (1-alpha)
    img_mosaic[:,:,c] = img_left_warp[:,:,c] * alpha + img_mosaic[:,:,c] * (1 - alpha_left)  
    
  return img_mosaic