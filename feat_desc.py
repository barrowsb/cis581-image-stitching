'''
  File name: feat_desc.py
  Author: Barrows, Fisher, & Woc
  Date created: 11/2/2019
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''

import numpy as np
from scipy.ndimage import gaussian_filter

def feat_desc(img, x, y):
    
    # Setup Parameters
    rows, cols = img.shape
    numPts, dims = x.shape
    matrix = np.zeros((8,8,numPts))
    descs = np.zeros((64,numPts))
    
    # Pad Input Image with Zeros
    img = np.pad(img, [20, 20], mode='constant')

    # Blur Image
    img = gaussian_filter(img, sigma=5)

    # Shift Input Coordinates To Be Within Padding
    x = x + 20
    y = y + 20    

    # Loop Through Interest Points
    for i in range(0,numPts):
        # Loop Through Rows
        for j in range(0,8):
            # Loop Through Cols
            for k in range(0,8):
                # Create 8x8 Patch from Maximums in 5x5's within 40x40
                    
                                
                '''
                # Old Version
                rowStart = int(y[i]-20+(j*5))
                rowEnd = int(y[i]-15+(j*5))
                colStart = int(x[i]-20+(k*5))
                colEnd = int(x[i]-15+(k*5))
                smallWindow = img[rowStart:rowEnd, colStart:colEnd]
                matrix[j,k,i] = np.max(smallWindow)
                # End of Old Version
                
                # New Version
                if (j <= 3):
                    row = int(y[i]-20+(j*5))
                else:
                    row = int(y[i]-15+(j*5))
                if (k <= 3):
                    col = int(x[i]-20+(k*5))
                else: 
                    col = int(x[i]-15+(k*5))
                matrix[j,k,i] = img[row, col]
                # End of New Version
                '''
                
                # Newest Version
                row = int(y[i]-18+(j*5))
                col = int(x[i]-18+(k*5))
                matrix[j,k,i] = img[row, col]
                # End of Newest Version
        
        # Normalize to Mean of 0 and Standard Deviation of 1
        matrix[:,:,i] = (matrix[:,:,i] - np.mean(matrix[:,:,i])) / np.std(matrix[:,:,i])
       
        # Reshape matrix to be 64xnumPts
        descs[:,i] = np.reshape(matrix[:,:,i], 64)
        
    return descs