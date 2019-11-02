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

def feat_desc(img, x, y):

    import numpy as np
    import cv2
    
    # Setup Parameters
    rows, cols = img.shape
    numPts, dim = x.shape
    descs = np.zeros((8,8,numPts))
    
    # Pad Input Image with Zeros
    img = np.pad(img, [20, 20], mode='constant')

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
                rowStart = int(y[i]-20+(j*5))
                rowEnd = int(y[i]-15+(j*5))
                colStart = int(x[i]-20+(k*5))
                colEnd = int(x[i]-15+(k*5))
                smallWindow = img[rowStart:rowEnd, colStart:colEnd]
                descs[j,k,i] = np.max(smallWindow)
        # Normalize to Mean of 0 and Standard Deviation of 1
        descs[:,:,i] = (descs[:,:,i] - np.mean(descs[:,:,i])) / np.std(descs[:,:,i])
    
    return descs