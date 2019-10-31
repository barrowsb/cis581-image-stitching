'''
  File name: feat_desc.py
  Author:
  Date created:
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
    descs = np.zeros(8)
    
    # Using SIFT
    #sift = cv2.xfeatures2d.SIFT_create
    #kp, descs = sift.compute(img,None)
    
    # Gaussian Distribution of Image
    #blur = cv2.GaussianBlur(img,(5,5),0)
    
    # Gaussian filter definition
    G = [2,4,5,4,2;4,9,12,9,4;5,12,15,12,5;4,9,12,9,4;2,4,5,4,2]
    G = 1/159.*G
    
    # Filter for horiztonal and vertical direction
    dx, dy = gradient(G)
    
    # Convolution of image with Gaussian
    Gx = conv2(G, dx, 'same')
    Gy = conv2(G, dy, 'same')
    
    # Convolution of image with Gx and Gy
    Magx = conv2(img, Gx, 'same')
    Magy = conv2(img, Gy, 'same')
    
    # Gradient Magnitude
    Mag = sqrt(Magy.^2 + Magx.^2)
    
    # Gradient Angle
    Ori = atan2(Magy, Magx)
    
    # Sample from a 40x40 window to get 8x8 pixels
    # Loop Through Interest Points
    '''
    for i in y:
        # Loop Through Rows of Patch Window
        for j in 8:
            # Loop Through Cols of Patch Window
            for k in 8:
                descs(j,k,i) = img(y(i-20,1), x(i-20,1))
                -20,-20     -20,-15     -20,-10
    '''
            
    
    # Normalize to Mean of 0
  
    # Normalize to Standard Deviation of 1
  
  
    return descs