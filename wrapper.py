'''
  File name: wrapper.py
  Authors: Brian Barrows, Zachary Fisher, Michael Woc
  Date created: 10/31/2019
'''

'''
  File clarification:
      This is the test script to execute all of the ".py" files in this folder.
      Adjust input images here if needed. 
    
'''

import numpy as np
import cv2
from corner_detector import *
from anms import *
from feat_desc import *
from feat_match import *
from ransac_est_homography import *
from mymosaic import *

# Import Images
imgL = cv2.imread('left.jpg')
imgM = cv2.imread('middle.jpg')
imgR = cv2.imread('right.jpg')

# Grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayM = cv2.cvtColor(imgM, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Feature Detection
[cimg] = corner_detector(grayL)

# Adaptive Non-Maximal Suppression
max_pts = 
[x,y,rmax] = anms(cimg, max_pts)

# Feature Descriptors
[descs] = feat_desc(grayL, x, y)

# Feature Matching
[match] = feat_match(descs1, descs2)

# RAndom Sampling Consensus (RANSAC)
[H, inlier_ind] = ransac_est_homography(x1, y1, x2, y2, thresh)

# Frame Mosaicing
[img_mosaic] = mymosaic(img_input)




# Show Image Code
#cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Left Image', 600, 600)
#cv2.imshow('Left Image',imgL)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
