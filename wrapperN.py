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
cimgL = corner_detector(grayL)
cimgM = corner_detector(grayM)
cimgR = corner_detector(grayR)
# Adaptive Non-Maximal Suppression
max_pts = 100
xL,yL,rmaxL = anms(cimgL, max_pts)
xM,yM,rmaxM = anms(cimgM, max_pts)
xR,yR,rmaxR = anms(cimgR, max_pts)
# Feature Descriptors
descsL = feat_desc(grayL, xL, yL)
descsM = feat_desc(grayM, xM, yM)
descsR = feat_desc(grayR, xR, yR)

# Feature Matching
matchL = feat_match(descsM, descsL)
matchR = feat_match(descsM, descsR)

# Get feature coordiantes
for i in range(len(matchL)):
    if (matchL(i) != -1):
        x1ML = xM[matchL(i)]
        y1ML = yM[matchL(i)]
        x2L = xL[matchL(i)]
        y2L = xL[matchL(i)]
    else:
        
for i in range(len(matchR)):
    if (matchR(i) != -1):
        x1MR = xM[matchL(i)]
        y1MR = yM[matchL(i)]
        x2R = xR[matchR(i)]
        y2R = xR[matchR(i)]
    else:
        
# RAndom Sampling Consensus (RANSAC)
threshL = 0.5
threshR = 0.5
HL, inlier_indL = ransac_est_homography(x1ML, y1ML, x2L, y2L, threshL)
HR, inlier_indR = ransac_est_homography(x1MR, y1MR, x2R, y2R, threshR)

# Frame Mosaicing
img_mosaic = mymosaic(imgL,imgM,imgR,HL,HR)




# Show Image Code
#cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Left Image', 600, 600)
#cv2.imshow('Left Image',imgL)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
