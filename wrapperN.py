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
import matplotlib.pyplot as plt

# Import Images
#imgL = cv2.imread('left.jpg')
#imgM = cv2.imread('middle.jpg')
#imgR = cv2.imread('right.jpg')

#imgL = cv2.imread('eng_left.jpg')
#imgM = cv2.imread('eng_middle.jpg')
#imgR = cv2.imread('eng_right.jpg')

imgL = cv2.imread('new_left.jpg')
imgM = cv2.imread('new_middle.jpg')
imgR = cv2.imread('new_right.jpg')

#imgL = cv2.imread('lib_left.jpg')
#imgM = cv2.imread('lib_middle.jpg')
#imgR = cv2.imread('lib_right.jpg')

# Image Resizing
scale_percent = 50 # percent of original size
width = int(imgL.shape[1] * scale_percent / 100)
height = int(imgL.shape[0] * scale_percent / 100)
dim = (width, height)
imgL = cv2.resize(imgL, dim, interpolation = cv2.INTER_AREA)
imgM = cv2.resize(imgM, dim, interpolation = cv2.INTER_AREA)
imgR = cv2.resize(imgR, dim, interpolation = cv2.INTER_AREA)

# Grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayM = cv2.cvtColor(imgM, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

print("preprocessing complete")
#%%

# Feature Detection
cimgL = corner_detector(grayL)
cimgM = corner_detector(grayM)
cimgR = corner_detector(grayR)

print("corner detector complete")
#%%

# Adaptive Non-Maximal Suppression
max_pts = 200
xL,yL,rmaxL = anms(cimgL, max_pts)
xM,yM,rmaxM = anms(cimgM, max_pts)
xR,yR,rmaxR = anms(cimgR, max_pts)

# plot results
anmsL = plt.imshow(imgL)
plt.scatter(x=xL, y=yL, c='r', s=5)
plt.show()
#
anmsM = plt.imshow(imgM)
plt.scatter(x=xM, y=yM, c='r', s=5)
plt.show()
#
anmsR = plt.imshow(imgR)
plt.scatter(x=xR, y=yR, c='r', s=5)
plt.show()

print("anms complete")
#%%

# Feature Descriptors
descsL = feat_desc(grayL, xL, yL)
descsM = feat_desc(grayM, xM, yM)
descsR = feat_desc(grayR, xR, yR)

print("decriptors complete")
#%%

# Feature Matching
matchL = feat_match(descsM, descsL)
matchR = feat_match(descsM, descsR)

# Get feature coordiantes
x1ML = []
y1ML = []
x2L = []
y2L = []
x1MR = []
y1MR = []
x2R = []
y2R = []
for i in range(len(matchL)):
    if (matchL[i] != -1):
        x1ML.append(xM[int(i)][0])
        y1ML.append(yM[int(i)][0])
        x2L.append(xL[int(matchL[i])][0])
        y2L.append(yL[int(matchL[i])][0])
    if (matchR[i] != -1):
        x1MR.append(xM[int(i)][0])
        y1MR.append(yM[int(i)][0])
        x2R.append(xR[int(matchR[i])][0])
        y2R.append(yR[int(matchR[i])][0])

# Plot results
correspLM = plt.imshow(np.concatenate((imgL,imgM),axis=1))
plt.scatter(x=xL, y=yL, c='r', s=5)
plt.scatter(x=xM+width, y=yM, c='r', s=5)
for i in range(len(x1ML)):
    plt.plot([x2L[i],x1ML[i]+width],[y2L[i],y1ML[i]],'y-',linewidth=1)
plt.show()
#
correspMR = plt.imshow(np.concatenate((imgM,imgR),axis=1))
plt.scatter(x=xM, y=yM, c='r', s=5)
plt.scatter(x=xR+width, y=yR, c='r', s=5)
for i in range(len(x1MR)):
    plt.plot([x2R[i]+width,x1MR[i]],[y2R[i],y1MR[i]],'y-',linewidth=1)
plt.show()

print("feature matching complete")
#%%

# RAndom Sampling Consensus (RANSAC)
threshL = 0.5
threshR = 0.5
HL, inlier_indL = ransac_est_homography(x1ML, y1ML, x2L, y2L, threshL)
HR, inlier_indR = ransac_est_homography(x1MR, y1MR, x2R, y2R, threshR)

print("ransac complete")
print(HL)
print(HR)
#%%

# Frame Mosaicing
img_mosaic = mymosaic(imgL,imgM,imgR,HL,HR)

# Show Mosaic
cv2.namedWindow('Mosaic', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Mosaic', 600, 600)
cv2.imshow('Mosaic', img_mosaic.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()