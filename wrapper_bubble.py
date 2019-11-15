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
import matplotlib.pyplot as plt
from corner_detector import *
from anms import *
from feat_desc import *
from feat_match import *
from ransac_est_homography import *
from mymosaic import *
from helper import *

# Import Images
imgL = cv2.imread('bubble_left.jpg')
imgM = cv2.imread('bubble_middle.jpg')
imgR = cv2.imread('bubble_right.jpg')

# Image Resizing
scale_percent = 640/imgL.shape[1]*100 # percent of original size
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

#%% Feature Detection

# run function
cimgL = corner_detector(grayL)
cimgM = corner_detector(grayM)
cimgR = corner_detector(grayR)

# plot results
#left
cd_L = plt.imshow(cimgL,cmap='gray')
plt.show()
# middle
cd_M = plt.imshow(cimgM,cmap='gray')
plt.show()
# right
cd_R = plt.imshow(cimgR,cmap='gray')
plt.show()

print("corner detector complete")

#%% ANMS

# call functions
max_pts = 300
xL,yL,rmaxL = anms(cimgL, max_pts)
xM,yM,rmaxM = anms(cimgM, max_pts)
xR,yR,rmaxR = anms(cimgR, max_pts)

# ignore features near edge
xL,yL = ignore_edge_pts(xL,yL,height,width,20)
xM,yM = ignore_edge_pts(xM,yM,height,width,20)
xR,yR = ignore_edge_pts(xR,yR,height,width,20)

# plot results
# left
anmsL = plt.imshow(imgL[:,:,[2,1,0]])
plt.scatter(x=xL, y=yL, c='r', s=5)
plt.show()
# middle
anmsM = plt.imshow(imgM[:,:,[2,1,0]])
plt.scatter(x=xM, y=yM, c='r', s=5)
plt.show()
# right
anmsR = plt.imshow(imgR[:,:,[2,1,0]])
plt.scatter(x=xR, y=yR, c='r', s=5)
plt.show()

print("anms complete")

#%% Feature Descriptors

descsLR = feat_desc(imgL[:,:,0], xL, yL)
descsLG = feat_desc(imgL[:,:,1], xL, yL)
descsLB = feat_desc(imgL[:,:,2], xL, yL)
descsL = np.concatenate((descsLR,descsLG,descsLB),axis=0)
descsMR = feat_desc(imgM[:,:,0], xM, yM)
descsMG = feat_desc(imgM[:,:,1], xM, yM)
descsMB = feat_desc(imgM[:,:,2], xM, yM)
descsM = np.concatenate((descsMR,descsMG,descsMB),axis=0)
descsRR = feat_desc(imgR[:,:,0], xR, yR)
descsRG = feat_desc(imgR[:,:,1], xR, yR)
descsRB = feat_desc(imgR[:,:,2], xR, yR)
descsR = np.concatenate((descsRR,descsRG,descsRB),axis=0)

print("decriptors complete")

#%% Feature Matching

# Call function
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
        x1ML.append(xM[int(i)])
        y1ML.append(yM[int(i)])
        x2L.append(xL[int(matchL[i])])
        y2L.append(yL[int(matchL[i])])
    if (matchR[i] != -1):
        x1MR.append(xM[int(i)])
        y1MR.append(yM[int(i)])
        x2R.append(xR[int(matchR[i])])
        y2R.append(yR[int(matchR[i])])

# Plot results
# Left & middle
plt.figure(figsize=(16,9))
correspLM = plt.imshow(np.concatenate((imgL,imgM),axis=1)[:,:,[2,1,0]])
plt.scatter(x=xL, y=yL, c='r', s=5)
plt.scatter(x=xM+width, y=yM, c='r', s=5)
for i in range(len(x1ML)):
    plt.plot([x2L[i],x1ML[i]+width],[y2L[i],y1ML[i]],'-',linewidth=1)
plt.show()
# Middle & right
plt.figure(figsize=(16,9))
correspMR = plt.imshow(np.concatenate((imgM,imgR),axis=1)[:,:,[2,1,0]])
plt.scatter(x=xM, y=yM, c='r', s=5)
plt.scatter(x=xR+width, y=yR, c='r', s=5)
for i in range(len(x1MR)):
    plt.plot([x2R[i]+width,x1MR[i]],[y2R[i],y1MR[i]],'-',linewidth=1)
plt.show()

## one-by-one (left & middle)
#for i in range(len(x1ML)):
#    print("left & middle #" + str(i))
#    plt.figure(figsize=(16,9))
#    correspLM = plt.imshow(np.concatenate((imgL,imgM),axis=1)[:,:,[2,1,0]])
#    plt.scatter(x=xL, y=yL, c='r', s=5)
#    plt.scatter(x=xM+width, y=yM, c='r', s=5)
#    plt.plot([x2L[i],x1ML[i]+width],[y2L[i],y1ML[i]],'y-',linewidth=2)
#    plt.show()
#    print()
#print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
## one-by-one (middle & right)
#for i in range(len(x1MR)):
#    print("middle and right #" + str(i))
#    plt.figure(figsize=(16,9))
#    correspMR = plt.imshow(np.concatenate((imgM,imgR),axis=1)[:,:,[2,1,0]])
#    plt.scatter(x=xM, y=yM, c='r', s=5)
#    plt.scatter(x=xR+width, y=yR, c='r', s=5)
#    plt.plot([x2R[i]+width,x1MR[i]],[y2R[i],y1MR[i]],'y-',linewidth=2)
#    plt.show()
#    print()

print("feature matching complete")

#%% Random Sampling Consensus (RANSAC)

# Call functions
threshL = 1
threshR = 1
HL, inlier_indL = ransac_est_homography(x2L, y2L, x1ML, y1ML, threshL)
HR, inlier_indR = ransac_est_homography(x2R, y2R, x1MR, y1MR, threshR)

# Plot results
# Left & middle
plt.figure(figsize=(16,9))
correspLM = plt.imshow(np.concatenate((imgL,imgM),axis=1)[:,:,[2,1,0]])
plt.scatter(x=xL, y=yL, c='b', s=5)
plt.scatter(x=xM+width, y=yM, c='b', s=5)
for i in range(len(x1ML)):
    if inlier_indL[i]==1:
        plt.plot([x2L[i],x1ML[i]+width],[y2L[i],y1ML[i]],'-',linewidth=1)
        plt.scatter(x=x2L[i], y=y2L[i], c='r', s=5)
        plt.scatter(x=x1ML[i]+width, y=y1ML[i], c='r', s=5)
plt.show()
# Middle & right
plt.figure(figsize=(16,9))
correspMR = plt.imshow(np.concatenate((imgM,imgR),axis=1)[:,:,[2,1,0]])
plt.scatter(x=xM, y=yM, c='b', s=5)
plt.scatter(x=xR+width, y=yR, c='b', s=5)
for i in range(len(x1MR)):
    if inlier_indR[i]==1:
        plt.plot([x2R[i]+width,x1MR[i]],[y2R[i],y1MR[i]],'-',linewidth=1)
        plt.scatter(x=x2R[i]+width, y=y2R[i], c='r', s=5)
        plt.scatter(x=x1MR[i], y=y1MR[i], c='r', s=5)
plt.show()

## one-by-one (left & middle)
#count = 1
#for i in range(len(x1ML)):
#    if inlier_indL[i]==1:
#        print("left & middle #" + str(count))
#        plt.figure(figsize=(16,9))
#        correspLM = plt.imshow(np.concatenate((imgL,imgM),axis=1)[:,:,[2,1,0]])
#        plt.scatter(x=xL, y=yL, c='r', s=5)
#        plt.scatter(x=xM+width, y=yM, c='r', s=5)
#        plt.plot([x2L[i],x1ML[i]+width],[y2L[i],y1ML[i]],'y-',linewidth=2)
#        plt.show()
#        count += 1
#        print()
#print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
## one-by-one (middle & right)
#count = 1
#for i in range(len(x1MR)):
#    if inlier_indR[i]==1:
#        print("middle and right #" + str(count))
#        plt.figure(figsize=(16,9))
#        correspMR = plt.imshow(np.concatenate((imgM,imgR),axis=1)[:,:,[2,1,0]])
#        plt.scatter(x=xM, y=yM, c='r', s=5)
#        plt.scatter(x=xR+width, y=yR, c='r', s=5)
#        plt.plot([x2R[i]+width,x1MR[i]],[y2R[i],y1MR[i]],'y-',linewidth=2)
#        plt.show()
#        count += 1
#        print()

print("HL:")
print(HL)
print("HR:")
print(HR)
print("ransac complete")

#%% Frame Mosaicing

# Call function
img_mosaic = mymosaic(imgL,imgM,imgR,HL,HR)
print("mosaic complete")

# Show Mosaic
cv2.namedWindow('Mosaic', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Mosaic', 1200, 400)
cv2.imshow('Mosaic', img_mosaic.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()