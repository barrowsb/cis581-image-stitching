'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''

import numpy as np
import cv2
from numpy import linalg as LA

def feat_match(descs1, descs2):

  # Setup Parameters
  numPixels1, numPts1 = descs1.shape  # Where numPixels is 64
  numPixels2, numPts2 = descs2.shape
  match = np.zeros((numPts1,1))       # Initializing match outputs
  threshold = 0.4                     # Ratio Test Threshold
  
  # Loop through each feature description in descs1
  for i in range(0,numPts1):
      
      bestComparison = 100000             # Set to be easily replaced
      secondBestComparison = 100000       # Set to be easily replaced
      
      # Loop through each feature description in descs2
      for j in range(0,numPts2):          
          # Compare every descs2 to each individual descs1
          comparison = LA.norm(descs1[:,i] - descs2[:,j])
          # If the current comparison is the best thus far
          if (comparison < bestComparison):
              secondComparison = bestComparison
              bestComparison = comparison  # udpate the variable
              index = j                    # save the index
          # If the current comparison is not the best and better than the second best
          elif (comparison < secondBestComparison):
              secondBestComparison = comparison  # update the variable
              
      # Ratio Test between best and second best comparisons
      if (bestComparison/secondBestComparison < threshold):
          match[i] = index  # Record the index in descs2 that matches descs1
      else:   # If the ratio test was not passed
          match[i] = -1    # return that no match is found

  return match