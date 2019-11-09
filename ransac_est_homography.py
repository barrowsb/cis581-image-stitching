'''
  File name: ransac_est_homography.py
  Author: Brian Barrows, Zachary Fisher, Michael Woc
  Date created: 11/3/2019
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''
import numpy as np
import random
from est_homography import est_homography

def ransac_est_homography(x1, y1, x2, y2, thresh):
  # Desired Number of Feature Pairs
  nPairs = 4
  # Desired Number of Iterations
  nRANSAC = 1000
  # Generate Indexes
  pairIndex = list(range(len(x1)))
  # T/F Vector for Inlier or Outlier, temporary
  inlier_indT = []
  # Index of Inlier Point
  inlier_idx = [] 
  # Number of Inliers
  nInliers = 0 
  
  im1_x = np.zeros(shape=(4,1))
  im1_y = np.zeros(shape=(4,1))
  im2_x = np.zeros(shape=(4,1))
  im2_y = np.zeros(shape=(4,1))
  
  for i in range(nRANSAC):
      # Choose 4 Random Indexes for Feature Pairs
      match_Sample = random.sample(pairIndex,nPairs)
      # range(0,nPairs)
      for j in range(nPairs):
          k = match_Sample[j]
          im1_x[j] = x1[k]
          im1_y[j] = y1[k]
          im2_x[j] = x2[k]
          im2_y[j] = y2[k]

      # Estimate Homography 
      H_Est = est_homography(im1_x,im1_y,im2_x,im2_y)
      
      # Compute Inliers
      for m in range(len(x1)):
          # Transpose Image 1 Point
          im1_pt = np.array([[x1[m]], [y1[m]], [1]])
          # Combine Image 2 Point Coordinates
          im2_pt = np.array([x2[m], y2[m]])
          # Transform Image 1 Point
          im1_T = np.dot(H_Est,im1_pt) 
          # Convert Homogenous Coordinates to X,Y coordinates
          #norm1 = np.array([float(im1_T[0])/float(im1_T[2]), float(im1_T[1])/float(im1_T[2])]) 
          norm1 = np.array([im1_T[0]/im1_T[2], im1_T[1]/im1_T[2]]) 
          # Calculate Distances and Error and Compare to Threshold, 
          # Must be less than or eqaul to threshold
          if (np.linalg.norm(norm1-im2_pt) <= thresh):
              inlier_idx.append(m) # Save index of inlier
              inlier_indT.append(1) # Denotes inlier
          else:
              inlier_indT.append(0) # Denotes outlier
              
      # If number of inliers is greater than before, update inliers list
      if (len(inlier_idx) > nInliers):
          inliers = inlier_idx # Save Inlier Index
          inlier_ind = inlier_indT # Save T/F Vector for Output
          nInliers = len(inlier_idx) # Update Current Number of Inliers
          
      # If number of inliers is less than 10, then increase iterations
      if ((i==nRANSAC) and (nInliers==9)):
          nRANSAC += 10
          print("Number of Iterations Increased, Not Enough Inliers")
          
     # print("Iteration Number:", i) # Keep Track of Number of Iterations
  
  # Get Coordinates for Inliers
  for i in range(len(inliers)):
      j = inliers[i]
      im1_xF = x1[j]
      im1_yF = y1[j]
      im2_xF = x2[j]
      im2_yF = y2[j]
  
  # Compute Homography with Inliers
  H = est_homography(im1_xF,im1_yF,im2_xF,im2_yF)
  
  return H, inlier_ind