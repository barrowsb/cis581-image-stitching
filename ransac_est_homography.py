'''
  File name: ransac_est_homography.py
  Author:
  Date created:
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
from est_homography import *

def ransac_est_homography(x1, y1, x2, y2, thresh):
  # Desired Number of Feature Pairs
  nPairs = 4
  # Desired Number of Iterations
  nRANSAC = 1000
  # Minimum Consensus
  min_Consensus = 10
  # Max Correspondence Error
  max_Error = 0.5
  # Generate Indexes
  pairIndex = list(range(0,len(x1)))
  for i in range(nRANSAC):
      # Choose 4 Random Indexes for Feature Pairs
      match_Sample = random.sample(pairIndex,nPairs)
      for j in range(0,nPairs):
          k = match_Sample[j]
      # Estimate Homography 
      H_Est = est_homography(x1[k],y1[k],x2[k],y2[k])
      
      # Compute Inliers
      im1_pts = [x1(:) y1(:)]
      im2_pts = [x2(:) y2(:)]
      qtrans1 = np.dot(H_Est,im1_pts)
      qtrans2 = np.dot(H_Est,im2_pts)
      # Calculate Distances and Compare to Threshold, must be less than threshold
      
  # Compute Homography with Inliers
  for i in range(nRANSAC):

      for j in range(0,nPairs):
          k = match_Sample[j]
      # Estimate Homography 
      H = est_homography(x1[k],y1[k],x2[k],y2[k])
  return H, inlier_ind