'''
  File name: anms.py
  Author: Group 12
  Date created: 10/31/19
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''

import numpy as np

def anms(cimg, max_pts):
    
    # Create meshgrids and flatten
    nr,nc = cimg.shape
    cols,rows = np.meshgrid(range(nc),range(nr)) 
    colsf = cols.flatten()
    rowsf = rows.flatten()
    cimgf = cimg.flatten()
    
    # Sort corners by decending score
    z = zip(cimgf,rowsf,colsf)
    z = sorted(z, key=lambda x: x[0],reverse=True)
    cimgf,rowsf,colsf = zip(*list(z))
    cimgf = list(cimgf)
    rowsf = list(rowsf)
    colsf = list(colsf)
    
    # Threshold
    thresh = 0.2*cimgf[0]
    cimgf[:] = [ele if ele > thresh else 0 for ele in cimgf]
    cimgf = np.trim_zeros(cimgf)
    length = len(cimgf)
    rowsf = rowsf[0:length]
    colsf = colsf[0:length]
    
    # Find distance to nearest corner with higher score (1-1.2x as sure)
    
    
    # Sort by decending radius
    
    
    # Trim to N = max_pts corners
    
    
    
    return x, y, rmax