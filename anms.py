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
    
    # Tune for thresholding
    thresh1 = 0.1 # pre-threshold
    thresh2 = 1.05 # anms-threshold
    
    # %% Pre-thresholding
    
    # Create meshgrids and flatten
    nr,nc = cimg.shape
    cols,rows = np.meshgrid(range(nc),range(nr))
    colsf = cols.flatten()
    rowsf = rows.flatten()
    cimgf = cimg.flatten()
    
    # Sort corners by decending corner-metric
    z = zip(cimgf,rowsf,colsf)
    z = sorted(z, key=lambda x: x[0],reverse=True)
    cimgf,rowsf,colsf = zip(*list(z))
    cimgf = list(cimgf)
    rowsf = list(rowsf)
    colsf = list(colsf)
    
    # Threshold
    prethresh = thresh1*cimgf[0]
    #cimgf[:] = [ele if ele > prethresh else 0 for ele in cimgf]
    cimgf = list(np.where(cimgf>prethresh, np.asarray(cimgf), np.zeros((nr*nc,))))
    cimgf = np.trim_zeros(cimgf)
    length = len(cimgf)
    rowsf = rowsf[0:length]
    colsf = colsf[0:length]
    
    # %% ANMS
    
    # Initialize variables
    # (for matrices: row index is reference corner, column index is corner being compared to)
    inf = (nr**2)+(nc**2) # longest possible distance^2
    greater = np.zeros((length,length),dtype=bool) # Boolean matrix (1 if above anms-thresh)
    dist2 = np.zeros((length,length)) # distance^2 matrix (from ref to compare corner)
    
    # Find all corners above anms-threshold
    compare,ref = np.meshgrid(cimgf,cimgf) # row-wise and column-wise matrices of cimgf vectors
    anmsthresh = ref*thresh2
    greater = compare > anmsthresh
    
    # Compute distance from all corners (ref) to all other corners (comp)
    rows_comp,cols_comp = np.meshgrid(rowsf,colsf) # row-wise and column-wise matrices of indices
    rows_ref = np.transpose(rows_comp) # (rows_ref does not change row-wise, rows_comp does)
    cols_ref = np.transpose(cols_comp) # (cols_ref does not change row-wise, cols_comp does)
    dist2 = (rows_ref - rows_comp)**2 + (cols_ref - cols_comp)**2

    # Find minimum distance to other corner above thresh
    logical_dist2 = np.where(greater,dist2,np.ones((length,length))*inf) # inf where not(greater)
    j_min = np.argmin(logical_dist2,axis=1) # col (j) indices of minimum distance in each row
    i_min =  np.asarray(range(length)) # row (i) indices
    min_dist2 = dist2[i_min,j_min] # minimum distance to sufficiently larger corner
    
    # Find row and col indices of minimum distance from above
    min_rows = np.asarray(rowsf)[j_min]
    min_cols = np.asarray(colsf)[j_min]
    
    # Sort by decending radius
    z2 = zip(min_dist2,min_rows,min_cols)
    z2 = sorted(z2, key=lambda x: x[0],reverse=True)
    min_dist2_sorted,min_rows_sorted,min_cols_sorted = zip(*list(z2))
    min_dist2_sorted = list(min_dist2_sorted)
    min_rows_sorted = list(min_rows_sorted)
    min_cols_sorted = list(min_cols_sorted)
    
    # Set outputs, trimmed to N = max_pts corners
    x = np.reshape(np.asarray(min_cols_sorted[0:max_pts]),(max_pts,1))
    y = np.reshape(np.asarray(min_rows_sorted[0:max_pts]),(max_pts,1))
    rmax = np.sqrt(min_dist2_sorted[max_pts])
    
    return x, y, rmax