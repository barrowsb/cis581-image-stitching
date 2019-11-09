'''
  File name: corner_detector.py
  Author: Brian Barrows, Zachary Fisher, Michael Woc
  Date created: 10/31/2019
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
import cv2

def corner_detector(img):
    
  # Calcuate Corners  
  cimg = cv2.cornerHarris(img,2,3,0.04)

  return cimg