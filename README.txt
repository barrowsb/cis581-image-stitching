CIS 581 - Project 3A
November 14, 2019
Authors: Brian Barrows, Zachary Fisher, Michael Woc (Group 12)
README

Notes About Code:
•	Code will print status updates to the command window to indicate which steps of images stitching have been completed.

How to Run Code:
1.	Ensure the “Current Folder” Python is running in matches the folder the code and images is in. Also, ensure the images and code are in the same folder and not in sub-folders.
2.	Image stitching can be performed on each set of images using the same code by uncommenting the lines of code relevant to reading in each set of images.

For example: 
imgL = cv2.imread('new_left.jpg')
imgM = cv2.imread('new_middle.jpg')
imgR = cv2.imread('new_right.jpg')

Where ‘new_left.jpg’, ‘new_middle.jpg’, and ‘new_right.jpg’ represent the names of the images being read in. These three images constitute one set of images.
