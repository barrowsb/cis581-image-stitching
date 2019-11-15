CIS 581 - Project 3A
November 14, 2019 (4 late days)
Authors: Brian Barrows, Zachary Fisher, Michael Woc (Group 12)
README

Notes About Code:
•	Code will print status updates to the command window to indicate which steps
    of images stitching have been completed. Additionally, it will print plots
    of the intermittent results when applicable.

How to Run Code:
1.	Ensure the “Current Folder” Python is running in matches the folder the code
    and images are in. Also, ensure the images and code are in the same folder
    and not in sub-folders.
2.	Image stitching can be performed on the provided set of images of Franklin
    Field by running "wrapper_franklin.py". For the results on our own image
    set, run "wrapper_bubble.py". Expect results in ~30 seconds. The final
    mosaics will open in an OpenCV image window. RANSAC is non-deterministic
    and, as a result you may need to run more than once to produce good results.
