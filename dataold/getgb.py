# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:48:40 2021

@author: ghedi
"""

import numpy as np
import cv2
from skimage import data, filters

# Open Video
cap = cv2.VideoCapture('cam4/background.avi')

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

# Display median frame
medianFrame = cv2.GaussianBlur(medianFrame,(5,5),0)
cv2.imshow('frame', medianFrame)
cv2.imwrite("background.png", medianFrame) 
# from PIL import Image
# im = Image.fromarray(medianFrame)
# im.save("background.png")

# import cv2
# vidcap = cv2.VideoCapture('cam3/background.avi')
# success,image = vidcap.read()
# count = 0
# while success:
#   cv2.imwrite("try.png" % count, image)     # save frame as JPEG file      
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1