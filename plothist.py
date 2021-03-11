# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:15:19 2021

@author: ghedi
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib qt


for i in range(4):
    plt.figure()
    fs = cv2.FileStorage('histogram' + str(i)+ '.yml', cv2.FILE_STORAGE_READ)
    mat = fs.getNode("hist").mat()
    
    plt.imshow(mat, aspect='auto')
    plt.figure()
    fs = cv2.FileStorage('histogram' + str(i+10)+ '.yml', cv2.FILE_STORAGE_READ)
    mat = fs.getNode("hist").mat()
    plt.imshow(mat, aspect='auto')

