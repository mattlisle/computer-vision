'''
  File name: group3_wrapper.py
  Author: Shiv, Matt
  Date created: 10/25/2018
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mymosaic import mymosaic

# I'm going to use 3 x H x W x D array for images
# May want to double check this is the right format at debug session
paths = ["intersection1500-1.jpg", "intersection1500-2.jpg", "intersection1500-3.jpg"]
img_input = np.zeros(3, dtype=object)
for i in range(3):
	img_input[i] = np.array(Image.open(paths[i]))

H, corners = mymosaic(img_input)