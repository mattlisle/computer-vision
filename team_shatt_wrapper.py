'''
  File name: 
  Author: Shiv, Matt
  Date created: 10/25/2018
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from helpers import rgb2gray
from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc

path1 = "1L.jpg"
img1 = Image.open(path1)
img1 = np.array(img1)[..., :3]
gray1 = rgb2gray(img1)

max_pts = 50

cimg1 = corner_detector(gray1)
x1, y1, rmax1 = anms(cimg1, max_pts)
descs1, boxes1, oris1, ori1 = feat_desc(gray1, x1, y1)

# plt.imshow(img1)
# plt.scatter(x1, y1)
# for i in range(boxes1.shape[2]):
# 	plt.plot(boxes1[:,0,i], boxes1[:,1,i], color="red")
# 	plt.plot(oris1[0,:,i], oris1[1,:,i], color="green")
# plt.show()

path2 = "1M.jpg"
img2 = Image.open(path2)
img2 = np.array(img2)[..., :3]
gray2 = rgb2gray(img2)

cimg2 = corner_detector(gray2)
x2, y2, rmax2 = anms(cimg2, max_pts)
descs2, boxes2, oris2, ori2 = feat_desc(gray2, x2, y2)

# plt.imshow(img2)
# plt.scatter(x2, y2)
# for i in range(boxes2.shape[2]):
# 	plt.plot(boxes2[:,0,i], boxes2[:,1,i], color="red")
# 	plt.plot(oris2[0,:,i], oris2[1,:,i], color="green")
# plt.show()

fig, (left, right) = plt.subplots(1, 2, sharey=True)

left.imshow(img1)
left.scatter(x1, y1)
for i in range(boxes1.shape[2]):
	left.plot(boxes1[:,0,i], boxes1[:,1,i], color="red")
	left.plot(oris1[0,:,i], oris1[1,:,i], color="green")

right.imshow(img2)
right.scatter(x2, y2)
for i in range(boxes2.shape[2]):
	right.plot(boxes2[:,0,i], boxes2[:,1,i], color="red")
	right.plot(oris2[0,:,i], oris2[1,:,i], color="green")

plt.show()