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
from feat_match import feat_match

print("---------- Processing First Image ----------")
path1 = "intersection1500-1.jpg"
img1 = Image.open(path1)
img1 = np.array(img1)[..., :3]
gray1 = rgb2gray(img1)

max_pts = 1000

print("Detecting corners")
cimg1 = corner_detector(gray1)

print("Suppressing non maxima")
x1, y1, rmax1 = anms(cimg1, max_pts)

print("Finding descriptors")
descs1, boxes1, oris1, ori1 = feat_desc(gray1, x1, y1)

# plt.imshow(img1)
# plt.scatter(x1, y1)
# for i in range(boxes1.shape[2]):
# 	plt.plot(boxes1[:,0,i], boxes1[:,1,i], color="red")
# 	plt.plot(oris1[0,:,i], oris1[1,:,i], color="green")
# plt.show()

print("---------- Processing Second Image ----------")
path2 = "intersection1500-2.jpg"
img2 = Image.open(path2)
img2 = np.array(img2)[..., :3]
gray2 = rgb2gray(img2)

print("Detecting corners")
cimg2 = corner_detector(gray2)

print("Suppressing non maxima")
x2, y2, rmax2 = anms(cimg2, max_pts)

print("Finding descriptors")
descs2, boxes2, oris2, ori2 = feat_desc(gray2, x2, y2)

print("---------- Stitching Images ----------")
print("Finding matching descriptors")
match = feat_match(descs1, descs2)
matches1 = np.where([match >= 0])[1].astype(int)
matches2 = match[match >= 0].astype(int)

# plt.imshow(img2)
# plt.scatter(x2, y2)
# for i in range(boxes2.shape[2]):
# 	plt.plot(boxes2[:,0,i], boxes2[:,1,i], color="red")
# 	plt.plot(oris2[0,:,i], oris2[1,:,i], color="green")
# plt.show()

##########
# Plot both images with interest points side by side
fig, (left, right) = plt.subplots(1, 2, sharey=True)

left.imshow(img1)
left.scatter(x1, y1)
# for i in range(boxes1.shape[2]):
# 	left.plot(boxes1[:,0,i], boxes1[:,1,i], color="red")
# 	left.plot(oris1[0,:,i], oris1[1,:,i], color="green")

right.imshow(img2)
right.scatter(x2, y2)
# for i in range(boxes2.shape[2]):
# 	right.plot(boxes2[:,0,i], boxes2[:,1,i], color="red")
# 	right.plot(oris2[0,:,i], oris2[1,:,i], color="green")

plt.show()
##########

# fig, (left, right) = plt.subplots(1, 2, sharey=True)

# left.imshow(img1)
# left.scatter(x1[matches1], y1[matches1])
# for i in matches1:
# 	left.plot(boxes1[:,0,i], boxes1[:,1,i], color="red")
# 	left.plot(oris1[0,:,i], oris1[1,:,i], color="green")

# right.imshow(img2)
# right.scatter(x2[matches2], y2[matches2])
# for i in matches2:
# 	right.plot(boxes2[:,0,i], boxes2[:,1,i], color="red")
# 	right.plot(oris2[0,:,i], oris2[1,:,i], color="green")
# plt.show()

both = np.concatenate((img1, img2), axis=1)
mx = np.zeros(2)
my = np.zeros(2)

plt.imshow(both)
plt.scatter(x1[matches1], y1[matches1])
plt.scatter(x2[matches2] + img2.shape[1], y2[matches2], color="blue")
for i in matches1:
	plt.plot(boxes1[:,0,i], boxes1[:,1,i], color="red")
	plt.plot(oris1[0,:,i], oris1[1,:,i], color="green")
	
for j, i in enumerate(matches2):
	plt.plot(boxes2[:,0,i] + img2.shape[1], boxes2[:,1,i], color="red")
	plt.plot(oris2[0,:,i] + img2.shape[1], oris2[1,:,i], color="green")
	mx = [x1[matches1[j]], x2[i] + img1.shape[1]]
	my = [y1[matches1[j]], y2[i]]
	plt.plot(mx, my, color="orange")


plt.show()