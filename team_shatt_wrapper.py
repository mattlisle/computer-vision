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
from ransac_est_homography import ransac_est_homography

print("---------- Processing First Image ----------")
path1 = "street-2.jpg"
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
path2 = "street-3.jpg"
img2 = Image.open(path2)
img2 = np.array(img2)[..., :3]
gray2 = rgb2gray(img2)

print("Detecting corners")
cimg2 = corner_detector(gray2)

print("Suppressing non maxima")
x2, y2, rmax2 = anms(cimg2, max_pts)

print("Finding descriptors")
descs2, boxes2, oris2, ori2 = feat_desc(gray2, x2, y2)

print("---------- Processing Third Image ----------")
path3 = "street-3.jpg"
img3 = Image.open(path3)
img3 = np.array(img3)[..., :3]
gray3 = rgb2gray(img3)

print("Detecting corners")
cimg3 = corner_detector(gray3)

print("Suppressing non maxima")
x3, y3, rmax3 = anms(cimg3, max_pts)

print("Finding descriptors")
descs3, boxes3, oris3, ori3 = feat_desc(gray3, x3, y3)

print("---------- Matching 1st and 2nd Image ----------")
print("Finding matching descriptors")
match = feat_match(descs1, descs2)
# match = np.load("match0.npy")
matches1 = np.where([match >= 0])[1].astype(int)
matches2 = match[match >= 0].astype(int)

print("Performing RANSAC")
x1m = x1[matches1]
x2m = x2[matches2]
y1m = y1[matches1]
y2m = y2[matches2]
thresh = .5
H, inlier_ind = ransac_est_homography(x1m, y1m, x2m, y2m, thresh)
xin1 = x1m[inlier_ind]
xin2 = x2m[inlier_ind]
yin1 = y1m[inlier_ind]
yin2 = y2m[inlier_ind]

# plt.imshow(img2)
# plt.scatter(x2, y2)
# for i in range(boxes2.shape[2]):
# 	plt.plot(boxes2[:,0,i], boxes2[:,1,i], color="red")
# 	plt.plot(oris2[0,:,i], oris2[1,:,i], color="green")
# plt.show()

##########
# Plot both images with interest points side by side
# fig, (left, right) = plt.subplots(1, 2, sharey=True)

# left.imshow(img1)
# left.scatter(x1, y1)
# for i in range(boxes1.shape[2]):
# 	left.plot(boxes1[:,0,i], boxes1[:,1,i], color="red")
# 	left.plot(oris1[0,:,i], oris1[1,:,i], color="green")

# right.imshow(img2)
# right.scatter(x2, y2)
# for i in range(boxes2.shape[2]):
# 	right.plot(boxes2[:,0,i], boxes2[:,1,i], color="red")
# 	right.plot(oris2[0,:,i], oris2[1,:,i], color="green")

# plt.show()
##########

# fig, (left, middle, right) = plt.subplots(1, 2, sharey=True)

# left.imshow(img1)
# left.scatter(x1[matches1], y1[matches1], color="red")
# for i in matches1:
# 	left.plot(boxes1[:,0,i], boxes1[:,1,i], color="blue")
# 	left.plot(oris1[0,:,i], oris1[1,:,i], color="green")

# middle.imshow(img2)
# middle.scatter(x2[matches2], y2[matches2], color="red")
# for i in matches2:
# 	middle.plot(boxes2[:,0,i], boxes2[:,1,i], color="blue")
# 	middle.plot(oris2[0,:,i], oris2[1,:,i], color="green")

# right.imshow(img3)
# right.scatter(x3[matches2], y3[matches2], color="red")
# for i in matches3:
# 	right.plot(boxes3[:,0,i], boxes3[:,1,i], color="blue")
# 	right.plot(oris3[0,:,i], oris3[1,:,i], color="green")
# plt.show()

##########
# Plot the corresponding points before RANSAC
# both = np.concatenate((img1, img2), axis=1)
# mx = np.zeros(2)
# my = np.zeros(2)

# plt.imshow(both)
# plt.scatter(x1[matches1], y1[matches1])
# plt.scatter(x2[matches2] + img2.shape[1], y2[matches2], color="C0")
# for i in matches1:
# 	plt.plot(boxes1[:,0,i], boxes1[:,1,i], color="red")
# 	plt.plot(oris1[0,:,i], oris1[1,:,i], color="green")
	
# for j, i in enumerate(matches2):
# 	plt.plot(boxes2[:,0,i] + img2.shape[1], boxes2[:,1,i], color="red")
# 	plt.plot(oris2[0,:,i] + img2.shape[1], oris2[1,:,i], color="green")
# 	mx = [x1[matches1[j]], x2[i] + img1.shape[1]]
# 	my = [y1[matches1[j]], y2[i]]
# 	plt.plot(mx, my, color="orange")


# plt.show()
##########

##########
# Plot inliers from RANSAC
both = np.concatenate((img1, img2), axis=1)
plt.imshow(both)
plt.scatter(x1m, y1m, color="blue")
plt.scatter(x2m + img1.shape[1], y2m, color="blue")
plt.scatter(xin1, yin1, color="red")
plt.scatter(xin2 + img1.shape[1], yin2, color="red")
for i in inlier_ind:
	mx = [x1m[i], x2m[i] + img1.shape[1]]
	my = [y1m[i], y2m[i]]
	plt.plot(mx, my, color="orange")
plt.title("RANSAC Results Images 2, 3")
plt.show()

##########

# fig, (left, middle, right) = plt.subplots(1, 3, sharey=True)
# left.set_title("Left Image")
# middle.set_title("Middle Image")
# right.set_title("Right Image")
# left.imshow(img1)
# middle.imshow(img2)
# right.imshow(img3)
# left.scatter(x1, y1, color="red")
# middle.scatter(x2, y2, color="red")
# right.scatter(x3, y3, color="red")
# plt.show()


# fig, (left, middle, right) = plt.subplots(1, 3, sharey=True)

# left.set_title("Left Image")
# middle.set_title("Middle Image")
# right.set_title("Right Image")

# left.imshow(img1)
# left.scatter(x1, y1, color="red")
# for i in range(100):
# 	left.plot(boxes1[:,0,i], boxes1[:,1,i], color="blue")
# 	left.plot(oris1[0,:,i], oris1[1,:,i], color="green")

# middle.imshow(img2)
# middle.scatter(x2, y2, color="red")
# for i in range(100):
# 	middle.plot(boxes2[:,0,i], boxes2[:,1,i], color="blue")
# 	middle.plot(oris2[0,:,i], oris2[1,:,i], color="green")

# right.imshow(img3)
# right.scatter(x3, y3, color="red")
# for i in range(100):
# 	right.plot(boxes3[:,0,i], boxes3[:,1,i], color="blue")
# 	right.plot(oris3[0,:,i], oris3[1,:,i], color="green")
# plt.show()