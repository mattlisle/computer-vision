'''
  File name: helpers.py
  Author: Matt Lisle
  Date created: 10/24/18
'''


def rgb2gray(rgb):
    import numpy as np
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def interp2(v, xq, yq):
	import numpy as np
	import pdb


	if len(xq.shape) == 2 or len(yq.shape) == 2:
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = v.shape[0]
	w = v.shape[1]
	if xq.shape != yq.shape:
		raise 'query coordinates Xq Yq should have same shape'


	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil = np.ceil(xq).astype(np.int32)
	y_ceil = np.ceil(yq).astype(np.int32)

	x_floor[x_floor<0] = 0
	y_floor[y_floor<0] = 0
	x_ceil[x_ceil<0] = 0
	y_ceil[y_ceil<0] = 0

	x_floor[x_floor>=w-1] = w-1
	y_floor[y_floor>=h-1] = h-1
	x_ceil[x_ceil>=w-1] = w-1
	y_ceil[y_ceil>=h-1] = h-1

	v1 = v[y_floor, x_floor]
	v2 = v[y_floor, x_ceil]
	v3 = v[y_ceil, x_floor]
	v4 = v[y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h,q_w)
	return interp_val


def warp_image(img, H, corners):
	import numpy as np
	from scipy.ndimage import map_coordinates
	import matplotlib.pyplot as plt
	import math

	xmin = math.floor(np.amin(corners[0]))
	xmax = math.ceil(np.amax(corners[0]))
	ymin = math.floor(np.amin(corners[1]))
	ymax = math.ceil(np.amax(corners[1]))

	x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
	h, w = x.shape

	# print(xmin, xmax, ymin, ymax)

	# plt.imshow(x)
	# plt.show()

	# Assignment suggested geometric_transform for this part, but I dunno how to use it, so I'll just brute-force vectorize
	pts = np.stack([x.reshape(h*w), y.reshape(h*w), np.ones(h*w)])

	Hinv = np.linalg.inv(H)
	Hinv = Hinv / Hinv[-1, -1]
	# print(corners.astype(int), np.matmul(Hinv, corners).astype(int))

	transformed = np.zeros((3, h * w))
	transformed = np.matmul(Hinv, pts)
	transformed = transformed / transformed[2]

	# t = transformed.astype(int)
	# plt.imshow(t[0].reshape(h, w))

	# h1, w1, d1 = img.shape
	# testx, testy = np.meshgrid(np.arange(0, w1), np.arange(0, h1))

	warped = np.zeros((h, w, 3))
	for c in range(3):
		# warped[..., c] = map_coordinates(img[..., c], [testy.reshape(h1*w1), testx.reshape(h1*w1)]).reshape(h1, w1)
		warped[..., c] = map_coordinates(img[..., c], [transformed[1], transformed[0]]).reshape(h, w)

	return warped.astype(int), ymin, xmax - xmin, ymax - ymin


def inlier_cost_func(H, x, y):
	import numpy as np

	H = H.reshape(3, 3)
	estimates = np.matmul(H, x)
	residuals = y - estimates / estimates[2]

	h, num_inliers = x.shape

	return residuals.reshape(h * num_inliers)