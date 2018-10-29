'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''

# Remaining elements to implement
# 1. Make argrelextrema work on values 0.9 as large - investigate rank filtering
# 2. I forget the second...

def anms(cimg, max_pts):
  import numpy as np
  from time import time
  from scipy import signal

  # Initialize array of minimum radii
  minimum_r = np.ones(cimg.shape)
  
  # Will be needing these later
  h, w = cimg.shape

  # ---------- Part 1: Find points that are local maxima --------- #
  # Define 4 kernels that will allow us to compare to 4-nearest pixel neighbors
  left  = np.array([ 0, 0.9, -1]).reshape(1, 3)
  right = np.array([-1, 0.9,  0]).reshape(1, 3)
  up    = np.array([ 0, 0.9, -1]).reshape(3, 1)
  down  = np.array([-1, 0.9,  0]).reshape(3, 1)

  # Generate comparison array, one array along 0th dim for each neighbor
  comps = np.zeros((4, h, w))
  comps[0:, ...] = signal.convolve2d(cimg,  left, mode="same")
  comps[1:, ...] = signal.convolve2d(cimg, right, mode="same")
  comps[2:, ...] = signal.convolve2d(cimg,    up, mode="same")
  comps[3:, ...] = signal.convolve2d(cimg,  down, mode="same")

  # Use comps to create 2d array of local maxima
  max_locs = np.all(comps > 0, axis=0)

  # ---------- Part 2: Loop through all points and find radii ---------- #
  # Initialize x and y with locations where points clear 4 nearest neighbors
  y, x = np.where(max_locs)
  values = cimg[max_locs]
  # print(len(values))

  # Sort these values in decreasing order
  sorter = np.argsort(-values)
  x = x[sorter]
  y = y[sorter]
  values = values[sorter]

  # Initialize array of radii for each interest point, already know value for first pt
  radii = np.zeros(len(values))
  radii[0] = np.nan_to_num(np.Inf)
  
  # Compute the Euclidean distance of every interest point to every other interest point
  # x1, x2 = np.meshgrid(x, x)
  # y1, y2 = np.meshgrid(y, y)
  # distances = np.abs(x1 - x2) + np.abs(y1 - y2)

  # Compute the Euclidean distance of every interest point to every other interest point
  distances = np.zeros(len(values))
  for i in range(1, len(values)):
    distances = np.sqrt(np.square(x[:i] - x[i]) + np.square(y[:i] - y[i]))
    radii[i] = np.amin(distances)

  # Take advantage of sorted order to ignore values less than the value of the interest point for each row
  # distances = np.triu(distances)

  # Finally, fill in the rest of the radii
  # radii[1:] = np.amin(distances, axis=0)[1:]

  # ---------- Part 3: Construct outputs based on max_pts ---------- #
  sorter = np.argsort(-radii)
  x = x[sorter]
  y = y[sorter]
  radii = radii[sorter]

  # If we've asked for more than we've got, let the user know
  if max_pts > len(x):
    print("Actual number of points: " + str(len(x)))
    rmax = radii[-1]

  # Otherwise cut out the fat and index the max radius
  else:
    x = x[:max_pts]
    y = y[:max_pts]
    rmax = radii[max_pts - 1]

  return x, y, rmax
