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

def anms(cimg, max_pts):
  import numpy as np
  from scipy import signal

  # Initialize array of minimum radii
  minimum_r = np.ones(cimg.shape)
  
  # Will be needing these later
  h, w = cimg.shape

  # ---------- Part 1: Remove points that aren't local maxima from consideration --------- #
  # Get local maxima along each axis
  maxima_0 = signal.argrelextrema(cimg, np.greater, axis=0)
  maxima_1 = signal.argrelextrema(cimg, np.greater, axis=1)
  
  # Generate logical arrays based on where the maxima are
  bools_0 = np.zeros(cimg.shape, dtype=bool)
  bools_1 = np.zeros(cimg.shape, dtype=bool)
  bools_0[maxima_0] = True
  bools_1[maxima_1] = True

  # For the next function we need to elimiate corners close to the frame of the image
  border = np.ones((cimg.shape[0] - 40, cimg.shape[1] - 40), dtype=bool)
  border = np.pad(borders, ((20, 20), (20, 20)), mode="constant")

  # Points have to be maxima in both logical arrays
  bools = np.logical_and(border, np.logical_and(bools_0, bools_1))

  # ---------- Part 2: Loop through all points and find minimum radius, diagonals == 1 unit ---------- #
  for i in range(h):
    for j in range(w):

      # If it's not a local maxima, we can continue
      if bools[i, j]:

        # Generate local matrix based on maximum condition
        comps = cimg > cimg[i, j]
        comps[i, j] = False
        
        # Increase r, and check if any element within that r of (i, j) reads as true
        r = 1
        while not np.any(comps[max(0, i - r): min(h, i + r) + 1, max(0, j - r): min(w, j + r) + 1]) and not r == max(h, w):
          r += 1

        # Store r in the minimum radii array
        minimum_r[i, j] = r

  # ---------- Part 3: Filter our corners based on max_pts ---------- #
  # Initialize x and y with locations where points clear 8 nearest neighbors
  y, x = np.where(minimum_r > 1)
  r = minimum_r[minimum_r > 1]

  # Sort x, y based on radii in decreasing order
  sorter = np.argsort(-minimum_r[minimum_r > 1])
  x = x[sorter]
  y = y[sorter]
  r = r[sorter]

  # Construct final x, y, and rmax based on max_pts
  if max_pts > len(x):
    print("Actual number of points: " + str(len(x)))
    rmax = r[-1]
  else:
    x = x[:max_pts]
    y = y[:max_pts]
    rmax = r[max_pts - 1]

  return x, y, rmax


        # if minimum_r[i, j] > 0:
        #   up = max(0, i - r)
        #   down = min(h, h + r) + 1
        #   left = max(0, j - r)
        #   right = min(w, j + r) + 1
        #   comps = cimg[up: down, left: right] > cimg[i, j] * 0.9
        #   comps[r, r] = False
        #   if np.any(comps):
        #     minimum_r[i, j] = r

  # for i in range(h):
  #   for j in range(w):
  #     comps = cimg > cimg[i, j] * 0.9
  #     comps[i, j] = False
  #     rows, cols = np.where(comps)
  #     minimum_r[i, j] = min(np.amin(abs(rows - i)), np.amin(abs(cols - j)))
      
  #   print(i)
