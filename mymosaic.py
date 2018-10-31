'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''

def mymosaic(img_input):
  import numpy as np
  from helpers import rgb2gray
  from corner_detector import corner_detector
  from anms import anms
  from feat_desc import feat_desc
  from feat_match import feat_match
  from ransac_est_homography import ransac_est_homography
  import math
  
  # Set out constants
  max_pts = 2000
  thresh = 5
  h, w, d = img_input[0].shape

  # ---------- Part 1: Get descs for each image ---------- #

  # Initialize all the cell arrays that we will be using
  # For now, I'm only saving the variables that matter for later steps
  x = np.zeros(3, dtype=object)
  y = np.zeros(3, dtype=object)
  descs = np.zeros(3, dtype=object)
  
  # Get x, y, and descs for each image
  for i in range(3):
    print("---------- Processing Image %d ----------" % (i + 1))
    gray = rgb2gray(img_input[i])

    print("Detecting corners")
    cimg = corner_detector(gray)

    print("Suppressing non maxima")
    x[i], y[i], rmax = anms(cimg, max_pts)

    print("Finding descriptors")
    descs[i] = feat_desc(gray, x[i], y[i])

  # ---------- Part 2: Estimate homographies ---------- #

  # Initialize all the cell arrays that we will be using
  # For now, I'm only saving the variables that matter for later steps
  H = np.zeros(3, dtype=object)
  inlier_ind = np.zeros(3, dtype=object)
  corners = np.zeros(3, dtype=object)
  corners[1] = np.stack([np.array([0, w, w, 0]), np.array([0, 0, h, h]), np.ones(4)])
  H[1] = np.identity(3)

  for i in [0, 2]:
    print("---------- Matching Images %d and %d ----------" % (i + 1, 2))
    print("Finding matching descriptors")
    match = feat_match(descs[1], descs[i])
    matches = (np.where([match >= 0])[1], match[match >= 0])  # Potential bug

    print("Performing RANSAC")
    H[i], inlier_ind[i] = ransac_est_homography(x[i][matches[1]], y[i][matches[1]], x[1][matches[0]], y[1][matches[0]], thresh)
    
    # Find the boundaries that the mosaic has to fit into
    warped_corners = np.matmul(H[i], corners[1])
    corners[i] = warped_corners / warped_corners[2]

  # ---------- Part 3: Assemble the mosaic ---------- #
  # Initialize the mosaic using corners
  xmin = int(math.floor(np.amin(corners[0][0])))
  xmax = int(math.ceil(np.amax(corners[2][0])))
  ymin = int(math.floor(np.amin([np.amin(corners[0][1]), np.amin(corners[2][1])])))
  ymax = int(math.ceil(np.amax([np.amax(corners[0][1]), np.amax(corners[2][1])])))
  origin = np.array([-xmin, -ymin])
  print(xmin, xmax, ymin, ymax)
  # img_mosaic = np.zeros((ymax - ymin, xmax - xmin, 3))

  return H, corners

  # Need to find the mesh to interpolate with
  img_mosaic[0] = warp_image(img_input[0], H[0], corners[0])
  img_mosaic[1] = image_input[1]
  img_mosaic[2] = warp_image(img_input[2], H[2], corners[2])

  return img_mosaic
