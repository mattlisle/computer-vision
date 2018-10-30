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
  
  # Set out constants
  max_pts = 2000
  thresh = 0.7
  h, w, d = img_input[0].shape
  corners = np.stack([np.array([0, w, w, 0]), np.array([0, 0, h, h]), np.ones(4)])

  # Initialize all the cell arrays that we will be using
  # For now, I'm only saving the variables that matter for later steps
  x = np.zeros(3, dtype=object)
  y = np.zeros(3, dtype=object)
  descs = np.zeros(3, dtype=object)
  
  # Get x, y, and descs for each image
  for i in range(3):
    print("---------- Processing Image %d ----------" % i + 1)
    gray = rgb2gray(img_input[i])

    print("Detecting corners")
    cimg = corner_detector(gray)

    print("Suppressing non maxima")
    x[i], y[i], rmax = anms(cimg, max_pts)

    print("Finding descriptors")
    descs[i] = feat_desc(gray, x[i], y[i])

  # Initialize all the cell arrays that we will be using
  # For now, I'm only saving the variables that matter for later steps
  H = np.zeros(2, dtype=object)
  xmin = 0
  xmax = w
  ymin = 0
  ymax = h

  print("---------- Matching Images %d and %d ----------" % (1, 2))
  print("Finding matching descriptors")
  match = feat_match(descs[1], descs[0])
  matches = (np.where([match >= 0])[1], match[match >= 0])  # Potential bug

  print("Performing RANSAC")
  H, inlier_ind = ransac_est_homography(x[1][matches[0]], y[1][matches[0]], x[0][matches[1]], y[0][matches[1]], thresh)

  # Find the boundaries that the mosaic has to fit into
  warped_corners = np.matmul(H, corners)
  warped_corners = warped_corners / warped_corners[2]
  xmin = np.amin([0, np.amin(warped_corners[0])])
  ymin = np.amin([0, np.amin(warped_corners[1])])
  ymax = np.amax([h, np.amax(warped_corners[1])])
  
    
  print("---------- Matching Images %d and %d ----------" % (2, 3))
  print("Finding matching descriptors")
  match = feat_match(descs[1], descs[2])
  matches = (np.where([match >= 0])[1], match[match >= 0])

  print("Performing RANSAC")
  H, inlier_ind = ransac_est_homography(x[1][matches[0]], y[1][matches[0]], x[2][matches[1]], y[2][matches[1]], thresh)
    
  # Find the boundaries that the mosaic has to fit into
  warped_corners = np.matmul(H, corners)
  warped_corners = warped_corners / warped_corners[2]
  xmax = np.amax([w, np.amax(warped_corners[0])])
  ymin = np.amin([ymin, np.amin(warped_corners[1])])
  ymax = np.amax([ymax, np.amax(warped_corners[1])])

  return img_mosaic