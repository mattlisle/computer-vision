'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''

def ransac_est_homography(x1, y1, x2, y2, thresh):
  import numpy as np
  from est_homography import est_homography
  
  # Number of RANSAC trials
  t = 1000
  c = 0

  # Length of matching points arrays
  n = len(x1)
  z = np.ones(n)

  inlier_ind = np.array([])
  while len(inlier_ind) <= 10 and c < 20: 
    c += 1

    for i in range(t):

      # Choose 4 points randomly and estimate a homography
      choices = np.random.choice(n, 4)
      H = est_homography(x1[choices], y1[choices], x2[choices], y2[choices])

      # Use the homography to transform points from img1 to estimated postions in img2
      estimates = np.matmul(H, np.stack([x1, y1, z]))
      
      # Normalize the estimates and extract x and y
      estimates = estimates / estimates[-1]
      x_est = estimates[0]
      y_est = estimates[1]

      # Compute sum of squared distances (That's what the assignment said, but I don't see why we would sum them?)
      distances = np.sqrt(np.square(x2 - x_est) + np.square(y2 - y_est))
      ind = np.where(distances < thresh)[0]

      # If we did better this time, save the indices
      if len(ind) > len(inlier_ind):
        found = len(ind)
        print("Found %d matches..." % found, end="\r", flush=True)
        inlier_ind = ind
        dist = distances
        Hout = H

  # Run least squares on the inliers to get the most reliable estimate of H

  print("Found %d matches..." % found)
  return Hout, inlier_ind