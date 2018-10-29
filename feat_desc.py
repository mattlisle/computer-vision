'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''

def feat_desc(img, x, y):
  import numpy as np
  from helpers import interp2
  from scipy.ndimage import filters
  from scipy import signal
  
  # Initialize output array
  descs = np.zeros((64, len(x)))

  # Default meshgrid
  defx, defy = np.meshgrid(np.arange(-17.5, 22.5, 5), np.arange(-17.5, 22.5, 5))

  # Pad the img array so we can include the edge pixels in the 36 x 36 window
  padded = np.pad(img, ((18, 18), (18, 18)), mode="symmetric")  # may want to change the mode here if results are bad

  # Account for the padding in x and y of interest points
  x = x + 18
  y = y + 18

  # Create gaussian filtered image to sample from of size 5 x 5
  # filtered = filters.gaussian_filter(padded, sigma=1, mode="same", truncate=2)
  # G = signal.gaussian(25, 1).reshape(5, 5)
  # filtered = signal.convolve2d(padded, G, mode="same")
  filtered = padded
  
  # Get orientations of interest points for to rotate sampling window
  dx_img, dy_img = np.gradient(filtered, axis=(1, 0))
  ori = np.arctan2(dy_img, dx_img)
  ori[ori > np.pi] = ori[ori > np.pi] - np.pi
  ori[ori < 0] = ori[ori < 0] + np.pi

  # For plotting boxes and orientations around interest points
  boxes = np.zeros((5, 2, len(x)))
  oris = np.zeros((2, 2, len(x)))
  xcorners = np.array([0, -1, -1, 0, 0])
  ycorners = np.array([0, 0, -1, -1, 0])

  # Loop over all descriptors in descs
  for j in range(descs.shape[1]):

    # Create a rotation matrix by which we will rotate the meshgrid
    theta = -ori[y[j], x[j]]
    # if j == 0:
    #   print(theta, ori[y[j], x[j]], y[j], x[j])
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    rotx, roty = np.einsum('ji, mni -> jmn', -R, np.dstack([defx, defy]))

    # Generate the 64 pixel locations in the form of a meshgrid
    thisx = rotx + x[j]
    thisy = roty + y[j]

    # For plotting boxes and orientations around interest points
    boxes[:, :, j] = np.dstack([thisx[xcorners, ycorners], thisy[xcorners, ycorners]]) - 18
    oris[:, :, j] = np.array([[x[j], x[j] + 50 * np.cos(-theta)], [y[j], y[j] + 50 * np.sin(-theta)]]) - 18
    # if j == 0:
    #   print(theta, oris[:,:,j])

    # Get the pixel values at those locations
    values = interp2(img, thisx, thisy)

    # Normalize the values to mu = 0, sigma = 1
    values = (values - np.mean(values)) / np.std(values)

    # Save the data into descs
    descs[:, j] = values.reshape(64)

  return descs, boxes, oris, ori