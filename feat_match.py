'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''

# Documentation on annoy: https://pypi.org/project/annoy/
# Something to consider when debugging RANSAC: the ratios don't have to be the same both ways
def feat_match(descs1, descs2):
  import numpy as np
  from annoy import AnnoyIndex
  import time
  start = time.time()
  
  # Pull out dims
  h, n1 = descs1.shape
  h, n2 = descs2.shape

  # Number of trees ANNOY will use
  trees = 5

  # Initialize our match array
  match = -np.ones((n1, 1))
  found = 0

  # Loop through match and run a NNS for every descriptor in descs1
  for i in range(n1):

    # Set up the ANNOY object
    forward = AnnoyIndex(h)
    forward.add_item(0, descs1[:, i])

    # Add all the neighbors
    for j in range(n2):
      forward.add_item(j + 1, descs2[:, j])

    # Build the tree
    forward.build(trees)

    # Indices and distances are sorted by the distances
    ind, dist = forward.get_nns_by_item(0, n2 + 1, include_distances=True)
    ratio = dist[1] / dist[2]

    # If it passes the ratio test, we need to check going backwards as well
    if ratio < 0.8:
      k = ind[1] - 1

      # Construct the backward tree for the candidate point in descs2
      backward = AnnoyIndex(h)
      backward.add_item(0, descs2[:, k])

      # Add all the elements in descs1
      for j in range(n1):
        backward.add_item(j + 1, descs1[:, j])

      # Run nearest neighbors
      backward.build(trees)
      ind, dist = backward.get_nns_by_item(0, n1 + 1, include_distances=True)
      ratio = dist[1] / dist[2]

      # If it passes both tests, we're good to go
      if (ind[1] - 1 == i) and (ratio < 0.8): 
        found += 1
        print("Found %d matches..." % found, end="\r", flush=True)
        match[i] = k

  # Debugging statments
  end = time.time()
  elasped = int(end - start)
  print("Found %d matches..." % found)
  print("Time to find matches: %d seconds" % elasped)

  return match.astype(int)