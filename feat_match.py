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
def feat_match(descs1, descs2):
  import numpy as np
  from annoy import AnnoyIndex
  
  # Pull out dims
  h, n1 = descs1.shape
  h, n2 = descs2.shape

  # # Number of trees ANNOY will use
  # trees = 10

  # # Initialize our match array
  # match = -np.ones((n1, 1))

  # # Loop through match and run a NNS for every descriptor in descs1
  # for i in range(n1):

  #   # Set up the ANNOY object
  #   indexer = AnnoyIndex(h)
  #   indexer.add_item(0, descs1[:, i])

  #   # Add all the neighbors
  #   for j in range(n2):
  #     indexer.add_item(j + 1, descs2[:, j])

  #   # Build the tree
  #   indexer.build(trees)

  #   # Indices and distances are sorted by the distances
  #   ind, dist = indexer.get_nns_by_item(0, n2 + 1, include_distances=True)
  #   ratio = dist[1] / dist[2]

  #   # If it passes the 0.7 test, let it into matches
  #   if ratio < 0.7:
  #     match[i] = ind[1]

  # Brute force code to compare against tree search
  # Right now, they output very different values
  match = -np.ones((n1, 1))
  dist = np.zeros(n2)
  for i in range(n1):
    for j in range(n2):
      dist[j] = np.sum(np.square(descs1[:, i] - descs2[:, j]))
    best = np.amin(dist)
    dist[dist == best] = np.Inf
    second = np.amin(dist)
    if (best / second) < 0.7:
      match[i] = i

  return match