'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''

def corner_detector(img):
	import numpy as np
	from skimage.feature import corner_harris
	
	img = np.pad(img, ((20, 20), (20, 20)), mode="symmetric")
	cimg = corner_harris(img, k=0.02, sigma=1.4)
	return cimg[20:-20, 20:-20]