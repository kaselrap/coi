#!/usr/bin/python

import sys
import numpy as np
import cv2
import random

global FILE_NAME
global FILE_NAME_GAUSION
FILE_NAME = './uploads/' + str(sys.argv[1])
FILE_NAME_GAUSION = './uploads/' + str(sys.argv[2])

def getImage(fileName):
  global FILE_NAME
  image = cv2.imread(FILE_NAME)
  return image

arrayOfPixels = getImage(FILE_NAME)

height, width, xxx = arrayOfPixels.shape

gaussian = np.random.randn(height, width, xxx) * 45
arrayOfPixels = arrayOfPixels + gaussian

cv2.imwrite(FILE_NAME_GAUSION, arrayOfPixels)
