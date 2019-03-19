#!/usr/bin/python

import sys
import numpy as np
import cv2

global FILE_NAME
global FILE_NAME_GRAY
FILE_NAME = './uploads/' + str(sys.argv[1])
FILE_NAME_GRAY = './uploads/' + str(sys.argv[2])

def getImage(fileName):
  global FILE_NAME
  image = cv2.imread(FILE_NAME)
  return image

def toGrayScale(array):
  return np.dot(array[...,:3], [0.3, 0.59, 0.11])

arrayOfPixels = getImage(FILE_NAME)
grayscale = toGrayScale(arrayOfPixels)
cv2.imwrite(FILE_NAME_GRAY, grayscale)
