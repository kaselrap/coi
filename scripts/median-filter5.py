#!/usr/bin/python

import sys
import numpy as np
import cv2
import random

global FILE_NAME
global FILE_NAME_MEDIAN
FILE_NAME = './uploads/' + str(sys.argv[1])
FILE_NAME_MEDIAN = './uploads/' + str(sys.argv[2])

def getImage(fileName):
  global FILE_NAME
  image = cv2.imread(FILE_NAME)
  return image

def toGrayScale(array):
  return np.dot(array[...,:3], [0.3, 0.59, 0.11])

arrayOfPixels = getImage(FILE_NAME)
height, width, qwe = arrayOfPixels.shape
lol = np.zeros((height, width, qwe))

for row in range(2, height-2):
  for col in range(2, width-2):
    xxx = np.array([
      [arrayOfPixels[row-2][col-2], arrayOfPixels[row-2][col-1], arrayOfPixels[row-2][col], arrayOfPixels[row-2][col+1], arrayOfPixels[row-2][col+2]],
      [arrayOfPixels[row-1][col-2], arrayOfPixels[row-1][col-1], arrayOfPixels[row-1][col], arrayOfPixels[row-1][col+1], arrayOfPixels[row-1][col+2]],
      [arrayOfPixels[row][col-2], arrayOfPixels[row][col-1], arrayOfPixels[row][col], arrayOfPixels[row][col+1], arrayOfPixels[row][col+2]],
      [arrayOfPixels[row+1][col-2], arrayOfPixels[row+1][col-1], arrayOfPixels[row+1][col], arrayOfPixels[row+1][col+1], arrayOfPixels[row+1][col+2]],
      [arrayOfPixels[row+2][col-2], arrayOfPixels[row+2][col-1], arrayOfPixels[row+2][col], arrayOfPixels[row+2][col+1], arrayOfPixels[row+2][col+2]],
    ])
    sortedXxx = np.sort(xxx, axis=None)
    lol[row][col] = sortedXxx[int(len(sortedXxx)/2)]

cv2.imwrite(FILE_NAME_MEDIAN, lol)
