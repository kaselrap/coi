#!/usr/bin/python

import sys
import numpy as np
import cv2
import random
import math

global FILE_NAME
global FILE_NAME_LOW_PREVITTA
FILE_NAME = './uploads/' + str(sys.argv[1])
FILE_NAME_LOW_PREVITTA = './uploads/' + str(sys.argv[2])

def getImage(fileName):
  global FILE_NAME
  image = cv2.imread(FILE_NAME)
  return image

def toGrayScale(array):
  return np.dot(array[...,:3], [0.3, 0.59, 0.11])

arrayOfPixels = getImage(FILE_NAME)
arrayOfPixels = toGrayScale(arrayOfPixels)
height, width = arrayOfPixels.shape
lol = np.zeros(arrayOfPixels.shape)

for row in range(1, height-1):
  for col in range(1, width-1):
    x = np.array([
      [arrayOfPixels[row-1][col-1], arrayOfPixels[row-1][col], arrayOfPixels[row-1][col+1]],
      [arrayOfPixels[row][col-1], arrayOfPixels[row][col], arrayOfPixels[row][col+1]],
      [arrayOfPixels[row+1][col-1], arrayOfPixels[row+1][col], arrayOfPixels[row+1][col+1]]
    ])

    Gx = (x[2][2] - x[1][1])
    Gy = (x[2][1] - x[1][2])

    lololo = math.sqrt((Gx ** 2) + (Gy ** 2))

    if lololo > 255:
      lol[row][col] = 255
    elif lololo < 0:
      lol[row][col] = 0
    else:
      lol[row][col] = lololo

cv2.imwrite(FILE_NAME_LOW_PREVITTA, lol)
