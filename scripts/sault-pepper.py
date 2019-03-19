#!/usr/bin/python

import sys
import numpy as np
import cv2
import random

global FILE_NAME
global FILE_NAME_SAULT
FILE_NAME = './uploads/' + str(sys.argv[1])
FILE_NAME_SAULT = './uploads/' + str(sys.argv[2])

def getImage(fileName):
  global FILE_NAME
  image = cv2.imread(FILE_NAME)
  return image

arrayOfPixels = getImage(FILE_NAME)

height, width, _ = arrayOfPixels.shape

for row in range(height):
  for col in range(width):
    if random.randint(0, 150) > 149:
      arrayOfPixels[row][col] = 255
    elif random.randint(0, 150) < 1:
      arrayOfPixels[row][col] = 0

cv2.imwrite(FILE_NAME_SAULT, arrayOfPixels)
