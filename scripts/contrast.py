#!/usr/bin/python

import sys
import numpy as np
import cv2

global FILE_NAME
global FILE_NAME_CONTRAST
FILE_NAME = './uploads/' + str(sys.argv[1])
FILE_NAME_CONTRAST = './uploads/' + str(sys.argv[2])

def getImage(fileName):
  global FILE_NAME
  image = cv2.imread(FILE_NAME)
  return image

def toGrayScale(array):
  return np.dot(array[...,:3], [0.3, 0.59, 0.11])

def getHistogram(array):
  height, width = array.shape
  countOfPixels = height * width
  histogram = np.zeros((height, width))

  countOfRepeat = np.zeros(256)

  for row in range(height):
    for col in range(width):
      if countOfRepeat[int(array[row][col])] < 255:
        countOfRepeat[int(array[row][col])] += 1

  return countOfRepeat

arrayOfPixels = getImage(FILE_NAME)
grayscale = toGrayScale(arrayOfPixels)
histogram = getHistogram(grayscale)
height, width, depth = arrayOfPixels.shape

minH = np.amin(grayscale)
maxH = np.amax(grayscale)

lol = np.zeros((height, width, depth))

for row in range(height):
  for col in range(width):
    lol[row][col] = ((arrayOfPixels[row][col] - minH) / (maxH - minH) * 255)

cv2.imwrite(FILE_NAME_CONTRAST, lol)
