#!/usr/bin/python

import sys
import numpy as np
import cv2
import random

global FILE_NAME
global FILE_NAME_LOW_FREQUENCY
FILE_NAME = './uploads/' + str(sys.argv[1])
FILE_NAME_LOW_FREQUENCY = './uploads/' + str(sys.argv[2])

global mask
global add
add = 0
if str(sys.argv[3]) == 'high-frequency':
  mask = [-1, -1, -1, -1, 8, -1, -1, -1, -1]
  kern = 1
elif str(sys.argv[3]) == 'low-frequency':
  mask = [3, 3, 3, 3, 3, 3, 3, 3, 3]
  kern = np.sum(mask)
elif str(sys.argv[3]) == 'gaussian-blur':
  mask = [1, 2, 1, 2, 4, 2, 1, 2, 1]
  kern = np.sum(mask)
elif str(sys.argv[3]) == 'embossing':
  mask = [0, -1, 0, -1, 4, -1, 0, -1, 0]
  kern = 1
  add = 128
elif str(sys.argv[3]) == 'vertical-linear':
  mask = [-3, -3, 5, -3, 0, 5, -3, -3, 5]
  kern = 1
elif str(sys.argv[3]) == 'horizontal-linear':
  mask = [1, 2, 1, 0, 0, 0, -1, -2, -1]
  kern = 1
elif str(sys.argv[3]) == 'diagonal':
  mask = [-3, -3, -3, 5, 0, -3, 5, 5, -3]
  kern = 1
elif str(sys.argv[3]) == 'laplasa':
  mask = [1, 1, 1, 1, -8, 1, 1, 1, 1]
  kern = 1
else:
  mask = [1, 1, 1, 1, 1, 1, 1, 1, 1]
  kern = np.sum(mask) / 2

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
    xxx = np.array([
      [arrayOfPixels[row-1][col-1], arrayOfPixels[row-1][col], arrayOfPixels[row-1][col+1]],
      [arrayOfPixels[row][col-1], arrayOfPixels[row][col], arrayOfPixels[row][col+1]],
      [arrayOfPixels[row+1][col-1], arrayOfPixels[row+1][col], arrayOfPixels[row+1][col+1]]
    ])

    pixelSum = np.sum([
      xxx[0][0] * mask[0],
      xxx[0][1] * mask[1],
      xxx[0][2] * mask[2],
      xxx[1][0] * mask[3],
      xxx[1][1] * mask[4],
      xxx[1][2] * mask[5],
      xxx[2][0] * mask[6],
      xxx[2][1] * mask[7],
      xxx[2][2] * mask[8]
    ])

    lololo = (int(pixelSum) / kern) + add

    if lololo > 255:
      lol[row][col] = 255
    elif lololo < 0:
      lol[row][col] = 0
    else:
      lol[row][col] = lololo

cv2.imwrite(FILE_NAME_LOW_FREQUENCY, lol)
