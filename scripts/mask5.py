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
  mask = [
    0, -1, -1, -1, 0,
    -1, 0, -1, 0, -1,
    -1, -1, 17, -1, -1,
    -1, 0, -1, 0, -1,
    0, -1, -1, -1, 0,
  ]
  kern = 1
elif str(sys.argv[3]) == 'low-frequency':
  mask = [
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
  ]
  kern = np.sum(mask)
elif str(sys.argv[3]) == 'gaussian-blur':
  mask = [
    1, 2, 1, 1, 2,
    1, 2, 1, 1, 2,
    1, 2, 1, 1, 2,
    1, 2, 1, 1, 2,
    1, 2, 1, 1, 2,
  ]
  kern = np.sum(mask)
elif str(sys.argv[3]) == 'embossing':
  mask = [
    1,  1,  1,  1,  0,
    1,  1,  1,  0, -1,
    1,  1,  1, -1, -1,
    1,  0, -1, -1, -1,
    0, -1, -1, -1, -1
  ]
  kern = 1
  add = 128
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

for row in range(2, height-2):
  for col in range(2, width-2):
    xxx = np.array([
      [arrayOfPixels[row-2][col-2], arrayOfPixels[row-2][col-1], arrayOfPixels[row-2][col], arrayOfPixels[row-2][col+1], arrayOfPixels[row-2][col+2]],
      [arrayOfPixels[row-1][col-2], arrayOfPixels[row-1][col-1], arrayOfPixels[row-1][col], arrayOfPixels[row-1][col+1], arrayOfPixels[row-1][col+2]],
      [arrayOfPixels[row][col-2], arrayOfPixels[row][col-1], arrayOfPixels[row][col], arrayOfPixels[row][col+1], arrayOfPixels[row][col+2]],
      [arrayOfPixels[row+1][col-2], arrayOfPixels[row+1][col-1], arrayOfPixels[row+1][col], arrayOfPixels[row+1][col+1], arrayOfPixels[row+1][col+2]],
      [arrayOfPixels[row+2][col-2], arrayOfPixels[row+2][col-1], arrayOfPixels[row+2][col], arrayOfPixels[row+2][col+1], arrayOfPixels[row+2][col+2]],
    ])

    pixelSum = np.sum([
      xxx[0][0] * mask[0],
      xxx[0][1] * mask[1],
      xxx[0][2] * mask[2],
      xxx[0][3] * mask[3],
      xxx[0][4] * mask[4],
      xxx[1][0] * mask[5],
      xxx[1][1] * mask[6],
      xxx[1][2] * mask[7],
      xxx[1][3] * mask[8],
      xxx[1][4] * mask[9],
      xxx[2][0] * mask[10],
      xxx[2][1] * mask[11],
      xxx[2][2] * mask[12],
      xxx[2][3] * mask[13],
      xxx[2][4] * mask[14],
      xxx[3][0] * mask[15],
      xxx[3][1] * mask[16],
      xxx[3][2] * mask[17],
      xxx[3][3] * mask[18],
      xxx[3][4] * mask[19],
      xxx[4][0] * mask[20],
      xxx[4][1] * mask[21],
      xxx[4][2] * mask[22],
      xxx[4][3] * mask[23],
      xxx[4][4] * mask[24],
    ])

    lololo = (int(pixelSum) / kern) + add

    if lololo > 255:
      lol[row][col] = 255
    elif lololo < 0:
      lol[row][col] = 0
    else:
      lol[row][col] = lololo

cv2.imwrite(FILE_NAME_LOW_FREQUENCY, lol)
