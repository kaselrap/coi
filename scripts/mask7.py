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
    -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, 49, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1,
  ]
  kern = 1
elif str(sys.argv[3]) == 'low-frequency':
  mask = [
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
  ]
  kern = np.sum(mask)
elif str(sys.argv[3]) == 'gaussian-blur':
  mask = [
    1, 1, 1, 1, 1, 1, 1,
    1, 2, 2, 2, 2, 2, 1,
    1, 2, 1, 1, 1, 2, 1,
    1, 2, 1, 1, 1, 2, 1,
    1, 2, 1, 1, 1, 2, 1,
    1, 2, 2, 2, 2, 2, 1,
    1, 1, 1, 1, 1, 1, 1,
  ]
  kern = np.sum(mask)
elif str(sys.argv[3]) == 'embossing':
  mask = [
    1, 1,   1,  1,  1,  1,  0,
    1, 1,   1,  1,  1,  0, -1,
    1, 1,   1,  1,  0, -1, -1,
    1, 1,   1,  1, -1, -1, -1,
    1, 1,   0, -1, -1, -1, -1,
    1, 0,  -1, -1, -1, -1, -1,
    0, -1, -1, -1, -1, -1, -1,
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

for row in range(3, height-3):
  for col in range(3, width-3):
    xxx = np.array([
      [arrayOfPixels[row-3][col-3], arrayOfPixels[row-3][col-2], arrayOfPixels[row-3][col-1], arrayOfPixels[row-3][col], arrayOfPixels[row-3][col+1], arrayOfPixels[row-3][col+2], arrayOfPixels[row-3][col+3]],
      [arrayOfPixels[row-2][col-3], arrayOfPixels[row-2][col-2], arrayOfPixels[row-2][col-1], arrayOfPixels[row-2][col], arrayOfPixels[row-2][col+1], arrayOfPixels[row-2][col+2], arrayOfPixels[row-2][col+3]],
      [arrayOfPixels[row-1][col-3], arrayOfPixels[row-1][col-2], arrayOfPixels[row-1][col-1], arrayOfPixels[row-1][col], arrayOfPixels[row-1][col+1], arrayOfPixels[row-1][col+2], arrayOfPixels[row-1][col+3]],
      [arrayOfPixels[row][col-3], arrayOfPixels[row][col-2], arrayOfPixels[row][col-1], arrayOfPixels[row][col], arrayOfPixels[row][col+1], arrayOfPixels[row][col+2], arrayOfPixels[row][col+3]],
      [arrayOfPixels[row+1][col-3], arrayOfPixels[row+1][col-2], arrayOfPixels[row+1][col-1], arrayOfPixels[row+1][col], arrayOfPixels[row+1][col+1], arrayOfPixels[row+1][col+2], arrayOfPixels[row+1][col+3]],
      [arrayOfPixels[row+2][col-3], arrayOfPixels[row+2][col-2], arrayOfPixels[row+2][col-1], arrayOfPixels[row+2][col], arrayOfPixels[row+2][col+1], arrayOfPixels[row+2][col+2], arrayOfPixels[row+2][col+3]],
      [arrayOfPixels[row+3][col-3], arrayOfPixels[row+3][col-2], arrayOfPixels[row+3][col-1], arrayOfPixels[row+3][col], arrayOfPixels[row+3][col+1], arrayOfPixels[row+3][col+2], arrayOfPixels[row+3][col+3]],
    ])

    pixelSum = np.sum([
      xxx[0][0] * mask[0],
      xxx[0][1] * mask[1],
      xxx[0][2] * mask[2],
      xxx[0][3] * mask[3],
      xxx[0][4] * mask[4],
      xxx[0][5] * mask[5],
      xxx[0][6] * mask[6],

      xxx[1][0] * mask[7],
      xxx[1][1] * mask[8],
      xxx[1][2] * mask[9],
      xxx[1][3] * mask[10],
      xxx[1][4] * mask[11],
      xxx[1][5] * mask[12],
      xxx[1][6] * mask[13],

      xxx[2][0] * mask[14],
      xxx[2][1] * mask[15],
      xxx[2][2] * mask[16],
      xxx[2][3] * mask[17],
      xxx[2][4] * mask[18],
      xxx[2][5] * mask[19],
      xxx[2][6] * mask[20],

      xxx[3][0] * mask[21],
      xxx[3][1] * mask[22],
      xxx[3][2] * mask[23],
      xxx[3][3] * mask[24],
      xxx[3][4] * mask[25],
      xxx[3][5] * mask[26],
      xxx[3][6] * mask[27],

      xxx[4][0] * mask[28],
      xxx[4][1] * mask[29],
      xxx[4][2] * mask[30],
      xxx[4][3] * mask[31],
      xxx[4][4] * mask[32],
      xxx[4][5] * mask[33],
      xxx[4][6] * mask[34],

      xxx[5][0] * mask[35],
      xxx[5][1] * mask[36],
      xxx[5][2] * mask[37],
      xxx[5][3] * mask[38],
      xxx[5][4] * mask[39],
      xxx[5][5] * mask[40],
      xxx[5][6] * mask[41],

      xxx[6][0] * mask[42],
      xxx[6][1] * mask[43],
      xxx[6][2] * mask[44],
      xxx[6][3] * mask[45],
      xxx[6][4] * mask[46],
      xxx[6][5] * mask[47],
      xxx[6][6] * mask[48],
    ])

    lololo = (int(pixelSum) / kern) + add

    if lololo > 255:
      lol[row][col] = 255
    elif lololo < 0:
      lol[row][col] = 0
    else:
      lol[row][col] = lololo

cv2.imwrite(FILE_NAME_LOW_FREQUENCY, lol)
