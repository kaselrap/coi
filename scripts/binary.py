#!/usr/bin/python

import sys
import numpy as np
import cv2
import math

global FILE_NAME
global FILE_NAME_BINARY
FILE_NAME = './uploads/' + str(sys.argv[1])
FILE_NAME_BINARY = './uploads/' + str(sys.argv[2])

def getHistogram(array):
  height, width = array.shape
  countOfPixels = height * width
  histogram = np.zeros((height, width))

  countOfRepeat = np.zeros(256)

  for row in range(height):
    for col in range(width):
      countOfRepeat[int(array[row][col])] += 1

  return countOfRepeat

def getImage(fileName):
  global FILE_NAME
  image = cv2.imread(FILE_NAME)
  return image

def toGrayScale(array):
  return np.dot(array[...,:3], [0.3, 0.59, 0.11])

arrayOfPixels = getImage(FILE_NAME)
height, width, _ = arrayOfPixels.shape
countOfPixels = height * width
grayscale = toGrayScale(arrayOfPixels)
histogram = getHistogram(grayscale)
histogram = histogram + 1
p = np.zeros(256)
maxSigma=-1000000
histogramLength = histogram.shape

for index in range(1, 255):
  p[index] = histogram[index] / (height * width)

S2=0
for t in range(3, len(p)-1):
  S2 = S2 + t * p[t]

w1 = p[1]
S1 = p[1]
w2 = 0

optimal = 0

for t in range(2, len(p)-1):
  w1 = w1 + p[t]
  w2 = 1 - w1
  S1 = S1 + (t * p[t])
  sigma = ((S1 * w2) - (S2 * w1))/(w1*w2)

  if maxSigma < sigma:
    maxSigma = sigma
    optimal = t

x = np.zeros((height, width))

for row in range(height):
  for col in range(width):
    if grayscale[row][col] > optimal:
      x[row][col] = 254
    else:
      x[row][col] = 0

cv2.imwrite(FILE_NAME_BINARY, x)
