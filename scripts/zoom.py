#!/usr/bin/python

import sys
import numpy as np
import cv2

global FILE_NAME
global FILE_NAME_ZOOM
FILE_NAME = './uploads/' + str(sys.argv[1])
FILE_NAME_ZOOM = './uploads/' + str(sys.argv[2])
COUNT = int(sys.argv[3])

def getImage(fileName):
  global FILE_NAME
  image = cv2.imread(FILE_NAME)
  return image

arrayOfPixels = getImage(FILE_NAME)
height, width, level = arrayOfPixels.shape

global DIRECTION
if COUNT > 0:
  DIRECTION = 'up'
else:
  DIRECTION = 'down'

zoomHowMuch = abs(COUNT)
if zoomHowMuch == 1:
  cv2.imwrite(FILE_NAME_ZOOM, arrayOfPixels)
  exit(0)

newHeight = int(height / zoomHowMuch)
newWidth = int(width / zoomHowMuch)
zoomInImage = np.zeros((height, width, level))

x = 0
y = 0

for row in range(height):
  x = 0
  for col in range(width):
    for lol in range(zoomHowMuch):
      zoomInImage[y + lol - 1][x + lol - 1] = arrayOfPixels[row][col]
      zoomInImage[y + lol - 1][x+lol] = arrayOfPixels[row][col]
      zoomInImage[y+lol][x + lol - 1] = arrayOfPixels[row][col]
      zoomInImage[y+lol][x+lol] = arrayOfPixels[row][col]
    if (x + zoomHowMuch > width - zoomHowMuch):
      break
    x += zoomHowMuch

  if (y + zoomHowMuch > height - zoomHowMuch):
    break
  y += zoomHowMuch


zoomOutImage = np.zeros((height, width, level))

x = 0
y = 0

for row in range(height):
  x = 0
  for col in range(width):
    for lol in range(zoomHowMuch):
      zoomOutImage[int(newHeight/2) + row][int(newWidth/2) + col] = arrayOfPixels[y][x]
    if (x + zoomHowMuch > width - zoomHowMuch):
      break
    x += zoomHowMuch

  if (y + zoomHowMuch > height - zoomHowMuch):
    break
  y += zoomHowMuch

if DIRECTION == 'up':
  cv2.imwrite(FILE_NAME_ZOOM, zoomInImage)
else:
  cv2.imwrite(FILE_NAME_ZOOM, zoomOutImage)

