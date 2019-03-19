#!/usr/bin/python

import sys
import numpy as np
import cv2

global FILE_NAME
global FILE_NAME_BRIGHT_UP
FILE_NAME = './uploads/' + str(sys.argv[1])
FILE_NAME_BRIGHT_UP = './uploads/' + str(sys.argv[2])
COUNT = float(sys.argv[3])

def getImage(fileName):
  global FILE_NAME
  image = cv2.imread(FILE_NAME)
  return image

arrayOfPixels = getImage(FILE_NAME)

cv2.imwrite(FILE_NAME_BRIGHT_UP, arrayOfPixels * COUNT)

