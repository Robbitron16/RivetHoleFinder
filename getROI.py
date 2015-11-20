__author__ = 'Robby Mitchell'
# Imports
import cv2 as cv
import numpy as np
# Global threshold constants.
# The error threshold defines how much the ROI should compensate for HoughCircles() error. Measured in pixels.
ERROR_THRESHOLD = 20

# Point Class
# This object stores the x and y coordinates of a point on a coordinate system.
class Point(object):
    def __init__(self, x = 0.0, y = 0.0):
        self.x = x
        self.y = y

# This method gets the region of interest that the circles are located inside.
# Takes the String filename to read the image.
# Returns the region of interest as a Mat.
def getROIMask(filename):
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1 ,20, param1=50, param2=30, maxRadius=0)

    circles = np.uint16(np.around(circles))

    rois = []
    for circle in circles[0]:
        # Debug image only has one circle. Expand to all circles in future.
        center = Point(circle[0], circle[1])
        radius = circle[2]
        # Define roi rectangle Mat.
        x = center.x - radius - ERROR_THRESHOLD
        y = center.y - radius - ERROR_THRESHOLD

        if (x < 0):
            x = 0
        if (y < 0):
            y = 0
        width = 2 * (radius + ERROR_THRESHOLD)
        height = 2 * (radius + ERROR_THRESHOLD)

        roi = img[y : y + height, x : x + width]
        rois.append(roi)
    return rois