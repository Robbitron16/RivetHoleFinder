__author__ = 'Robby Mitchell'
# 10/28/15
# Boeing Advanced Research Center
#
# This class manages circle identification based on points on that circle from images.

import math
import numpy as np
import cv2 as cv
from Point import *

CANNY_ONE = 10
CANNY_TWO = 70

# Implements the float xrange for Python, so that you can step by floats.
def frange(start, stop, step):
    x = start
    while x < stop:
        yield x
        x += step

# This method finds circle contours given an image
# Pre:  Takes a Mat that represents a BGR image. The image cannot be null, and must be color.
# Post: Returns an array of contours that represent all of the circles in the image and the canny image itself.
def getCircleContours(img):
    circleContours = []
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, CANNY_ONE, CANNY_TWO, 2)
    cv.imshow("Canny", canny)
    canny, contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv.approxPolyDP(contour, cv.arcLength(contour, False) * 0.1, False)
        if cv.contourArea(contour) < 20:
            continue
        elif approx.size > 6:
            area = cv.contourArea(contour)
            x, y, w, h = cv.boundingRect(contour)
            radius = w / 2
            if (math.fabs(1 - float(w / h)) <= 1.5 and math.fabs(1 - float(area / (math.pi * math.pow(radius, 2)))) <= 1.5):
                circleContours.append(contour)
    return (circleContours, canny)


class CircleManager(object):
    # Class Constants
    ## Save the value of pi.
    PI = math.pi
    ## Error margin in pixels for trimming points
    ERROR_MARGIN = 10
    ## Circle validation constants
    MIN_ID = 2.0
    MAX_IDM = 100.0
    AVG_ITERATIONS = 100


    # Constructor
    # Pre:  Takes the contour of the circle. The contour should be the contour that defines the circle, given from
    #       the circle-contour identification algorithm.
    # Post: Generates a new CircleManager object.
    def __init__(self, contour):
        self.center = None
        self.radius = -1
        self.contour = contour

    def getContour(self):
        return self.contour

    # This method calculates the center and radius of the circle based on three points on that circle.
    # Pre:  Takes three points on the circle, represented as a coordinate array (ie. (0, 1)). The three points
    #       must be in this format.
    # Post: Returns the calculated
    def getCircle(self, p1, p2, p3):

        x1 = float(p1[0])
        x2 = float(p2[0])
        x3 = float(p3[0])

        y1 = float(p1[1])
        y2 = float(p2[1])
        y3 = float(p3[1])

        centerEstimate = Point()
        divisor = (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2))
        if divisor == 0:
            return (None, -1)
        # Calculate center of circle using formula.
        centerEstimate.x = (x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)

        centerEstimate.x = centerEstimate.x / divisor

        centerEstimate.y = (x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)
        centerEstimate.y = centerEstimate.y / divisor

        radiusEstimate = math.sqrt((centerEstimate.x - x1)**2 + (centerEstimate.y - y1)**2)

        return (centerEstimate, radiusEstimate)

    # This method calculates the percentage fit of a circle given it's radius and center on the image it was derived
    # from.
    # Pre:  Takes the transformed image, the center of the circle, the radius of the circle and an empty set that will
    #       contain the points that match the circle on the image.
    # Post: Returns the percentage of points that matched the image over the amount of the points defined by the circle
    #       as a float (0.0 - 1.0).
    def verifyCircle(self, dt, centerEst, radiusEst, inlierSet):
        counter = 0.0
        inlier = 0.0
        rows = len(dt)
        cols = len(dt[0])
        maxInlierDist = radiusEst / 25.0

        if (maxInlierDist < self.MIN_ID):
            maxInlierDist = self.MIN_ID
        if (maxInlierDist > self.MAX_IDM):
            maxInlierDist = self.MAX_IDM

        for t in frange(0, self.PI * 2, 0.05):
            counter = counter + 1
            cX = radiusEst * math.cos(t) + centerEst.x
            cY = radiusEst * math.sin(t) + centerEst.y
            if (cX < cols and cX >= 0 and cY < rows and cY >= 0 and dt[cY][cX] < maxInlierDist):
                inlier += 1
                inlierSet = np.append(inlierSet, [[cX, cY]])

        return inlier / counter

