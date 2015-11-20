__author__ = 'Robby Mitchell'

import cv2 as cv
import random as rand
import numpy as np

import CircleManager as cm
from Point import *
import getROI as gr

# Get circle edges from the image.
rois = gr.getROIMask('circle_detected.jpg')
#rois = gr.getROIMask('ALFtp.jpg')
dts = []
circleOutputs = []
finalPics = []
for roi in rois:
    circle, canny = cm.getCircleContours(roi)
    if len(circle) > 0:
        circleOutputs.append(cm.CircleManager(circle[0]))
        dts.append(cv.distanceTransform(canny, 1, 3))
        finalPics.append(roi)
MIN_RADIUS = 5.0

for circle in circleOutputs:
    # Loop through the circle contour in each manager AVG_ITERATIONS times.
    for i in range(0, cm.CircleManager.AVG_ITERATIONS, 1):
        bestCirclePercentage = 0.0
        curr = cm.CircleManager(circle.contour)
        # Find the best circle radius and center.
        while bestCirclePercentage <= 0.8:
            idx1 = rand.randint(0, len(curr.contour) - 1)
            idx2 = rand.randint(0, len(curr.contour) - 1)
            idx3 = rand.randint(0, len(curr.contour) - 1)

            if idx1 == idx2 or idx2 == idx3 or idx1 == idx3:
                continue

            p1 = (curr.contour[idx1])[0]
            p2 = (curr.contour[idx2])[0]
            p3 = (curr.contour[idx3])[0]

            center, radius = curr.getCircle(p1, p2, p3)
            inlierSet = np.empty([0, 2])
            cPerc = -1.0
            if (center == None) and (radius == -1):
                continue
            else:
                cPerc = curr.verifyCircle(dts[circleOutputs.index(circle)], center, radius, inlierSet)

            if cPerc > bestCirclePercentage:
                if radius > MIN_RADIUS:
                    bestCirclePercentage = cPerc
                    curr.radius = int(radius)
                    curr.center = Point(int(center.x), int(center.y))
        # Average the new radius and center against the last center and radius.
        # Creates a rolling average.
        if not(curr.radius == -1) and not(curr.center == None):
            actual = circleOutputs[circleOutputs.index(circle)]
            if (actual.center == None):
                actual.center = curr.center
                actual.radius = curr.radius
            else:
                actual.center.x = int(curr.center.x + actual.center.x)
                actual.center.y = int(curr.center.y + actual.center.y)
                actual.radius = int(curr.radius + actual.radius)

for curr in circleOutputs:
    if curr.center != None and curr.radius != -1:
        curr.center.x = int(curr.center.x / cm.CircleManager.AVG_ITERATIONS)
        curr.center.y = int(curr.center.y / cm.CircleManager.AVG_ITERATIONS)
        curr.radius = int(curr.radius / cm.CircleManager.AVG_ITERATIONS)
        cv.circle(finalPics[circleOutputs.index(curr)], (curr.center.x, curr.center.y),curr.radius, (0, 0, 255), 2)
        cv.circle(finalPics[circleOutputs.index(curr)], (curr.center.x, curr.center.y), 3, (0, 255, 0), 3)
        print "Final Radius: ", curr.radius
        print "Final Center: (", curr.center.x, ", ", curr.center.y, ")"
    cv.imshow('Output', finalPics[circleOutputs.index(curr)])
    cv.waitKey(0)
cv.destroyAllWindows()