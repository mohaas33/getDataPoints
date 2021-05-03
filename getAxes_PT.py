#!/usr/bin/env python3

import sys
import math

import cv2
import numpy as np
from collections import defaultdict

ERRORVAL = np.nan

def grey(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
def canny(image):
    edges = cv2.Canny(image,50,150)
    return edges


def segment_by_angle_kmeans(lines):
    """Groups lines based on angle 
    """
    l_theta = [0 for i in range(len(lines))]
    i = 0 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dy = (y2-y1)
        dx = (x2-x1)
        theta = math.atan(dy/dx)
        l_theta[i] = theta
        i+=1
    #segmented = list(segmented.values())
    #return segmented
    #print(l_theta)
    return l_theta

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def line_intersect(line1,line2):
    """ returns a (x, y) tuple or None if there is no intersection """

    Ax1, Ay1, Ax2, Ay2 = line1
    Bx1, By1, Bx2, By2 = line2

    # Line AB represented as a1x + b1y = c1 
    a1 = Ay2 - Ay1; 
    b1 = Ax1 - Ax2; 
    c1 = a1*(Ax1) + b1*(Ay1); 
  
    # Line CD represented as a2x + b2y = c2 
    a2 = By2 - By1
    b2 = Bx1 - Bx2
    c2 = a2*(Bx1)+ b2*(By1)
  
    determinant = a1*b2 - a2*b1

    if determinant == 0:
        # The lines are parallel. This is simplified 
        # by returning a pair of None
        return [[None,None]]
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return [[x, y]] 


    
def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    intLines = []
    for i, group in enumerate(lines[:-1]):
        #print(i, group)
        for j, next_group in enumerate(lines[i + 1:],i+1):
            #print(next_group)
            for line1 in group:
                for line2 in next_group:
                    intersect = line_intersect(line1, line2)
                    if intersect[0][0] is not None:
                        intersections.append(intersect) 
                        intLines.append([i,j])
    #for i in range(len(lines)):
    #    for j in range(len(lines)):
    #        if i!=j:
    #            intersect = line_intersect(lines[i], lines[j])
    #            if intersect[0][0] is not None:
    #                intersections.append(intersect) 
    #                intLines.append([i,j])

    return intersections, intLines

def mark_intersections(
    img, intersections, color=None, markerType=cv2.MARKER_TRIANGLE_UP, markerSize=5
):
    color = color or (0, 255, 255) if len(img.shape) == 3 else 255
    for point in intersections:
        for x, y in point:
            if ERRORVAL in [x, y]:
                print("Point {} has np.nan vals; not drawing.".format(point))
                continue
            cv2.drawMarker(img, (int(x), int(y)), color, markerType, markerSize)

    return img

def find_lower_left_point(intersections):
    """Finds lower left point"""
    x_ll = 1e10
    y_ll = 1e10
    for intersection in intersections:
        x, y = intersection[0]
        if x<=x_ll and y<=y_ll:
            x_ll = x 
            y_ll = y

    return [x_ll, y_ll]
def assign_axes(lines, intLines, intersections, startingPoint):
    """Assignes X and Y axes"""
    X = []
    Y = []
    for (intLine, intersection) in zip(intLines, intersections):
        if startingPoint == intersection[0]:
            x1, y1, x2, y2 = lines[intLine[0]][0]
            if x1 == x2:
                Y = lines[intLine[0]]
                X = lines[intLine[1]]
            if y1 == y2:
                X = lines[intLine[0]]
                Y = lines[intLine[1]]

    return X, Y
#img = cv2.imread('./Figures/R_Current_1.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#blur = cv2.medianBlur(gray, 5)
#adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
#thresh_type = cv2.THRESH_BINARY_INV
#bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)


# Read image 
#figureName = '../sPHENIX/LFT/currents_GEM3_NoGlue_zoom_16.png'
figureName = './Figures/R_Current_1.png'
img = cv2.imread(figureName, cv2.IMREAD_COLOR) # road.png is the filename
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the edges in the image using canny detector
v = np.median(gray)
sigma = 0.33
#---- Apply automatic Canny edge detection using the computed median----
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edges = cv2.Canny(gray, lower, upper,3)
cv2.imwrite('./edges.jpg',edges)

# Detect points that form a line
max_slider, minLineLength, maxLineGap    = 150, 2, 250
lines = cv2.HoughLinesP(edges, 10, np.pi/10, max_slider, minLineLength, maxLineGap)
# Find all groups of lines
segmented = segment_by_angle_kmeans(lines)
# Find intersections and associate lines
intersections, intLines = segmented_intersections(lines)
#print(intLines)

# Choose lower left corner and associated lines
startingPoint = find_lower_left_point(intersections)
# Assign axes
X, Y = assign_axes(lines, intLines, intersections, startingPoint)
print(X)
print(Y)
Xx1, Xy1, Xx2, Xy2 = X[0]  
Yx1, Yy1, Yx2, Yy2 = X[0]  
# Crop image for X & Y
# Sets the Region of Interest 
# Note that the rectangle area has to be __INSIDE__ the image 
# You just iterate througt x and y.
pixels_to_cut = int((Xx1 - Xx2)/5)
img = img[ Xy1+pixels_to_cut: Xy2-pixels_to_cut,Xx1:Xx2]
# Create destination image 
# Note that cvGetSize will return the width and the height of ROI 
img_X = cv2.CreateImage(cv2.GetSize(img), 
                                   img.depth, 
                                   img.nChannels)

# copy subimage
cv2.cvCopy(img, img_X, NULL)

# always reset the Region of Interest */
cv2.cvResetImageROI(img)
cv2.imwrite('./axis_X.jpg',img)

# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

#Draw intersections
mark_intersections(img,intersections, (255, 153, 255), markerType=cv2.MARKER_TILTED_CROSS , markerSize=15)

# Show result
#cv2.imshow("Result Image", img)
cv2.imwrite('./result.jpg',img)