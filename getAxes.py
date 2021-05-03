#!/usr/bin/env python3

import sys
import math

import cv2
import numpy as np
from collections import defaultdict

ERRORVAL = np.nan

def grey(image):
    """ Returns gray scale """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gauss(image):
    """ Returns gaussian bluring"""
    return cv2.GaussianBlur(image, (5, 5), 0)

def canny(image):
    """ Returns edges """
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
        for j, next_group in enumerate(lines[i + 1:],i+1):
            for line1 in group:
                for line2 in next_group:
                    intersect = line_intersect(line1, line2)
                    if intersect[0][0] is not None:
                        intersections.append(intersect) 
                        intLines.append([i,j])


    return intersections, intLines

def segmented_intersections_with_axis(axis, lines):
    """Finds the intersections between axis and lines."""

    intersections = []
    intLines = []
    for i, line in enumerate(lines):
        intersect = line_intersect(axis[0], line[0])
        if intersect[0][0] is not None:
            intersections.append(intersect) 
            intLines.append([i])
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
    y_ll = 0 #1e10
    for intersection in intersections:
        x, y = intersection[0]
        if x<=x_ll and y>=y_ll:
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

def correct_coordinates(line, dx, dy): 
    """Corrects line coordinates """
    x1, y1, x2, y2 = line[0]
    return [[int(x1+dx), int(y1+dy), int(x2+dx), int(y2+dy)]]

def checkOrtho(line1,line2): 
    """Checks if two straight lines are orthogonal or not """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    # Both lines have infinite slope 
    if (x2 - x1 == 0 and x4 - x3 == 0): 
        return False
  
    # Only line 1 has infinite slope 
    elif (x2 - x1 == 0): 
        m2 = (y4 - y3) / (x4 - x3) 
  
        if (m2 == 0): 
            return True
        else: 
            return False
  
    # Only line 2 has infinite slope 
    elif (x4 - x3 == 0): 
        m1 = (y2 - y1) / (x2 - x1); 
  
        if (m1 == 0): 
            return True
        else: 
            return False
  
    else: 
          
        # Find slopes of the lines 
        m1 = (y2 - y1) / (x2 - x1) 
        m2 = (y4 - y3) / (x4 - x3) 
  
        # Check if their product is -1 
        if (m1 * m2 == -1): 
            return True
        else: 
            return False

def leave_ortho_lines(axis, lines):
    """Loops over the lines and only orthogonal to the axis"""
    ortho_lines = []
    for line in lines:
        if checkOrtho(axis,line):
            ortho_lines.append(line)
    return ortho_lines


# Read image 
#figureName = '../sPHENIX/LFT/currents_GEM3_NoGlue_zoom_16.png'
figureName = './Figures/R_Current_1.png'
#figureName = './Figures/h_dataMC_eta_PixelSharedHits.png'
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


# Choose lower left corner and associated lines
startingPoint = find_lower_left_point(intersections)
# Assign axes
X, Y = assign_axes(lines, intLines, intersections, startingPoint)
print(X)
print(Y)

Xx1, Xy1, Xx2, Xy2 = X[0]  
Yx1, Yy1, Yx2, Yy2 = Y[0]  

# Crop image for X & Y
# Sets the Region of Interest 
# Create destination image 
height, width, channels = img.shape
pixels_to_cut_Y = int((height- Xy1)*0.3)
pixels_to_cut_X = int(Yx1*0.3)
print("pixels_to_cut_X = {}".format(pixels_to_cut_X))
# X-axis ranges
range_Y1_X = Xy1-pixels_to_cut_Y
range_Y2_X = Xy2+15
range_X1_X = Xx1
range_X2_X = Xx2
# Y-axis ranges
range_Y1_Y = Yy2
range_Y2_Y = Yy1
range_X1_Y = Yx1-15
range_X2_Y = Yx2+pixels_to_cut_X
img_X = img[range_Y1_X:range_Y2_X , range_X1_X:range_X2_X]
img_Y = img[range_Y1_Y:range_Y2_Y , range_X1_Y:range_X2_Y]



# Prepare images for line search algorithm
grey_img_X  = grey(img_X)
gauss_img_X = gauss(grey_img_X)
edges_img_X = canny(gauss_img_X)

grey_img_Y  = grey(img_Y)
gauss_img_Y = gauss(grey_img_Y)
edges_img_Y = canny(gauss_img_Y)

# Detect points that form a line

#Create FLD detector
#Param               Default value   Description
#length_threshold    10            - Segments shorter than this will be discarded
#distance_threshold  1.41421356    - A point placed from a hypothesis line
#                                    segment farther than this will be
#                                    regarded as an outlier
#canny_th1           50            - First threshold for
#                                    hysteresis procedure in Canny()
#canny_th2           50            - Second threshold for
#                                    hysteresis procedure in Canny()
#canny_aperture_size 3             - Aperturesize for the sobel
#                                    operator in Canny()
#do_merge            false         - If true, incremental merging of segments
#                                    will be perfomred
length_threshold = 15
distance_threshold = 40
canny_th1 = 50.0
canny_th2 = 50.0
canny_aperture_size = 3
do_merge = False #True


fld_Y = cv2.ximgproc.createFastLineDetector(length_threshold,distance_threshold,canny_th1,canny_th2,canny_aperture_size,do_merge)

length_threshold = 12
fld_X = cv2.ximgproc.createFastLineDetector(length_threshold,distance_threshold,canny_th1,canny_th2,canny_aperture_size,do_merge)
lines_Y = fld_Y.detect(grey_img_Y)
lines_X = fld_X.detect(grey_img_X)



# Check that lines are penpendicular to the X or Y axes
# Function to 
ortho_lines_X = leave_ortho_lines(X, lines_X)
ortho_lines_Y = leave_ortho_lines(Y, lines_Y)

ortho_lines_X_corrected = []
ortho_lines_Y_corrected = []

dx_X = range_X1_X
dy_X = range_Y1_X
dx_Y = range_X1_Y
dy_Y = range_Y1_Y
for line in ortho_lines_X:
    ortho_lines_X_corrected.append(correct_coordinates(line, dx_X, dy_X))
for line in ortho_lines_Y:
    ortho_lines_Y_corrected.append(correct_coordinates(line, dx_Y, dy_Y))

# Find intersections and associate lines
intersections_X, intLines_X = segmented_intersections_with_axis(X, ortho_lines_X_corrected)
intersections_Y, intLines_Y = segmented_intersections_with_axis(Y, ortho_lines_Y_corrected)

cv2.imwrite('./axis_edges_X.jpg',edges_img_X)
cv2.imwrite('./axis_edges_Y.jpg',edges_img_Y)

# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
for line in ortho_lines_X_corrected:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)


for line in ortho_lines_Y_corrected:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

#Draw intersections
#mark_intersections(img,intersections, (255, 153, 255), markerType=cv2.MARKER_TILTED_CROSS , markerSize=15)
mark_intersections(img,[[startingPoint]], (255, 153, 255), markerType=cv2.MARKER_TILTED_CROSS , markerSize=15)
mark_intersections(img,intersections_X, (255, 153, 255), markerType=cv2.MARKER_TILTED_CROSS , markerSize=15)
mark_intersections(img,intersections_Y, (255, 153, 255), markerType=cv2.MARKER_TILTED_CROSS , markerSize=15)
# Show result
#cv2.imshow("Result Image", img)
cv2.imwrite('./result.jpg',img)

cv2.imwrite('./axis_X.jpg',img_X)
cv2.imwrite('./axis_Y.jpg',img_Y)
