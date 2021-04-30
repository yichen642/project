import cv2 as cv
import math
import numpy as np


# DRAW RECTANGULAR AREA FOR DETECTION OF PERSONS
def drawGreenRect(area, Image):
    img = Image
    bblines = []
    for (x, y, w, h) in area:
        a = []
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # Draw green rectangle
        a.append(x)
        a.append(y)
        a.append(x + w)
        a.append(y + h)
        bblines.append(a)

    return bblines


#CALCULATE FEET PIXEL LOCATION OF ALL BOUNDING BOXES
def findFeetPix(RectangleLines):
    bblines = RectangleLines
    feet = []
    for i in range(len(bblines)):  # A for loop for row entries
        a = []
        a.append((bblines[i][0] + bblines[i][2]) // 2) #find x centre
        a.append(bblines[i][3])
        feet.append(a)

    return feet


#WARP IMAGE PERSPECTIVE TO TOP VIEW
def topViewWarp(Image, CameraAngle, width, height):
    img = Image
    camAngle = CameraAngle
    if 0 <= camAngle <= 60:
        warpExtend = 0.00036265*(camAngle**4) - 0.0262*(camAngle**3) + 0.7407*(camAngle**2) + 1.5556*camAngle # equation for warp
        warpExtend = warpExtend * width / 1000 # for different widths
        pts1 = np.float32([[0,-0],[width,0],[-warpExtend,height],[width+warpExtend,height]]) #points for waro
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]]) #framesize of new image warp
        M = cv.getPerspectiveTransform(pts1,pts2)
        imgWarped = cv.warpPerspective(img,M,(width,height)) # w and h is the display framesize

    elif 60 < camAngle <= 75:
        warpExtend = (250/3) * camAngle - 4380  # equation for warp
        warpExtend = warpExtend * width / 1000  # for different widths
        pts1 = np.float32([[0, 0.3 * height], [width, 0.3 * height], [-warpExtend, height], [width + warpExtend, height]])  # points for waro
        pts2 = np.float32([[0, 0], [width, 0], [0, 0.7 * height], [width, 0.7 * height]])  # framesize of new image warp
        M = cv.getPerspectiveTransform(pts1, pts2)
        imgWarped = cv.warpPerspective(img, M, (width, int(0.7 * height)))  # w and h is the display framesize

    elif 75 < camAngle <= 90:
        warpExtend = (110) * camAngle - 7840  # equation for warp
        warpExtend = warpExtend * width / 1000  # for different widths
        pts1 = np.float32([[0, 0.6 * height], [width, 0.6 * height], [-warpExtend, height], [width + warpExtend, height]])  # points for waro
        pts2 = np.float32([[0, 0], [width, 0], [0, 0.4* height], [width, 0.4 * height]])  # framesize of new image warp
        M = cv.getPerspectiveTransform(pts1, pts2)
        imgWarped = cv.warpPerspective(img, M, (width, int(0.4 * height)))  # w and h is the display framesize

    else:
        print('Enter a camera angle between 0 and 90 degrees!')
        quit()

    return imgWarped, M

# CALCULATE FEET PIXEL LOCATIONS IN WARPED IMAGE
def findWarpedFeetPix(FeetPixelLocations, WarpMatrix):
    feet = FeetPixelLocations
    M = WarpMatrix
    feetWarped = [[0 for col in range(3)] for row in range(len(feet))]
    for i in range(len(feet)): # length is 3 of ROWS
        for j in range(3):
            feetWarped[i][j] = M[j][0]*(feet[i][0]) + (M[j][1])*(feet[i][1]) + (M[j][2])*1

    return feetWarped

# ADJUST FEET Y PIXELS LOCATIONS IN WARPED IMAGE
def adjustWarpedFeetYPix(FeetPixelLocations, WarpedFeetPixelLocations,CameraAngle, height):
    feet = FeetPixelLocations
    feetWarped = WarpedFeetPixelLocations
    camAngle = CameraAngle
    if 0 <= camAngle <= 60:
        for i in range(len(feetWarped)):
            origMultiplier = ((feetWarped[i][1]/feetWarped[i][2])/feet[i][1])
            newMultiplier = (origMultiplier+1)/2
            feetWarped[i][1] = feet[i][1]*newMultiplier # Replace old feetWarped Y value

    elif 60 < camAngle <= 75:
        for i in range(len(feetWarped)):
            origMultiplier = ((feetWarped[i][1]/feetWarped[i][2]) / (feet[i][1])-0.3*height)
            newMultiplier = (origMultiplier + 1) / 2
            feetWarped[i][1] = ((feet[i][1]) - 0.3 * height) * newMultiplier  # Replace old feetWarped Y value

    elif 75 < camAngle <= 90:
        for i in range(len(feetWarped)):
            origMultiplier = ((feetWarped[i][1]/feetWarped[i][2]) / (feet[i][1])-0.6*height)
            newMultiplier = (origMultiplier + 1) / 2
            feetWarped[i][1] = ((feet[i][1])-0.6*height) * newMultiplier  # Replace old feetWarped Y value
    return feetWarped

#CALCULATE PIXEL DISTANCE BETWEEN PEOPLE'S FEET
def findPixDist(FeetPixelLocations, WarpedFeetPixelLocations):
    feet = FeetPixelLocations
    feetWarped = WarpedFeetPixelLocations
    pxDistance = []
    for i in range(len(feet)-1):
        i2 = i
        while i2 < (len(feet)-1):
            a = []
            x_abs_diff = abs(((feetWarped[i][0]) / feetWarped[i][2]) - (feetWarped[i2 + 1][0] / feetWarped[i2 + 1][2]))
            a.append(x_abs_diff)
            y_abs_diff = abs((feetWarped[i][1]) - (feetWarped[i2 + 1][1]))
            a.append(y_abs_diff)
            euclidean = math.sqrt(x_abs_diff ** 2 + y_abs_diff ** 2)
            a.append(euclidean)
            a.append(i+1) # 1st person
            a.append(i2+2) # 2nd person
            pxDistance.append(a)
            i2 = i2 + 1

    return pxDistance

#SCALING PIXEL DISTANCE TO REAL DISTANCE
def pixToReal(PixelDistance,scaleFactor):
    pxDistance = PixelDistance
    realDistance = []
    for i in range(len(pxDistance)):
        a = []
        for j in range(3):
            a.append(pxDistance[i][j] * scaleFactor)
        a.append(pxDistance[i][3])
        a.append(pxDistance[i][4])
        realDistance.append(a)

    return realDistance

#OUTPUT MESSAGE/ALERT IF LESS THAN X METRES & #HIGHLIGHT PEOPLE IN RED FOR IMAGES & BLUE FOR VIDEOS
def drawRedRect(Image, RectangleLines, realDistance, minimumAllowedDistance):
    img = Image
    bblines = RectangleLines
    for i in range(len(realDistance)):
        if realDistance[i][2] < minimumAllowedDistance:
            print('Person', realDistance[i][3], '&', realDistance[i][4], 'please social distance!')
            for j in range(2):
                cv.rectangle(img,
                             (bblines[(realDistance[i][j+3])-1][0], bblines[(realDistance[i][j+3])-1][1]),
                             (bblines[(realDistance[i][j+3])-1][2], bblines[(realDistance[i][j+3])-1][3]),
                             (255, 0, 0),
                             2)

    return