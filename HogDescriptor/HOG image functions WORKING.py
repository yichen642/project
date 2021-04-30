import cv2 as cv
from matplotlib import pyplot as plt
import socialdistance as sd

#USER INPUT PARAMETERS
camAngle = 60
scaleFactor = 10/1000 # (dist/imgWidth) approx distance (metres) between bottom left & right corners of image
imgWidth = 1000
minimumDistance = 1

# INITIALISE THE HOG PERSON DETECTOR
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

# READ IMAGE
imgOrig = cv.cvtColor(cv.imread('silhouette.jpg'), cv.COLOR_BGR2RGB) # Convert image to RGB

# RESIZE IMAGE
width = int(imgWidth)
ratio = imgOrig.shape[1]/width
height = int(imgOrig.shape[0]/ratio)
dsize = (width, height)
img = cv.resize(imgOrig, dsize)

# DETECTION OF PERSONS INSIDE THE IMAGE
(area, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(4, 4), scale=1.05)

bblines = sd.drawGreenRect(area, img) # Draw green box for detection of persons & return bounding box lines

feet = sd.findFeetPix(bblines) # Calculate all feet pixel locations

imgWarped, M = sd.topViewWarp(img, camAngle, width, height) # Warp image perspective to obtain top view

feetWarped = sd.findWarpedFeetPix(feet, M) # Calculate all feet pixel locations in warped image

feetWarped = sd.adjustWarpedFeetYPix(feet, feetWarped, camAngle, height) # Adjust feet y pixel locations in warped image

pxDistance = sd.findPixDist(feet, feetWarped) # Calculate pixel distance between people's feet

realDistance = sd.pixToReal(pxDistance, scaleFactor) # Scaling pixel distance to real distance

sd.drawRedRect(img, bblines, realDistance, minimumDistance) # Draw red box around people under minimum distance & output warning message

cv.imwrite('HOGImageWarped.jpg', imgWarped)

#DISPLAY IMAGES
plt.subplot(121),plt.imshow(img),plt.title('Original Image')
plt.subplot(122),plt.imshow(imgWarped),plt.title('Warped Image')
plt.show()

cv.waitKey(1)
cv.destroyAllWindows()