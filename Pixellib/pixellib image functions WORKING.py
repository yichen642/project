# Code adapted from Ayoola Lafenwa
# https://github.com/ayoolaolafenwa/PixelLib

import pixellib
from pixellib.instance import instance_segmentation
import cv2 as cv
from matplotlib import pyplot as plt
import socialdistance as sd

#USER INPUT PARAMETERS
camAngle = 60
scaleFactor = 1/40 # (dist/imgWidth) approx distance (metres) between bottom left & right corners of image
imgWidth = 1000
minimumAllowedDistance = 1

imgOrig = cv.imread('sample2edit.jpg')

# RESIZE IMAGE
width = int(imgWidth)
ratio = imgOrig.shape[1]/width
height = int(imgOrig.shape[0]/ratio)
dsize = (width, height)
print(dsize)
img = cv.resize(imgOrig, dsize)
cv.imwrite("pixellibImgResize.jpg", img)

segment_image = instance_segmentation() #rapid decrease accuracy for small people
segment_image.load_model("mask_rcnn_coco.h5")
target_classes = segment_image.select_target_classes(person=True)
segmask, img = segment_image.segmentImage("pixellibImgResize.jpg", segment_target_classes = target_classes)

cv.imwrite("pixellibImgOutput.jpg", img)

# REORGANISE SEGMASK['ROIS'] ARRAY
bblines = []
for i in range(len(segmask['rois'])):
    a = []
    a.append(segmask['rois'][i][1])
    a.append(segmask['rois'][i][0])
    a.append(segmask['rois'][i][3])
    a.append(segmask['rois'][i][2])
    bblines.append(a)

feet = sd.findFeetPix(bblines) # Calculate all feet pixel locations

imgWarped, M = sd.topViewWarp(img, camAngle, width, height) # Warp image perspective to obtain top view

feetWarped = sd.findWarpedFeetPix(feet, M) # Calculate all feet pixel locations in warped image

feetWarped = sd.adjustWarpedFeetYPix(feet, feetWarped, camAngle, height) # Adjust feet y pixel locations in warped image

pxDistance = sd.findPixDist(feet, feetWarped) # Calculate pixel distance between people's feet

realDistance = sd.pixToReal(pxDistance, scaleFactor) # Scaling pixel distance to real distance

sd.drawRedRect(img, bblines, realDistance, minimumAllowedDistance) # Draw red box around people under minimum distance & output warning message

cv.imwrite('pixellibImgWarped.jpg', imgWarped)

#DISPLAY IMAGES
plt.subplot(121),plt.imshow(img),plt.title('Original Image')
plt.subplot(122),plt.imshow(imgWarped),plt.title('Warped Image')
plt.show()

cv.waitKey(1)
cv.destroyAllWindows()
