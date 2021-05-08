# Code adapted from Ayoola Lafenwa
# https://github.com/ayoolaolafenwa/PixelLib

import pixellib
from pixellib.instance import instance_segmentation
import cv2 as cv
import socialdistance as sd

#USER INPUT PARAMETERS
camAngle = 60
scaleFactor = 1/40 # (dist/imgWidth) approx distance (metres) between bottom left & right corners of image
imgWidth = 1000
minimumAllowedDistance = 1

# INITIALISE THE PIXELLIB PERSON DETECTOR
segment_video = instance_segmentation(infer_speed = "rapid")
segment_video.load_model("mask_rcnn_coco.h5")
target_classes = segment_video.select_target_classes(person=True)

# https://pixellib.readthedocs.io/en/latest/video_instance.html
# https://towardsdatascience.com/video-segmentation-with-5-lines-of-code-87f798afb93

# OPEN WEBCAM VIDEO STREAM
capture = cv.VideoCapture(0)

# CHECK FRAME SIZE
if capture.isOpened():
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))  # float `height`
    print('width, height:', width, height)
    fps = capture.get(cv.CAP_PROP_FPS)
    print(fps) #30

# CREATE VIDEO OUTPUT IN .AVI FORMAT
out = cv.VideoWriter('pixelibVideoOutput.avi', cv.VideoWriter_fourcc(*'MJPG'), 20.0, (640, 480))

while True:
    ret, frame = capture.read() # Capture video footage
    frame = cv.resize(frame, (width, height)) # Resize for faster detection
    segmask, img = segment_video.segmentFrame(frame) # Detection of persons in image

    # REORGANISE SEGMASK['ROIS'] ARRAY
    bblines = [] # Reorganise segmaskarray
    for i in range(len(segmask['rois'])):
        a = []
        a.append(segmask['rois'][i][1])
        a.append(segmask['rois'][i][0])
        a.append(segmask['rois'][i][3])
        a.append(segmask['rois'][i][2])
        bblines.append(a)

    feet = sd.findFeetPix(bblines)  # Calculate all feet pixel locations

    imgWarped, M = sd.topViewWarp(img, camAngle, width, height)  # Warp image perspective to obtain top view

    feetWarped = sd.findWarpedFeetPix(feet, M)  # Calculate all feet pixel locations in warped image

    feetWarped = sd.adjustWarpedFeetYPix(feet, feetWarped, camAngle, height)  # Adjust feet y pixel locations in warped image

    pxDistance = sd.findPixDist(feet, feetWarped)  # Calculate pixel distance between people's feet

    realDistance = sd.pixToReal(pxDistance, scaleFactor)  # Scaling pixel distance to real distance

    sd.drawRedRect(img, bblines, realDistance, minimumAllowedDistance)  # Draw blue box around people under minimum distance & output warning message

    # WRITE THE OUTPUT VIDEO
    out.write(frame.astype('uint8'))
    # DISPLAY THE RESULTING FRAME
    cv.imshow("frame", frame)
    if cv.waitKey(25) & 0xff == ord('q'):
        break

# When finished, release the capture, release the output, & close all windows
capture.release()
out.release()
cv.destroyAllWindows()
