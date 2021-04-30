import cv2 as cv
from matplotlib import pyplot as plt
import socialdistance as sd

#USER INPUT PARAMETERS
camAngle = 60
scaleFactor = 10/1000 # (dist/imgWidth) approx distance (metres) between bottom left & right corners of image
minimumDistance = 1

# INITIALISE THE HOG PERSON DETECTOR
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cv.startWindowThread()

# OPEN WEBCAM VIDEO STREAM
cap = cv.VideoCapture(0)

# CHECK FRAME SIZE
if cap.isOpened():
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # float `height`
    print('width, height:', width, height)
    fps = cap.get(cv.CAP_PROP_FPS)
    print(fps) #30

# CREATE VIDEO OUTPUT IN .AVI FORMAT
out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'MJPG'), 20.0, (640,480))

while (True):
    ret, frame = cap.read() # Capture video footage
    frame = cv.resize(frame, (width, height))  # Resize for faster detection

    (area, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05) # Detection of persons in image
    img = frame

    bblines = sd.drawGreenRect(area, img)  # Draw green box for detection of persons & return bounding box lines

    feet = sd.findFeetPix(bblines) # Calculate all feet pixel locations

    imgWarped, M = sd.topViewWarp(img, camAngle, width, height) # Warp image perspective to obtain top view

    feetWarped = sd.findWarpedFeetPix(feet, M)  # Calculate all feet pixel locations in warped image

    feetWarped = sd.adjustWarpedFeetYPix(feet, feetWarped, camAngle, height) # Adjust feet y pixel locations in warped image

    pxDistance = sd.findPixDist(feet, feetWarped)  # Calculate pixel distance between people's feet

    realDistance = sd.pixToReal(pxDistance, scaleFactor)  # Scaling pixel distance to real distance

    sd.drawRedRect(img, bblines, realDistance, minimumDistance)  # Draw blue box around people under minimum distance & output warning message

    # WRITE THE OUTPUT VIDEO
    out.write(frame.astype('uint8'))
    # DISPLAY THE RESULTING FRAME
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When finished, release the capture, release the output, & close all windows
cap.release()
out.release()
cv.destroyAllWindows()