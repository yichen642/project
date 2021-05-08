# Code adapted from Satya Mallik
# https://github.com/spmallick/learnopencv/tree/master/PyTorch-Mask-RCNN

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
import time
import os
import socialdistance as sd

# USER INPUT PARAMETERS
camAngle = 60
scaleFactor = 10 / 1000  # (dist/imgWidth) approx distance (metres) between bottom left & right corners of image
imgWidth = 1000
minimumDistance = 1

# INITIALISE THE TORCHVISION PERSON DETECTOR
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# OPEN WEBCAM VIDEO STREAM
capture = cv2.VideoCapture(0)

# CHECK FRAME SIZE
if capture.isOpened():
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    print('width, height:', width, height)
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(fps) #30

# CREATE VIDEO OUTPUT IN .AVI FORMAT
out = cv2.VideoWriter('TVVideoOutput.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (640, 480))

# These are the classes that are available in the COCO-Dataset
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def random_colour_masks(image):
    """
    random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img_path, threshold):
    """
    get_prediction
    parameters:
      - img_path - path of the input image
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
        ie: eg. segment of cat is made 1 and rest of the image is made 0
    """
    img = frame
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class

def instance_segmentation_api(img_path, threshold=0.5, rect_th=3, text_size=1, text_th=1):
    """
    instance_segmentation_api
    parameters:
      - img_path - path to input image
    method:
      - prediction is obtained by get_prediction
      - each mask is given random color
      - each mask is added to the image in the ration 1:0.8 with opencv
      - final output is displayed
    """
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    img = frame
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=2)

    return boxes, img

# We will use the following colors to fill the pixels
colours = [[0, 255, 0],
           [0, 0, 255],
           [255, 0, 0],
           [0, 255, 255],
           [255, 255, 0],
           [255, 0, 255],
           [80, 70, 180],
           [250, 80, 190],
           [245, 145, 50],
           [70, 150, 250],
           [50, 190, 190]]

while True:
    ret, frame = capture.read() # Capture video footage
    frame = cv2.resize(frame, (width, height)) # Resize for faster detection

    boxes, img = instance_segmentation_api(frame, 0.75) # Detection of persons in image

    # REORGANISE BOXES MATRIX AND CONVERT TO INTEGERS
    bblines = []
    for i in range(len(boxes)):
        a = []
        a.append(round(boxes[i][0][0]))
        a.append(round(boxes[i][0][1]))
        a.append(round(boxes[i][1][0]))
        a.append(round(boxes[i][1][1]))
        bblines.append(a)

    feet = sd.findFeetPix(bblines)  # Calculate all feet pixel locations

    imgWarped, M = sd.topViewWarp(img, camAngle, width, height)  # Warp image perspective to obtain top view

    feetWarped = sd.findWarpedFeetPix(feet, M)  # Calculate all feet pixel locations in warped image

    feetWarped = sd.adjustWarpedFeetYPix(feet, feetWarped, camAngle, height)  # Adjust feet y pixel locations in warped image

    pxDistance = sd.findPixDist(feet, feetWarped)  # Calculate pixel distance between people's feet

    realDistance = sd.pixToReal(pxDistance, scaleFactor)  # Scaling pixel distance to real distance

    sd.drawRedRect(img, bblines, realDistance, minimumDistance)  # Draw red box around people under minimum distance & output warning message

    # WRITE THE OUTPUT VIDEO
    out.write(frame.astype('uint8'))
    # DISPLAY THE RESULTING FRAME
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When finished, release the capture, release the output, & close all windows
cap.release()
out.release()
cv.destroyAllWindows()
