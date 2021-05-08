# Code adapted from Satya Mallik
# https://github.com/spmallick/learnopencv/tree/master/PyTorch-Segmentation-torchvision

from torchvision import models
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import numpy as np
import torchvision.transforms as tv

# Define the helper function
def decode_segmap(image, nc=21): #https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    # Black image  rgb is 0 0 0
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    #Join a sequence of arrays
    rgb = np.stack([r, g, b], axis=2)
    return rgb


imgOrig = cv.imread('kids.jpeg')

# Resize image
width = int(600)
ratio = imgOrig.shape[1]/width
height = int(imgOrig.shape[0]/ratio)
dsize = (width, height)
print(dsize)
img = cv.resize(imgOrig, dsize)

# You may need to convert the color.
imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Apply required transformations
transf = tv.Compose([tv.ToTensor(), #Converts the image to type torch.Tensor and scales the values to [0, 1] range
                     tv.Normalize(mean = [0.485, 0.456, 0.406], #Normalizes the image with the given mean and standard deviation.
                              std = [0.229, 0.224, 0.225])])
inp = transf(imgRGB).unsqueeze(0)

# Pass the input through the net
out = fcn(inp)['out']
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
segmentRGB = decode_segmap(om)

#Merge images
overlap = cv.add(imgRGB,segmentRGB)

titles = ['Original Image', 'shape coloured','overlap']
images = [imgRGB, segmentRGB, overlap]

for i in range(3):
    plt.subplot(1, 3, i + 1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
