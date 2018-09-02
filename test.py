import os
from sys import argv

import matplotlib.pyplot as plt
from PSPNet import PSPNet50
import cv2
import numpy as np


def Get_Bounding_Box(colored_image, mask_im):

    gray_masked = np.array(mask_im).astype(np.uint8)
    dilated = cv2.dilate(gray_masked, np.ones((11, 11)))
    im2, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)
    output = cv2.connectedComponentsWithStats(im2, 4, cv2.CV_32S)

    index = np.argmax(output[2][1:], 0)[4] + 1  # will give output

    # get labelled matrix
    labelled = output[1]

    # make an image with the connected component of largest value
    im = np.array((labelled == index) * 255, dtype=np.uint8)

    # get bounding rectangle
    x, y, w, h = cv2.boundingRect(im)
    print(x, y, w, h)
    x_scale=512/colored_image.shape[1]
    y_scale=512/colored_image.shape[0]
    print(x_scale,y_scale)
    x=int(x/x_scale)
    y=int(y/y_scale)
    w=int(w/x_scale)
    h=int(h/y_scale)

    print(x, y, w, h)

    return x, y, x+w, y+h



def inference(model, inputs):
    inputs = np.array(inputs)
    inputs = inputs.reshape((1, 512, 512, 3))
    outputs = model.predict(inputs)
    outputs = outputs.reshape(512, 512, 2)
    outputs = outputs.argmax(axis=2)

    return outputs

weight ="/Users/nomanshafqat/Google Drive/upwork/backupPSP/LIP_PSPNet50_mask15.hdf5"
imgdir="/Users/nomanshafqat/Downloads/Archive"

# load model
model = PSPNet50()
model.load_weights(weight)
files = os.listdir(imgdir)
for filename in files:
    print(filename)
    if ".DS_Store" in filename:
        continue
    # inference
    inputs = cv2.imread(os.path.join(imgdir, filename))
    orig=inputs.copy()
    inputs= cv2.resize(inputs, (512, 512),interpolation=cv2.INTER_LINEAR)
    outputs = inference(model, inputs)
    #cv2.imwrite("results/" + filename+"_0.jpg", np.array( outputs, "uint8")*255)
    #inputs[:, :, 0] = inputs[:, :, 0] * (1 - outputs)
    #inputs[:, :, 1] = inputs[:, :, 1] * (1 - outputs)

    #cv2.imwrite("results/" + filename+"_re.jpg", np.array( inputs, "uint8"))

    print(outputs.shape)
    xmin,ymin,xmax,ymax =Get_Bounding_Box(orig,outputs*255)

    cv2.rectangle(orig,(xmin,ymin),(xmax,ymax),(0,0,255),5)

    print(outputs)
    cv2.imwrite("results/" + filename+".jpg", np.array( orig, "uint8"))
