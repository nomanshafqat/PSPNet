import os
from sys import argv

import matplotlib.pyplot as plt
from PSPNet import PSPNet50
import cv2
import numpy as np


def inference(model, inputs):
    orig_w, orig_h = inputs.shape[0:2]
    inputs = cv2.resize(inputs, (512, 512))
    inputs = np.array(inputs)
    inputs = inputs.reshape((1, 512, 512, 3))
    outputs = model.predict(inputs)
    outputs = outputs.reshape(512, 512, 2)
    outputs = outputs.argmax(axis=2)

    return outputs

weight =argv[1]
imgdir=argv[2]
#"./pretrained_mask/LIP_PSPNet50_mask02.hdf5"

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
    outputs = inference(model, inputs)
    inputs=cv2.resize(inputs, (512, 512))

    inputs[:, :, 0] = inputs[:, :, 0] * (1 - outputs)
    inputs[:, :, 1] = inputs[:, :, 1] * (1 - outputs)
    print(outputs)
    cv2.imwrite("results/" + filename+"re.jpg", np.array(outputs, "uint8") * 255)
    cv2.imwrite("results/" + filename+".jpg", np.array( inputs, "uint8"))

    # plt.show()

