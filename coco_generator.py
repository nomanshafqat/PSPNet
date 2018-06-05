from scipy import ndimage

from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate
import os, glob
import numpy as np
import cv2

__author__ = 'tylin'
__version__ = 0.9
import json
import datetime
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import cv2
import os


class Datahandler_COCO():
    def __init__(self, image_dir, annotation_file):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()

        self.annotation_file = annotation_file
        self.image_dir = image_dir
        print("loading dataset")

        dataset = json.load(open(self.annotation_file, 'r'))

        self.coco = COCO(self.annotation_file)
        # Load all classes (Only Building in this version)
        self.classIds = self.coco.getCatIds()
        # print(self.classIds)
        # Load all images
        self.image_ids = list(self.coco.imgs.keys())
        # print(self.image_ids)
        for image_id in self.image_ids:
            self.anns[image_id] = []

        self.categories = self.coco.loadCats([100])
        # print(self.categories)
        for object in dataset["annotations"]:
            self.anns[object["image_id"]].append(object)
            # print (object)
            # print(self.anns)

    def msklab(self, labels, n_labels):
        dims = labels.shape
        x = np.zeros([dims[0], dims[1], n_labels])
        for i in range(dims[0]):
            for j in range(dims[1]):
                x[i, j, labels[i][j]] = 1
        x = x.reshape(dims[0], dims[1], n_labels)
        return x

    def get_mask(self, id):

        temp = self.anns[id]
        m = self.coco.annToMask(temp[0])
        for ob in temp[1:]:
            m1 = self.coco.annToMask(ob)
            m = m | m1
        # print(temp)
        return m

    def make_batches(self, batchsize=4, inputshape=(320, 320,3), augmentation=None, Train=True):
        batch_images = []
        batch_masks = []
        list = self.image_ids
        # print(self.image_ids)

        while True:
            for id in list:
                # print(id)
                filename = self.coco.imgs[id]["file_name"]
                path = os.path.join(self.image_dir, filename)
                # print(filename)
                img = cv2.imread(path)
                mask = self.get_mask(id)
                # cv2.imwrite(filename+"bef.jpg",mask)

                img = cv2.resize(img, (700, 700))
                mask = cv2.resize(mask, (700, 700))
                if augmentation != None:
                    _aug = augmentation._to_deterministic()

                    img = _aug.augment_image(img)
                    mask = _aug.augment_image(mask)

                img = cv2.resize(img, inputshape[:-1])
                mask = cv2.resize(mask, inputshape[:-1])

                mask = self.msklab(mask, 2)
                # print(mask)
                batch_images.append(img)
                batch_masks.append(mask)

                #cv2.imwrite("gt.jpg",img)
                #cv2.imwrite("b.jpg",mask*255)
                if len(batch_images) == batchsize:
                    yield (np.array(batch_images), np.array(batch_masks))
                    batch_images = []
                    batch_masks = []

    def get_batch(self, batch_size=1, train=True):
        a = next(self.make_batches(batch_size, train))
        for b in a:
            print(b)
            return np.array(b), np.expand_dims(np.array(b), axis=-1)

            # self.coco.
            # register classes


class Default_Generator():
    def __init__(self, image_dir, annotation_dir):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()

        self.annotation_file = annotation_dir
        self.image_dir = image_dir
        self.mask_list = []
        self.img_list = []
        print("loading dataset")
        msk_dir = os.listdir(self.annotation_file)

        # mask_list=[[cv2.imread(os.path.join(annotation_dir,name)]) for name in mask_list]

        for name in msk_dir:
            if ".DS_Store" in name:
                continue
            image = cv2.imread(os.path.join(annotation_dir, name), 0)
            #print(annotation_dir, name)
            image[image < 50] = 0
            #image[image > 200] = 0
            image[image > 50] = 1

            # cv2.imshow("sds",image*255)
            # cv2.waitKey(5000)
            #print(image_dir, name)
            self.mask_list.append(image)
            image=cv2.imread(os.path.join(image_dir, name))
            self.img_list.append(image)

            #print("len",image.shape)
            # cv2.destroyAllWindows()

    def msklab(self, labels, n_labels):
        dims = labels.shape
        x = np.zeros([dims[0], dims[1], n_labels])
        for i in range(dims[0]):
            for j in range(dims[1]):
                x[i, j, labels[i][j]] = 1
        x = x.reshape(dims[0], dims[1], n_labels)
        return x

    def make_batches(self, batchsize=4, inputshape=(320, 320,3), augmentation=None, Train=True):
        batch_images = []
        batch_masks = []
        # print(self.image_ids)

        while True:
            for i in range(len(self.mask_list)):

                img = self.img_list[i]
                mask = self.mask_list[i]

                img = cv2.resize(img, (700, 700))
                mask = cv2.resize(mask, (700, 700))
                if augmentation != None:
                    _aug = augmentation._to_deterministic()

                    img = _aug.augment_image(img)
                    mask = _aug.augment_image(mask)

                img = cv2.resize(img, inputshape[:-1])
                mask = cv2.resize(mask, inputshape[:-1])
                #cv2.imwrite("gt.jpg", img)
                #v2.imwrite("b.jpg", mask * 255)

                mask = self.msklab(mask, 2)

                batch_images.append(img)
                batch_masks.append(mask)

                if len(batch_images) == batchsize:
                    yield (np.array(batch_images), np.array(batch_masks))
                    batch_images = []
                    batch_masks = []

