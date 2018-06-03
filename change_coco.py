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

import sys
class Datahandler_COCO():
    def __init__(self,image_dir,annotation_file):
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()


        self.annotation_file=annotation_file
        self.image_dir = image_dir
        print("loading dataset")

        dataset = json.load(open(self.annotation_file, 'r'))

        self.coco = COCO(self.annotation_file)
        # Load all classes (Only Building in this version)
        self.classIds = self.coco.getCatIds()
        #print(self.classIds)
        # Load all images
        self.image_ids = list(self.coco.imgs.keys())
        #print(self.image_ids)
        for image_id in self.image_ids:
            self.anns[image_id]=[]

        self.categories=self.coco.loadCats([100])
        #print(self.categories)
        for object in dataset["annotations"]:
            self.anns[object["image_id"]].append(object)
            #print (object)
        #print(self.anns)
    def get_mask(self,id):

        temp=self.anns[id]
        m=self.coco.annToMask(temp[0])
        for ob in temp[1:]:
            m1=self.coco.annToMask(ob)
            m=m|m1
        #print(temp)
        return m

    def make_batches(self,batchsize=4,Train=True):
        batch_images = []
        batch_masks = []
        list = self.image_ids
        #print(self.image_ids)

        for id in list:
            #print(id)
            filename=self.coco.imgs[id]["file_name"]
            path=os.path.join(self.image_dir,filename)
            print(path)
            img=cv2.imread(path)
            mask=self.get_mask(id)
            #img=cv2.resize(img,(320,320))
            #mask=cv2.resize(mask,(320,320))

            batch_images.append(img)

            batch_masks.append(mask)

            cv2.imwrite("images/"+filename+".jpg",img)
            cv2.imwrite("masks/"+filename+".jpg",mask*255)


    def get_batch(self,batch_size=1, train=True):
        a=next(self.make_batches(batch_size,train))
        for b in a:
            print(b)
            return np.array(b),np.expand_dims(np.array(b),axis=-1)

        #self.coco.
        # register classes

ft=Datahandler_COCO(sys.argv[0],sys.argv[1])
ft.make_batches()