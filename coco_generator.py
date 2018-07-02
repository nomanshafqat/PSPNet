

__author__ = 'tylin'
__version__ = 0.9
import json
import numpy as np
from pycocotools.coco import COCO
import cv2
import os
import xml.etree.cElementTree as ET
from six import raise_from


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

    def make_batches(self, batchsize=4, inputshape=(320, 320, 3), augmentation=None, Train=True):
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

                # cv2.imwrite("gt.jpg",img)
                # cv2.imwrite("b.jpg",mask*255)
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
            # print(annotation_dir, name)
            image[image < 50] = 0
            # image[image > 200] = 0
            image[image > 50] = 1

            # cv2.imshow("sds",image*255)
            # cv2.waitKey(5000)
            # print(image_dir, name)
            self.mask_list.append(image)
            image = cv2.imread(os.path.join(image_dir, name))
            self.img_list.append(image)

            # print("len",image.shape)
            # cv2.destroyAllWindows()

    def msklab(self, labels, n_labels):
        dims = labels.shape
        x = np.zeros([dims[0], dims[1], n_labels])
        for i in range(dims[0]):
            for j in range(dims[1]):
                x[i, j, labels[i][j]] = 1
        x = x.reshape(dims[0], dims[1], n_labels)
        return x

    def make_batches(self, batchsize=4, inputshape=(320, 320, 3), augmentation=None, Train=True):
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
                # cv2.imwrite("gt.jpg", img)
                # v2.imwrite("b.jpg", mask * 255)

                mask = self.msklab(mask, 2)

                batch_images.append(img)
                batch_masks.append(mask)

                if len(batch_images) == batchsize:
                    yield (np.array(batch_images), np.array(batch_masks))
                    batch_images = []
                    batch_masks = []


class Pascal_Generator():
    def __init__(self, data_dir, annotation_file):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()

        self.annotation_file = annotation_file
        self.data_dir = data_dir
        self.mask_list = []
        self.img_list = []

        print("loading dataset")
        self.image_names = [l.strip().split(None, 1)[0] for l in open(
            os.path.join(self.data_dir, 'ImageSets', 'Main', self.annotation_file + '.txt')).readlines()]

        # mask_list=[[cv2.imread(os.path.join(annotation_dir,name)]) for name in mask_list]


    def msklab(self, labels, n_labels):
        dims = labels.shape
        print(dims)
        x = np.zeros([dims[0], dims[1], n_labels])
        for i in range(dims[0]):
            for j in range(dims[1]):
                x[i, j, labels[i][j]] = 1
        x = x.reshape(dims[0], dims[1], n_labels)
        return x

    def make_batches(self, batchsize=4, inputshape=(320, 320, 3), augmentation=None, Train=True):
        batch_images = []
        batch_masks = []
        # print(self.image_ids)

        while True:
            for i,name in enumerate(self.image_names):
                print(os.path.join(self.data_dir,"JPEGImages",name+".jpg"))
                img = cv2.imread(os.path.join(self.data_dir,"JPEGImages",name+".jpg"))
                annotation=self.load_annotations(name+".xml")
                print(annotation)
                mask=np.zeros(img.shape[:-1],dtype=np.uint8)

                for box in annotation:
                    xmin,ymin,xmax,ymax=box
                    xmin=int(xmin)
                    xmax=int(xmax)
                    ymin = int(ymin)
                    ymax = int(ymax)
                    if "image" not in name:
                        ymin=int(ymin*1.34)
                        ymax=int(ymax*1.34)

                    aspect=((ymax-ymin)/(xmax-xmin))
                    mask[ymin:ymax,xmin:xmax]=1
                img[:,:,1]=mask*255

                cv2.imwrite("mask.jpg",mask*255)

                cv2.imwrite("img.jpg",img)
                print(aspect)
                if  aspect < 9:
                    print("skip...",i)
                    continue
                if augmentation != None:
                    _aug = augmentation._to_deterministic()

                    img = _aug.augment_image(img)
                    mask = _aug.augment_image(mask)

                img = cv2.resize(img, inputshape[:-1])
                mask = cv2.resize(mask, inputshape[:-1])


                mask = self.msklab(mask, 2)
                #cv2.imwrite("gt.jpg", img)
                cv2.imwrite("b.jpg", mask[:,:,1] * 255)

                batch_images.append(img)
                batch_masks.append(mask)

                if len(batch_images) == batchsize:
                    yield (np.array(batch_images), np.array(batch_masks))
                    batch_images = []
                    batch_masks = []

    def __parse_annotation(self, element):
        truncated = int(0)
        difficult = int(0)

        class_name = _findNode(element, 'name').text

        box = np.zeros((1, 4))

        bndbox = _findNode(element, 'bndbox')
        box[0, 0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[0, 1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[0, 2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[0, 3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1
        # print((box[0, 3]-box[0, 1])/(box[0, 2]-box[0, 0]))
        return truncated, difficult, box

    def __parse_annotations(self, xml_root):
        boxes = np.zeros((0, 4))
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            boxes = np.append(boxes, box, axis=0)

        return boxes

    def load_annotations(self, filename):
        try:
            tree = ET.parse(os.path.join(self.data_dir, 'Annotations', filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)

def _findNode(parent, name, debug_name = None, parse = None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


pg=Pascal_Generator("/Users/nomanshafqat/Google Drive/upwork/wholedataset","/Users/nomanshafqat/Google Drive/upwork/wholedataset/ImageSets/Main/trainval")
a=pg.make_batches(100)
next(a)

