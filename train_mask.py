# -*- coding: utf-8 -*-
import argparse

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint

import os
import json
from PSPNet import PSPNet50

from coco_generator import Datahandler_COCO, Default_Generator,Pascal_Generator
from imgaug import augmenters as iaa
import imgaug as ia

sometimes = lambda aug: iaa.Sometimes(0.4, aug)




def main(args):
    # set the necessary list
    # train_list = pd.read_csv(args.train_list,header=None)
    # val_list = pd.read_csv(args.val_list,header=None)

    # set the necessary directories
    trainimg_dir = args.trainimg_dir
    trainmsk_dir = args.trainmsk_dir
    valimg_dir = args.valimg_dir
    valmsk_dir = args.valmsk_dir

    # get old session
    # old_session = KTF.get_session()

    with tf.Graph().as_default():
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)

        # set callbacks
        fpath = './pretrained_mask/'+args.name+'{epoch:02d}.hdf5'
        cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, mode='auto',
                                period=1)
        tb_cb = TensorBoard(log_dir="./pretrained_mask", write_graph=True, write_images=True)

        seq = iaa.Sequential([
            iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
             sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10),  # rotate by -45 to +45 degrees
            )),
        ], random_order=True)
        if args.dataset == 'coco':
            train_gen = Datahandler_COCO(trainimg_dir, trainmsk_dir).make_batches(batchsize=args.batch_size,
                                                                                  inputshape=args.input_shape,
                                                                                  augmentation=seq)
            val_gen = Datahandler_COCO(valimg_dir, valmsk_dir).make_batches(batchsize=args.batch_size,
                                                                            inputshape=args.input_shape,
                                                                            augmentation=None)
        elif args.dataset == 'pascal_khamba':
            train_gen = Pascal_Generator(trainimg_dir, trainmsk_dir).make_batches(batchsize=args.batch_size,
                                                                                  inputshape=args.input_shape,
                                                                                  augmentation=seq)
            val_gen = Pascal_Generator(valimg_dir, valmsk_dir).make_batches(batchsize=args.batch_size,
                                                                            inputshape=args.input_shape,
                                                                            augmentation=None)

        else:
            train_gen = Default_Generator(trainimg_dir, trainmsk_dir).make_batches(batchsize=args.batch_size,
                                                                                   inputshape=args.input_shape,
                                                                                   augmentation=seq)
            val_gen = Default_Generator(valimg_dir, valmsk_dir).make_batches(batchsize=args.batch_size,
                                                                             inputshape=args.input_shape,
                                                                             augmentation=None)

        # set model
        pspnet = PSPNet50(input_shape=args.input_shape,
                          n_labels=args.n_labels,
                          output_mode=args.output_mode,
                          upsample_type=args.upsample_type)
        print(pspnet.summary())
        if args.load is not None:
            print("loadinf weights")
            pspnet.load_weights(args.load)

        # compile model
        pspnet.compile(loss=args.loss,
                       optimizer=args.optimizer,

                       metrics=["accuracy"])

        # fit with genarater
        pspnet.fit_generator(generator=train_gen,
                             steps_per_epoch=args.steps,
                             epochs=args.epochs,
                             validation_data=val_gen,
                             validation_steps=args.val_steps,
                             callbacks=[cp_cb, tb_cb],
                             verbose=True)

    # save model
    with open("./pretrained_mask/"+args.name+".json", "w") as json_file:
        json_file.write(json.dumps(json.loads(pspnet.to_json()), indent=2))

    print("save json model done...")


if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="SegUNet LIP dataset")
    parser.add_argument("--name",
                        help="name of experiment",
                        default='PSPNET'
                        )
    parser.add_argument("--dataset",
                        help="daase",
                        default='dir'
                        )
    parser.add_argument("--load",
                        help="ckpt",
                        default=None
                        )
    parser.add_argument("--trainimg_dir",
                        help="train image dir path")
    parser.add_argument("--trainmsk_dir",
                        help="train mask dir path")
    parser.add_argument("--valimg_dir",
                        help="val image dir path")
    parser.add_argument("--valmsk_dir",
                        help="val mask dir path")
    parser.add_argument("--batch_size",
                        default=2,
                        type=int,
                        help="batch size")
    parser.add_argument("--epochs",
                        default=30,
                        type=int,
                        help="number of epoch")
    parser.add_argument("--steps",
                        default=3000,
                        type=int,
                        help="number of epoch step")
    parser.add_argument("--val_steps",
                        default=500,
                        type=int,
                        help="number of valdation step")
    parser.add_argument("--n_labels",
                        default=2,
                        type=int,
                        help="Number of label")
    parser.add_argument("--input_shape",
                        default=(512, 512, 3),
                        help="Input images shape")
    parser.add_argument("--output_stride",
                        default=16,
                        type=int,
                        help="output_stride")
    parser.add_argument("--output_mode",
                        default="sigmoid",
                        type=str,
                        help="output mode")
    parser.add_argument("--upsample_type",
                        default="deconv",
                        type=str,
                        help="upsampling type")
    parser.add_argument("--loss",
                        default="binary_crossentropy",
                        type=str,
                        help="loss function")
    parser.add_argument("--optimizer",
                        default="adadelta",
                        type=str,
                        help="oprimizer")
    parser.add_argument("--gpu_num",
                        default="0",
                        type=str,
                        help="num of gpu")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    main(args)


# python train_mask.py --trainimg_dir /Users/nomanshafqat/val/images/ --trainmsk_dir /Users/nomanshafqat/val/annotation-small.json --valimg_dir /Users/nomanshafqat/val/images/ --valmsk_dir=/Users/nomanshafqat/val/annotation-small.json --steps=3
