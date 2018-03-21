#coding:utf-8
import os

import numpy as np
import tensorflow as tf
from scipy.misc import imread

from cvtron.model_zoo.deeplab.deeplabV3 import deeplab_v3
from cvtron.preprocessor import training
from cvtron.preprocessor.read_data import (scale_image_with_crop_padding,
                                           tf_record_parser)

slim = tf.contrib.slim

class ImageSegmentor(object):
    def __init__(self, model_path, args):
        self.sess = tf.InteractiveSession()
        self.model_path = model_path
        self.resnet_model = args['resnet_model']
        self.x = tf.placeholder("float",[None, None, None, 3])
        self.network = deeplab_v3(self.x, args, is_training=False, reuse=False)
        self.pred = tf.argmax(self.network, axis=3)
    def _init_model_(self):
        print('init model')
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(self.model_path,'deeplabv3/model.ckpt'))
    def segment(self, img_file):
        from cvtron.utils.image_loader import load_image
        from cvtron.utils.image_loader import write_image
        image = imread(img_file)
        image  = image.reshape((1, image.shape[0], image.shape[1],3))
        pred_image = self.sess.run(self.pred, feed_dict={self.x: image})

        pred_image = np.reshape(pred_image, (image.shape[1], image.shape[2] ))
        write_image(pred_image,'./test.jpg')
        return pred_image
