# coding:utf-8
import os
import sys

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from cvtron.Base.decorator import singleton
from cvtron.utils.config_loader import MODEL_ZOO_PATH


@singleton
class ImageClassifier(object):
    def __init__(self, model_name='vgg_19', model_path=MODEL_ZOO_PATH):
        self.model_name = model_name
        self.model_path = model_path
        if model_name not in ['vgg_19', 'inception_v3']:
            raise NotImplementedError
        self.download(self.model_path)
        self.sess = tf.InteractiveSession()
        if model_name == 'vgg_19':
            from cvtron.model_zoo.vgg.vgg_19 import VGG19
            vgg19 = VGG19()
            self.x = tf.placeholder("float", [None, 224, 224, 3])
            self.network = vgg19.inference(self.x)
        elif model_name == 'inception_v3':
            from cvtron.model_zoo.inception.inceptionV3 import InceptionV3
            inception_config = {
                'num_classes': 1001,
                'is_training': False
            }
            inceptionv3 = InceptionV3(inception_config)
            self.x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
            self.network = inceptionv3.inference(self.x)
        else:
            raise ValueError('Only VGG 19 and Inception V3 are allowed')
        y = self.network.outputs
        self.probs = tf.nn.softmax(y, name="prob")
        self._init_model_()

    def _init_model_(self):
        if self.model_name not in ['vgg_19', 'inception_v3']:
            raise ValueError('Only VGG 19 and Inception V3 are allowed')
        if self.model_name == 'vgg_19':
            tl.layers.initialize_global_variables(self.sess)
            npz = np.load(os.path.join(self.model_path, 'vgg19.npy'),
                          encoding='latin1').item()
            params = []
            for val in sorted(npz.items()):
                W = np.asarray(val[1][0])
                b = np.asarray(val[1][1])
                print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
                params.extend([W, b])
            tl.files.assign_params(self.sess, params, self.network)
        elif self.model_name == 'inception_v3':
            saver = tf.train.Saver()
            saver.restore(self.sess, os.path.join(
                self.model_path, 'inception_v3.ckpt'))

    def classify(self, img_file):
        from cvtron.utils.image_loader import load_image
        if self.model_name == 'vgg_19':
            image = load_image(img_file, 224, 224)
            image = image.reshape((1, 224, 224, 3))
        elif self.model_name == 'inception_v3':
            image = load_image(img_file, 299, 299)
            image = image.reshape((1, 299, 299, 3))
        prob = self.sess.run(self.probs, feed_dict={self.x: image})
        return prob

    def download(self, path):
        if self.model_name == 'vgg_19':
            from cvtron.utils.download_utils import download_vgg_19
            download_vgg_19(path=path)
        elif self.model_name == 'inception_v3':
            from cvtron.utils.download_utils import download_inception_v3
            download_inception_v3(path=path)
