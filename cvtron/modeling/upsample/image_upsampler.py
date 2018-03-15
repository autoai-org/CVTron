#coding:utf-8
import os
import sys

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from cvtron.model_zoo.lapsrn.lapsrn import LapSRN
from cvtron.modeling.base.singleton import singleton
from cvtron.utils.config_loader import MODEL_ZOO_PATH


@singleton
class ImageUpsampler(object):
    def __init__(self,model_name='laplacian', model_path=MODEL_ZOO_PATH):
        self.model_name = model_name
        self.model_path = model_path
        if model_path not in ['laplacian']:
            raise ValueError('Only laplacian network is supported yet')
        self.download(self.model_path)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        if model_name=='laplacian':
            self.x = tf.placeholder('float32', [None,size[0],size[1],size[2]], name='input_image')
            net_g, _, _, _ = LapSRN(self.x, is_train=False, reuse=False)
            tl.layers.initialize_global_variables(self.sess)
            tl.files.load_and_assign_npz(sess=self.sess, name=model_path+'/lapsrn.npz', network=net_g)

    def download(self,path):
        pass
