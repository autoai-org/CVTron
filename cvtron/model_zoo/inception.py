#coding:utf-8
import os
import time
import numpy as np 

import skimage 
import skimage.io
import skimage.transform

import tensorflow as tf 
import tensorlayer as tl 
from scipy.misc import imread, imresize

from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import (inception_v3,inception_v3_arg_scope,inception_v3_base)

slim = tf.contrib.slim

def simple_api(rgb):
    """
    Build the inception v3 Model
    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    start_time = time.time()
    print("build model started")
    rgb_scaled = rgb * 255.0
    net_in = tl.layers.InputLayer(rgb, name='input_layer')
    with slim.arg_scope(inception_v3_arg_scope()):
        network = tl.layers.SlimNetsLayer(layer=net_in, slim_layer=inception_v3,
                                    slim_args= {
                                             'num_classes' : 1001,
                                             'is_training' : False,
                                            #  'dropout_keep_prob' : 0.8,       # for training
                                            #  'min_depth' : 16,
                                            #  'depth_multiplier' : 1.0,
                                            #  'prediction_fn' : slim.softmax,
                                            #  'spatial_squeeze' : True,
                                            #  'reuse' : None,
                                            #  'scope' : 'InceptionV3'
                                            },
                                        name='InceptionV3'  # <-- the name should be the same with the ckpt model
                                        )
    print("build model finished: %fs" % (time.time() - start_time))
    return network
 
