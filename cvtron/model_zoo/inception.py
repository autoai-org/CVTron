#coding:utf-8
import time
import numpy as np 

import skimage.io

import tensorflow as tf 
import tensorlayer as tl 
from scipy.misc import imresize

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import (inception_v3,inception_v3_arg_scope)

slim = tf.contrib.slim

def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001):
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc

def simple_api(rgb):
    """
    Build the inception v3 Model
    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    start_time = time.time()
    print("build model started")
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
 