#coding:utf-8
import numpy as np
import tensorflow as tf
import tensorlayer as tl

import misc


def conv_block(input, num_out):
    with tf.variable_scope("block1"):
        conv1 = tl.layers.Conv2dLayer(x,)
