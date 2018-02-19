#coding:utf-8
import time
import numpy as np
import skimage
import skimage.io
import skimage.transform

import tensorflow as tf 
import tensorlayer as tl
from tensorlayer.layers import * 

from cvtron.model_zoo.constant import VGG_MEAN


def _load_image(path):
    img = skimage.io.imread(path)
    img = img/255.0 
    if not (0<=img).all() and (img<=1.0).all():
        raise ValueError('(0<=img).all() and (img<=1.0).all() expected but not satisified')
    # center crop
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

def build_network(net_in):
    network = Conv2dLayer(net_in, act = tf.nn.relu,
                shape = [3, 3, 3, 64], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv1_1')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 64, 64], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv1_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME',
                pool = tf.nn.max_pool, name ='pool1')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 64, 128], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv2_1')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 128, 128], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv2_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME',
                pool = tf.nn.max_pool, name ='pool2')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 128, 256], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv3_1')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 256, 256], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv3_2')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 256, 256], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv3_3')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 256, 256], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv3_4')
    network = PoolLayer(network, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME',
                pool = tf.nn.max_pool, name ='pool3')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 256, 512], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv4_1')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 512, 512], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv4_2')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 512, 512], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv4_3')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 512, 512], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv4_4')
    network = PoolLayer(network, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME',
                pool = tf.nn.max_pool, name ='pool4')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 512, 512], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv5_1')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 512, 512], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv5_2')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 512, 512], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv5_3')
    network = Conv2dLayer(network, act = tf.nn.relu,
                shape = [3, 3, 512, 512], strides = [1, 1, 1, 1],
                padding='SAME', name ='conv5_4')
    network = PoolLayer(network, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME',
                pool = tf.nn.max_pool, name ='pool5')
    network = FlattenLayer(network, name='flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
    network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
    return network

def VGG19(rgb):
    """
    Build the VGG 19 Model
    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    start_time = time.time()
    print("build model started")
    rgb_scaled = rgb * 255.0
    # Convert RGB to BGR
    if tf.__version__ <= '0.11':
        red, green, blue = tf.split(3, 3, rgb_scaled)
    else: # TF 1.0
        print(rgb_scaled)
        red, green, blue = tf.split(rgb_scaled, 3, 3)
    if not red.get_shape().as_list()[1:] == [224, 224, 1]:
        raise ValueError('red.get_shape().as_list()[1:] == [224, 224, 1] expected but not satisfied')
    if not green.get_shape().as_list()[1:] == [224, 224, 1]:
        raise ValueError('green.get_shape().as_list()[1:] == [224, 224, 1] expected but not satified')
    if not blue.get_shape().as_list()[1:] == [224, 224, 1]:
        raise ValueError('blue.get_shape().as_list()[1:] == [224, 224, 1] expected but not satisfied')
    if tf.__version__ <= '0.11':
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
    else:
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], axis=3)
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    net_in = InputLayer(bgr, name='input')
    network = build_network(net_in)
    print("build model finished: %fs" % (time.time() - start_time))
    return network

def simple_api(rgb):
    """
    Build the VGG 19 Model
    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    start_time = time.time()
    print("build model started")
    rgb_scaled = rgb * 255.0
    # Convert RGB to BGR
    if tf.__version__ <= '0.11':
        red, green, blue = tf.split(3, 3, rgb_scaled)
    else: # TF 1.0
        print(rgb_scaled)
        red, green, blue = tf.split(rgb_scaled, 3, 3)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    if tf.__version__ <= '0.11':
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
    else:
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], axis=3)
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    """ input layer """
    net_in = InputLayer(bgr, name='input')
    """ conv1 """
    network = build_network(net_in)
    print("build model finished: %fs" % (time.time() - start_time))
    return network