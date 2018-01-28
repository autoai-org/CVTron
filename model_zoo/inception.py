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

try:
    from data.imagenet_classes import *
except Exception as e:
    raise Exception("{} / download the file from: https://github.com/zsdonghao/tensorlayer/tree/master/example/data".format(e))

def print_prob(prob, limit):
    synset = class_names
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    topn = [(synset[pred[i]], prob[pred[i]]) for i in range(limit)]
    print("Top: ", topn)
    return topn
