#coding:utf-8
import sys
import os 
import tensorflow as tf 
import tensorlayer as tl 
import numpy as np 
from cvtron.modeling.base.singleton import singleton

@singleton
class ImageUpsampler(object):
