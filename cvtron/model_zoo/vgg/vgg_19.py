# coding:utf-8
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Conv2dLayer, DenseLayer, FlattenLayer,
                                InputLayer, PoolLayer)

from cvtron.Base.Model import Model

from .constant import VGG_MEAN


class VGG19(Model):
    def __init__(self):
        self.name = 'vgg19'

    def _build_arch(self, net_in):
        network = Conv2dLayer(net_in, act=tf.nn.relu,
                              shape=[3, 3, 3, 64], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv1_1')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 64, 64], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv1_2')
        network = PoolLayer(network, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME',
                            pool=tf.nn.max_pool, name=self.name+'_pool1')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 64, 128], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv2_1')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 128, 128], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv2_2')
        network = PoolLayer(network, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME',
                            pool=tf.nn.max_pool, name=self.name+'_pool2')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 128, 256], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv3_1')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 256, 256], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv3_2')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 256, 256], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv3_3')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 256, 256], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv3_4')
        network = PoolLayer(network, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME',
                            pool=tf.nn.max_pool, name=self.name+'_pool3')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 256, 512], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv4_1')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv4_2')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv4_3')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv4_4')
        network = PoolLayer(network, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME',
                            pool=tf.nn.max_pool, name=self.name+'_pool4')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv5_1')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv5_2')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv5_3')
        network = Conv2dLayer(network, act=tf.nn.relu,
                              shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                              padding='SAME', name=self.name+'_conv5_4')
        network = PoolLayer(network, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME',
                            pool=tf.nn.max_pool, name=self.name+'_pool5')
        network = FlattenLayer(network, name=self.name+'_flatten')
        network = DenseLayer(network, n_units=4096,
                             act=tf.nn.relu, name=self.name+'_fc6')
        network = DenseLayer(network, n_units=4096,
                             act=tf.nn.relu, name=self.name+'_fc7')
        network = DenseLayer(network, n_units=1000,
                             act=tf.identity, name=self.name+'_fc8')
        return network

    def inference(self, x):
        """
        Build the network architecture and output the result
        Parameters
        -----------
        x: rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
        """
        rgb_scaled = x * 255.0
        red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        net_in = InputLayer(bgr, name=self.name+'_input')
        out = self._build_arch(net_in)
        return out
