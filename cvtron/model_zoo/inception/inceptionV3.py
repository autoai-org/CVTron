import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import (inception_v3,
                                                                   inception_v3_arg_scope)
from tensorlayer.layers import InputLayer, SlimNetsLayer

from cvtron.Base.Model import Model

slim = tf.contrib.slim


class InceptionV3(Model):
    def __init__(self, config):
        self.name = 'Google Inception V3'
        self.slim_args = config

    def _build_arch(self, net_in):
        with slim.arg_scope(inception_v3_arg_scope()):
            network = SlimNetsLayer(
                layer=net_in,
                slim_layer=inception_v3,
                slim_args=self.slim_args,
                name='InceptionV3'
            )
        return network

    def inference(self, x):
        """
        Build the network architecture and output the result
        Parameters
        -----------
        x: rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
        """
        net_in = InputLayer(x, name=self.name+'_input')
        out = self._build_arch(net_in)
        return out
