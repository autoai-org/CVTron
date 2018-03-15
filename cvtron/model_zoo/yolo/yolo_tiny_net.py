# coding:utf-8
import re

import numpy as np
import tensorflow as tf

from cvtron.model_zoo.yolo.net import Net


class YoloTinyNet(Net):

    def __init__(self, common_params, net_params, test=False):
        """
        common params: a params dict
        net_params   : a params dict
        """
        super(YoloTinyNet, self).__init__(common_params, net_params,isTest=test)
        # process params
        self.image_size = int(common_params['image_size'])
        self.num_classes = int(common_params['num_classes'])
        self.cell_size = int(net_params['cell_size'])
        self.boxes_per_cell = int(net_params['boxes_per_cell'])
        self.batch_size = int(common_params['batch_size'])
        self.weight_decay = float(net_params['weight_decay'])

        if not test:
            self.object_scale = float(net_params['object_scale'])
            self.noobject_scale = float(net_params['noobject_scale'])
            self.class_scale = float(net_params['class_scale'])
            self.coord_scale = float(net_params['coord_scale'])

    def inference(self, images):
        """Build the yolo model
        Args:
          images:  4-D tensor [batch_size, image_height, image_width, channels]
        Returns:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        """
        conv_num = 1

        temp_conv = self.conv2d('conv' + str(conv_num),
                                images, [3, 3, 3, 16], stride=1)
        conv_num += 1

        temp_pool = self.max_pool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num),
                                temp_pool, [3, 3, 16, 32], stride=1)
        conv_num += 1

        temp_pool = self.max_pool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num),
                                temp_pool, [3, 3, 32, 64], stride=1)
        conv_num += 1

        temp_conv = self.max_pool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num),
                                temp_conv, [3, 3, 64, 128], stride=1)
        conv_num += 1

        temp_conv = self.max_pool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num),
                                temp_conv, [3, 3, 128, 256], stride=1)
        conv_num += 1

        temp_conv = self.max_pool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num),
                                temp_conv, [3, 3, 256, 512], stride=1)
        conv_num += 1

        temp_conv = self.max_pool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num),
                                temp_conv, [3, 3, 512, 1024], stride=1)
        conv_num += 1

        temp_conv = self.conv2d('conv' + str(conv_num),
                                temp_conv, [3, 3, 1024, 1024], stride=1)
        conv_num += 1

        temp_conv = self.conv2d('conv' + str(conv_num),
                                temp_conv, [3, 3, 1024, 1024], stride=1)
        conv_num += 1

        temp_conv = tf.transpose(temp_conv, (0, 3, 1, 2))

        # Fully connected layer
        local1 = self.local('local1', temp_conv,
                            self.cell_size * self.cell_size * 1024, 256)

        local2 = self.local('local2', local1, 256, 4096)

        local3 = self.local('local3', local2, 4096, self.cell_size * self.cell_size * (
            self.num_classes + self.boxes_per_cell * 5), leaky=False, pretrain=False, train=True)

        n1 = self.cell_size * self.cell_size * self.num_classes

        n2 = n1 + self.cell_size * self.cell_size * self.boxes_per_cell

        class_probs = tf.reshape(
            local3[:, 0:n1], (-1, self.cell_size, self.cell_size, self.num_classes))
        scales = tf.reshape(
            local3[:, n1:n2], (-1, self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = tf.reshape(
            local3[:, n2:], (-1, self.cell_size, self.cell_size, self.boxes_per_cell * 4))

        local3 = tf.concat([class_probs, scales, boxes], 3)

        predicts = local3

        return predicts
