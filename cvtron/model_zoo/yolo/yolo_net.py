#coding:utf-8
import re

import numpy as np
import tensorflow as tf

from cvtron.model_zoo.yolo.net import Net


class YoloNet(Net):
    def __init__(self, common_params, net_params, test=False):
        super(YoloNet, self).__init__(common_params, net_params, isTest=test)
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
        temp_conv = self.conv2d('conv' + str(conv_num), images, [7, 7, 3, 64], stride=2)
        conv_num += 1


        temp_pool = self.max_pool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [3, 3, 64, 192], stride=1)
        conv_num += 1

        temp_pool = self.max_pool(temp_conv, [2, 2], 2)

        temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [1, 1, 192, 128], stride=1)
        conv_num += 1
        
        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1)
        conv_num += 1

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 256, 256], stride=1)
        conv_num += 1

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
        conv_num += 1

        temp_conv = self.max_pool(temp_conv, [2, 2], 2)

        for i in range(4):
            temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 512, 256], stride=1)
            conv_num += 1

            temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
            conv_num += 1

            temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 512, 512], stride=1)
            conv_num += 1

            temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
            conv_num += 1

            temp_conv = self.max_pool(temp_conv, [2, 2], 2)

        for i in range(2):
            temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 1024, 512], stride=1)
            conv_num += 1

            temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
            conv_num += 1

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
        conv_num += 1

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=2)
        conv_num += 1

        #
        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
        conv_num += 1
        
        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
        conv_num += 1


        #Fully connected layer
        local1 = self.local('local1', temp_conv, 49 * 1024, 4096)


        local1 = tf.nn.dropout(local1, keep_prob=0.5)

        local2 = self.local('local2', local1, 4096, self.cell_size * self.cell_size * ( self.num_classes + 5 * self.boxes_per_cell), leaky=False)

        local2 = tf.reshape(local2, [tf.shape(local2)[0], self.cell_size, self.cell_size, self.num_classes + 5 * self.boxes_per_cell])

        predicts = local2

        return predicts
