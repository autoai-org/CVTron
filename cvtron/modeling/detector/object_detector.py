# coding:utf-8
import os
import sys

import numpy as np
import tensorflow as tf

from cvtron.Base.decorator import singleton
from cvtron.model_zoo.yolo.yolo_tiny_net import YoloTinyNet
from cvtron.utils.config_loader import MODEL_ZOO_PATH


@singleton
class ObjectDetector(object):
    def __init__(self, model_name='yolo_tiny', model_path=MODEL_ZOO_PATH):
        self.model_name = model_name
        self.model_path = model_path
        if model_name not in ['yolo', 'ssd', 'yolo_tiny']:
            raise NotImplementedError
        self.download(self.model_path)
        self.sess = tf.Session()
        if model_name == 'yolo_tiny':
            common_params = {'image_size': 448, 'num_classes': 20,
                             'batch_size': 1}
            net_params = {'cell_size': 7,
                          'boxes_per_cell': 2, 'weight_decay': 0.0005}

            self.net = YoloTinyNet(common_params, net_params, test=True)
            self.image = tf.placeholder(tf.float32, (1, 448, 448, 3))
            self.predicts = self.net.inference(self.image)

        else:
            raise NotImplementedError
        self._init_model_()

    def _init_model_(self):
        if self.model_name not in ['yolo', 'ssd', 'yolo_tiny']:
            raise NotImplementedError
        if self.model_name == 'yolo_tiny':
            saver = tf.train.Saver(self.net.trainable_collection)
            saver.restore(self.sess, os.path.join(
                self.model_path, 'yolo_tiny.ckpt'))
        else:
            raise NotImplementedError

    def detect(self, img_file):
        from cvtron.utils.image_loader import load_image
        if self.model_name == 'yolo_tiny':
            image = load_image(img_file, 448, 448)
            image = image.astype(np.float32)
            image = image / 255.0 * 2 - 1
            image = np.reshape(image, (1, 448, 448, 3))
            predict = self.sess.run(
                self.predicts, feed_dict={self.image: image})
            xmin, ymin, xmax, ymax, class_num = self.process_predicts(predict)
            result = {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'class_num': str(class_num)
            }
            return result
        else:
            raise NotImplementedError

    def process_predicts(self, predicts):
        p_classes = predicts[0, :, :, 0:20]
        C = predicts[0, :, :, 20:22]
        coordinate = predicts[0, :, :, 22:]

        p_classes = np.reshape(p_classes, (7, 7, 1, 20))
        C = np.reshape(C, (7, 7, 2, 1))

        P = C * p_classes

        # print P[5,1, 0, :]

        index = np.argmax(P)

        index = np.unravel_index(index, P.shape)

        class_num = index[3]

        coordinate = np.reshape(coordinate, (7, 7, 2, 4))

        max_coordinate = coordinate[index[0], index[1], index[2], :]

        xcenter = max_coordinate[0]
        ycenter = max_coordinate[1]
        w = max_coordinate[2]
        h = max_coordinate[3]

        xcenter = (index[1] + xcenter) * (448/7.0)
        ycenter = (index[0] + ycenter) * (448/7.0)

        w = w * 448
        h = h * 448

        xmin = xcenter - w/2.0
        ymin = ycenter - h/2.0

        xmax = xmin + w
        ymax = ymin + h

        return xmin, ymin, xmax, ymax, class_num

    def download(self, path):
        if self.model_name == 'yolo_tiny':
            from cvtron.utils.download_utils import download_yolo_tiny
            download_yolo_tiny(path=path)
