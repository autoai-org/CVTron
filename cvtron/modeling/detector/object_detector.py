# coding:utf-8
import os
import sys
import tensorflow as tf
import numpy as np
from cvtron.utils.config_loader import MODEL_ZOO_PATH
from cvtron.modeling.base.singleton import singleton


@singleton
class ObjectDetector(object):
    def __init__(self, model_name='yolo_tiny', model_path=MODEL_ZOO_PATH):
        self.model_name = model_name
        self.model_path = model_path
        if model_name not in ['yolo', 'ssd', 'yolo_tiny']:
            raise NotImplementedError

    def detect(self, img_file):
        from cvtron.utils.image_loader import load_image
        if self.model_name == 'yolo_tiny':
            image = load_image(img_file, 448, 448)
            image = image.astype(np.float32)
            image = image / 255.0 * 2 - 1
            image = np.reshape(image, (1, 448, 448, 3))

        else:
            raise NotImplementedError

    def download(self, path):
        if self.model_name == 'yolo_tiny':
            from cvtron.utils.download_utils import download_yolo_tiny
            download_yolo_tiny(path=path)
