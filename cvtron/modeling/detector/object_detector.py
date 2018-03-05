#coding:utf-8
import os
import sys
import tensorflow as tf 
from cvtron.utils.config_loader import MODEL_ZOO_PATH
from cvtron.modeling.base.singleton import singleton

@singleton
class ObjectDetector(object):
    def __init__(self, model_name='yolo', model_path=MODEL_ZOO_PATH):
        self.model_name = model_name
        self.model_path = model_path
        if model_name not in ['yolo','ssd']:
            raise NotImplementedError
    def download