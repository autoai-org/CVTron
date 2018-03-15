# coding:utf-8
import os

from cvtron.modeling.detector.object_detector import ObjectDetector
from cvtron.utils.config_loader import MODEL_ZOO_PATH
from cvtron.utils.reporter import print_detect_result


def simple_detect_api(img_file,
                      model_name='yolo_tiny',
                      model_path=MODEL_ZOO_PATH):
    if model_name not in ['ssd', 'yolo', 'yolo_tiny']:
        raise NotImplementedError
    objectDetector = ObjectDetector(model_name, model_path)
    result = objectDetector.detect(img_file)
    print_detect_result(result)
    return result


def get_detector(model_name='yolo_tiny',
                 model_path=MODEL_ZOO_PATH):
    if model_name not in ['ssd', 'yolo', 'yolo_tiny']:
        raise NotImplementedError
    objectDetector = ObjectDetector(model_name, model_path)
    return objectDetector
