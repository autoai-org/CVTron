# coding:utf-8
import os

from cvtron.modeling.detector.object_detector import ObjectDetector
from cvtron.utils.config_loader import MODEL_ZOO_PATH
from cvtron.utils.reporter import print_detect_result

# from object_detection.object_detection_class import ObjectDetection

train_config = {'pipeline_config_file':'/home/wujia/examples/platform/test-platform/CVTron/cvtron/object_detection/samples/configs/ssd_inception_v2_pets.config',
          'train_dir':'/home/wujia/examples/platform/test-platform/CVTron/cvtron/tests/train_dir',
          'gpu_id':1,
          'log_every_n_steps':100}

infer_config = {'model_name':'yolo'}


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

def get_object_detector(configs):
    # detector = ObjectDetection(configs)
    return None

def get_train_config():
    return train_config

def get_infer_config():
    return infer_config
