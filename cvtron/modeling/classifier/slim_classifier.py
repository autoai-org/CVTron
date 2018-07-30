import os
import math
import json
import random
from shutil import copy

import tensorflow as tf

from cvtron.Base.decorator import singleton
from cvtron.utils.logger.Logger import logger
from cvtron.thirdparty.slim.datasets import dataset_factory, dataset_utils
from cvtron.thirdparty.slim.deployment import model_deploy
from cvtron.thirdparty.slim.nets import nets_factory
from cvtron.thirdparty.slim.preprocessing import preprocessing_factory
slim = tf.contrib.slim

scope_map = {
  'alexnet_v2': 'alexnet_v2',
  'cifarnet': 'CifarNet',
  'overfeat': 'overfeat',
  'vgg_a': 'vgg_a',
  'vgg_16': 'vgg_16',
  'vgg_19': 'vgg_19',
  'inception_v1': 'InceptionV1',
  'inception_v2': 'InceptionV2',
  'inception_v3': 'InceptionV3',
  'inception_v4': 'InceptionV4',
  'inception_resnet_v2': 'InceptionResnetV2',
  'lenet': 'LeNet',
  'resnet_v1_50': 'resnet_v1_50',
  'resnet_v1_101': 'resnet_v1_101',
  'resnet_v1_152': 'resnet_v1_152',
  'resnet_v1_200': 'resnet_v1_200',
  'resnet_v2_50': 'resnet_v2_50',
  'resnet_v2_101': 'resnet_v2_101',
  'resnet_v2_152': 'resnet_v2_152',
  'resnet_v2_200': 'resnet_v2_200'
}

exclude_scopes_map = {
  'alexnet_v2': '{}/fc7,{}/fc8',
  'cifarnet': '{}/logits',
  'overfeat': '{}/fc7,{}/fc8',
  'vgg_a': '{}/fc7,{}/fc8',
  'vgg_16': '{}/fc7,{}/fc8',
  'vgg_19': '{}/fc7,{}/fc8',
  'inception_v1': '{}/Logits,{}/AuxLogits',
  'inception_v2': '{}/Logits,{}/AuxLogits',
  'inception_v3': '{}/Logits,{}/AuxLogits',
  'inception_v4': '{}/Logits,{}/AuxLogits',
  'inception_resnet_v2': '{}/Logits,{}/AuxLogits',
  'lenet': '{}/Logits',
  'resnet_v1_50': '{}/logits',
  'resnet_v1_101': '{}/logits',
  'resnet_v1_152': '{}/logits',
  'resnet_v1_200': '{}/logits',
  'resnet_v2_50': '{}/logits',
  'resnet_v2_101': '{}/logits',
  'resnet_v2_152': '{}/logits',
  'resnet_v2_200': '{}/logits'
}

@singleton
class SlimClassifier(object):
    def __init__(self):
        self.hasInitialized = False

    def init(self, model_name, model_path):
        if not self.hasInitialized:
            self._init_model_(model_name, model_path)
        else:
            logger.info('model has been initialized, skip')

    def _init_model_(self, model_name, model_path):
        pass     

    def classify(self, img_file, model_name, model_path):
        labels_to_names = None
        if dataset_utils.has_labels(model_path, 'labels.txt'):
            labels_to_names = dataset_utils.read_label_file(model_path, 'labels.txt')
        else:
            tf.logging.error('No label map')
            return None    
        keys = list(labels_to_names.keys())

        with tf.Graph().as_default():
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                                        model_name, is_training=False)
            network_fn = nets_factory.get_network_fn(
                            model_name,
                            num_classes=len(keys),
                            is_training=False)          
            
            image_string = tf.read_file(img_file)
            image = tf.image.decode_jpeg(image_string, channels=3)      
            processed_image = image_preprocessing_fn(image, 
                              network_fn.default_image_size, network_fn.default_image_size)
            image_expanded = tf.expand_dims(processed_image, axis=0)

            logits, _ = network_fn(image_expanded)
            probabilites = tf.nn.softmax(logits)     
            predictions = tf.argmax(logits, 1)
            latest_checkpoint = tf.train.latest_checkpoint(model_path)    
            init_fn = slim.assign_from_checkpoint_fn(latest_checkpoint, 
                        slim.get_model_variables(scope_map[model_name]))              
    
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True
            with tf.Session(config=session_config) as sess:
                init_fn(sess)
                probs, pred = sess.run([probabilites, predictions])
                result =[]
                for i in range(len(probs[0])):
                    result.append({'type': labels_to_names[keys[i]], 'prob': str(probs[0][i])})
                sorted_result = sorted(result, key=lambda k: float(k['prob']), reverse=True)        
                return sorted_result
