#coding:utf-8
import os

import tensorflow as tf

from cvtron.model_zoo.deeplab.deeplabV3 import deeplab_v3
from cvtron.preprocessor import training
from cvtron.preprocessor.read_data import (scale_image_with_crop_padding,
                                           tf_record_parser)


class ImageSegmentor(object):
    def __init__(self, model_path):
        self.sess = tf.InteractiveSession()
        self.model_path = model_path
    def _load_tfrecords(self, test_filenames, batch_size, train_args):
        class_labels = [v for v in range((train_args['number_of_classes']+1))]
        class_labels[-1] = 255
        test_data = tf.data.TFRecordDataset(test_filenames)
        test_data = test_data.map(tf_record_parser)  # Parse the record into tensors.
        test_data = test_data.map(scale_image_with_crop_padding)
        test_data = test_data.batch(batch_size)
        iterator = test_data.make_one_shot_iterator()
        batch_images_tf, batch_labels_tf, batch_shapes_tf = iterator.get_next()
        logits_tf =  deeplab_v3(batch_images_tf, train_args, is_training=False, reuse=False)
        valid_labels_batch_tf, valid_logits_batch_tf = training.get_valid_logits_and_labels(
            annotation_batch_tensor=batch_labels_tf,
            logits_batch_tensor=logits_tf,
            class_labels=class_labels)
        cross_entropies_tf = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tf,
                                                             labels=valid_labels_batch_tf)
        cross_entropy_mean_tf = tf.reduce_mean(cross_entropies_tf)
        tf.summary.scalar('cross_entropy', cross_entropy_mean_tf)
        
    def _init_model_(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(self.model_path,'deeplabv3.ckpt'))
    def segment(self, img_file):
        pass
