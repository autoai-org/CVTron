# coding:utf-8
import os

import numpy as np
import scipy.io as spio
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.misc import imread


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class TFRecordConverter(object):
    def __init__(self,
                 base_dataset_dir_voc,
                 images_folder_name_voc,
                 annotations_folder_name_voc,
                 base_dataset_dir_aug_voc,
                 images_folder_name_aug_voc,
                 annotations_folder_name_aug_voc):
        self.base_dataset_dir_voc = base_dataset_dir_voc
        self.images_folder_name_voc = images_folder_name_voc
        self.annotations_folder_name_voc = annotations_folder_name_voc
        self.base_dataset_dir_aug_voc = base_dataset_dir_aug_voc
        self.images_folder_name_aug_voc = images_folder_name_aug_voc
        self.annotations_folder_name_aug_voc = annotations_folder_name_aug_voc
        self.images_dir_voc = os.path.join(base_dataset_dir_voc, images_folder_name_voc)
        self.annotations_dir_voc = os.path.join(base_dataset_dir_voc, annotations_folder_name_voc)
        self.images_dir_aug_voc = os.path.join(base_dataset_dir_aug_voc, images_folder_name_aug_voc)
        self.annotations_dir_aug_voc = os.path.join(base_dataset_dir_aug_voc, annotations_folder_name_aug_voc)

    def get_files_list(self, filename):
        with open(filename, 'r') as f:
            images_filename_list = [line for line in f]
        return images_filename_list

    def shuffle(self, ratio, images_filename_list):
        np.random.shuffle(images_filename_list)
        val_images_filename_list = images_filename_list[:int(ratio*len(images_filename_list))]
        train_images_filename_list = images_filename_list[int(ratio*len(images_filename_list)):]
        return train_images_filename_list, val_images_filename_list
        
    def read_annotation_from_mat_file(self, annotations_dir, image_name):
        annotations_path = os.path.join(annotations_dir, (image_name.strip() + ".mat"))
        mat = spio.loadmat(annotations_path)
        img = mat['GTcls']['Segmentation'][0][0]
        return img

    def convert(self, filename_list, writer):
        for i, image_name in enumerate(filename_list):
            try:
                image_np = imread(os.path.join(self.images_dir_aug_voc, image_name.strip()+'.jpg'))
            except FileNotFoundError:
                image_np = imread(os.path.join(self.images_dir_voc, image_name.strip()+".jpg"))
            
            try:
                annotation_np = self.read_annotation_from_mat_file(self.annotations_dir_aug_voc, image_name)
            except FileNotFoundError:
                annotation_np = imread(os.path.join(self.annotations_dir_voc, image_name.strip() + ".png"))
            image_h = image_np.shape[0]
            image_w = image_np.shape[1]

            img_raw = image_np.tostring()
            annotation_raw = annotation_np.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(image_h),
                    'width': _int64_feature(image_w),
                    'image_raw': _bytes_feature(img_raw),
                    'annotation_raw': _bytes_feature(annotation_raw)}))
            writer.write(example.SerializeToString())
        print ('Total Image Written:', i)
        writer.close()
