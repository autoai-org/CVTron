#coding:utf-8

from cvtron.trainers.segmentor.deeplab_trainer import DeepLabTrainer

config = {
    'batch_norm_epsilon':1e-5,
    'batch_norm_decay':0.9997,
    'number_of_classes':21,
    'l2_regularizer':0.0001,
    'starting_learning_rate':0.00001,
    'multi_grid':[1,2,4],
    'output_stride':16,
    'gpu_id':0,
    'resnet_model':'resnet_v2_152',
    'train_filename':'/home/sfermi/Documents/Programming/project/cv/tmp/train.tfrecords',
    'train_buffer_size':500,
    'batch_size':1,
    'valid_filename':'/home/sfermi/Documents/Programming/project/cv/tmp/validation.tfrecords',
    'valid_buffer_size':100,
    'log_folder':'/home/sfermi/Documents/Programming/project/cv/tmp/',
    'log_per_step':10,
    'train_steps':100,
    'eval_steps':100,
}
dlt = DeepLabTrainer(config)
dlt.train()
'''
datasetConfig = {
    'base_dataset_dir_voc':'/home/sfermi/Documents/Programming/dataset/VOC/VOC2012',
    'images_folder_name_voc':'JPEGImages/',
    'annotations_folder_name_voc':'SegmentationClass_1D/',
    'base_dataset_dir_aug_voc':'/home/sfermi/Documents/Programming/dataset/SBD/dataset',
    'images_folder_name_aug_voc':'img/',
    'annotations_folder_name_aug_voc':'cls/',
    'shuffle_ratio':0.1,
    'train_dataset_dir':'/home/sfermi/Documents/Programming/project/cv/tmp',
    'filename':'/home/sfermi/Documents/Programming/dataset/SBD/dataset/train.txt'
}

dlt.parseDataset(datasetConfig)
'''
