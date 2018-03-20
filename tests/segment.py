#coding:utf-8
from cvtron.modeling.segmentor.ImageSegmentor import ImageSegmentor
from cvtron.utils.config_loader import MODEL_ZOO_PATH

config = {
    'batch_norm_epsilon':1e-5,
    'batch_norm_decay':0.9997,
    'number_of_classes':21,
    'l2_regularizer':0.0001,
    'starting_learning_rate':0.00001,
    'multi_grid':[1,2,4],
    'output_stride':16,
    'gpu_id':0,
    'resnet_model':'resnet_v2_50',
    'train_filename':'/home/sfermi/Documents/Programming/project/cv/tmp/train.tfrecords',
    'train_buffer_size':500,
    'batch_size':1,
    'valid_filename':'/home/sfermi/Documents/Programming/project/cv/tmp/validation.tfrecords',
    'valid_buffer_size':100,
    'log_folder':'/home/sfermi/Documents/Programming/project/cv/tmp/',
    'log_per_step':1,
    'train_steps':100,
    'eval_steps':100,
}

imageSegmentor = ImageSegmentor(MODEL_ZOO_PATH, config)
imageSegmentor._init_model_()
pred = imageSegmentor.segment('tests/21.jpg')
