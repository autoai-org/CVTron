from cvtron.modeling.segmentor.ImageSegmentor import ImageSegmentor
from cvtron.trainers.segmentor.deeplab_trainer import DeepLabTrainer
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
    'log_folder':'/home/sfermi/Documents/Programming/project/web/cvtron-serve/cvtron-serve/static/log',
    'log_per_step':1,
    'train_steps':100,
    'eval_steps':100,
}


def get_segmentor(model_name='deeplabv3',
                  model_path=MODEL_ZOO_PATH):
    if model_name not in ['deeplabv3']:
        raise NotImplementedError
    imageSegmentor = ImageSegmentor(MODEL_ZOO_PATH, config)
    imageSegmentor._init_model_()
    return imageSegmentor

def get_segmentor_trainer(config):
    dlt = DeepLabTrainer(config)
    return dlt

def get_defaultConfig():
    print(config)
    return config
