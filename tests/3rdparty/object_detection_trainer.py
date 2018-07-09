# trainer test
from cvtron.trainers.detector.object_detection_trainer import ObjectDetectionTrainer

# train_config = {'pipeline_config_file':'/media/sfermi/Programming/temp/test/ssd_inception_v2.config',
#           'train_dir':'/media/sfermi/Programming/temp/cvtron-objdet/2017',
#           'weblog_dir':'/media/sfermi/Programming/temp/test/log',
#           'log_every_n_steps':100}

# trainer = ObjectDetectionTrainer(train_config, '/media/sfermi/Programming/temp/cvtron-objdet/ssd_v2')


# trainer.start()
'''
train_config = {'pipeline_config_file':'/media/sfermi/Programming/project/cv/cvtron/cvtron/3rdparty/object_detection/samples/configs/faster_rcnn_resnet101_pets.config',
          'train_dir':'/media/sfermi/Programming/temp/pets',
          'weblog_dir':'/media/sfermi/Programming/temp/pets/log',
          'log_every_n_steps':100,
          'fine_tune_ckpt':'/media/sfermi/Programming/temp/pets/model.ckpt',
          'data_dir':'/media/sfermi/Programming/temp/pets'}

trainer = ObjectDetectionTrainer(train_config, '/media/sfermi/Programming/temp/pets')

trainer.start()
'''
train_config = {'pipeline_config_file':'/media/sfermi/Programming/temp/test/ssd_inception_v2.config',
          'train_dir':'/media/sfermi/Programming/temp/cvtron-objdet/2017s',
          'weblog_dir':'/media/sfermi/Programming/temp/pets/log',
          'log_every_n_steps':100,
          'fine_tune_ckpt':'/media/sfermi/Programming/temp/pets/model.ckpt',
          'data_dir':'/media/sfermi/Programming/temp/cvtron-objdet/ssd_v2'}

trainer = ObjectDetectionTrainer(train_config, '/media/sfermi/Programming/temp/cvtron-objdet/ssd_v2')

trainer.start()