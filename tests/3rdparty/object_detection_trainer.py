# trainer test
from cvtron.trainers.detector.object_detection_trainer import ObjectDetectionTrainer

train_config = {'pipeline_config_file':'/media/sfermi/Programming/temp/test/ssd_inception_v2.config',
          'train_dir':'/media/sfermi/Programming/temp/cvtron-objdet/2017',
          'weblog_dir':'/media/sfermi/Programming/temp/test/log',
          'log_every_n_steps':100}

trainer = ObjectDetectionTrainer(train_config, '/media/sfermi/Programming/temp/cvtron-objdet/2017')

trainer.start()