#coding:utf-8
import os
from cvtron.trainers.classifier_trainer import ClassifierTrainer
from cvtron.utils.config_loader import MODEL_ZOO_PATH
trainer = ClassifierTrainer(batch_size=64, 
                            epochs=100, 
                            learning_rate=1e-3,
                            log_dir='./logs',
                            dataset_dir='./cvtron/data_zoo/hico/tfrecords/train', 
                            checkpoint=MODEL_ZOO_PATH, 
                            num_classes=600, 
                            is_fine_tune=True)
trainer.bootstrap()