#coding:utf-8
import math
import tensorflow as tf 

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import (inception_v3,inception_v3_arg_scope)
from cvtron.data_zoo.hico import hico
from cvtron.preprocessor.inception_preprocessing import preprocess_image
slim = tf.contrib.slim

def get_init_fn(checkpoint_dir):
    checkpoint_exclude_scopes = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(os.path.join(checkpoint_dir, 'inception_v3.ckpt'),
            variables_to_restore)

class ClassifierTrainer(object):
    def __init__(self, batch_size,epochs,learning_rate,log_dir,dataset_dir,checkpoint, num_classes, is_fine_tune=True):
        if not is_fine_tune:
            raise ValueError('Only Fine Tune is supported yet')
        #TODO: use inception.default image size instead
        self.image_size = 299
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.dataset_dir = dataset_dir
        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.is_fine_tune = is_fine_tune

    def bootstrap(self):
        if not tf.gfile.Exists(self.log_dir):
            tf.gfile.MakeDirs(self.log_dir)
        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.DEBUG)
            dataset = hico.get_split('train', self.dataset_dir)
            data_provider = slim.dataset_data_provider.DatasetDataProvider(
                            dataset, 
                            num_readers=4,
                            common_queue_capacity=20 * self.batch_size, 
                            common_queue_min=10 * self.batch_size)
            image, label = data_provider.get(['image', 'label'])
            label = tf.decode_raw(label, tf.float32)
            label = tf.reshape(label, [self.num_classes])
            image = preprocess_image(image, self.image_size, self.image_size,
                    is_training=True)
            images, labels = tf.train.batch(
                    [image, label],
                    batch_size = self.batch_size,
                    num_threads = 1,
                    capacity = 5 * self.batch_size)        
            # Create the model
            with slim.arg_scope(inception_v3_arg_scope()):
                logits, _ = inception_v3(images, num_classes = self.num_classes, is_training=True)

            predictions = tf.nn.sigmoid(logits, name='prediction')
            
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
            loss = tf.reduce_mean(cross_entropy)

            # Add summaries
            tf.summary.scalar('loss', loss)

            # Fine-tune only the new layers
            trainable_scopes = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
            scopes = [scope.strip() for scope in trainable_scopes]
            variables_to_train = []
            for scope in scopes:
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                variables_to_train.extend(variables)
            

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            train_op = slim.learning.create_train_op(loss, optimizer, variables_to_train=variables_to_train)

            num_batches = math.ceil(data_provider.num_samples()/float(self.batch_size)) 
            num_steps = self.epochs * int(num_batches)
            slim.learning.train(
                train_op,
                logdir=FLAGS.log_dir,
                init_fn=get_init_fn(FLAGS.checkpoint),
                number_of_steps=num_steps,
                save_summaries_secs=300,
                save_interval_secs=300
            )

