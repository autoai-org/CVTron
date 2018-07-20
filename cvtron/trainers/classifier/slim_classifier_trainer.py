#coding:utf-8
import os
import math
import json
import random
from shutil import copy

import tensorflow as tf

from cvtron.thirdparty.slim.datasets import dataset_factory, dataset_utils
from cvtron.thirdparty.slim.deployment import model_deploy
from cvtron.thirdparty.slim.nets import nets_factory
from cvtron.thirdparty.slim.preprocessing import preprocessing_factory

from cvtron.utils.logger.Logger import Logger

from cvtron.wrappers.classification import learning
slim = tf.contrib.slim

RANDOM_SEED = 0

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

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

class SlimClassifierTrainer(object):

  def __init__(self, config=None, local_path=None):
    self.config = config
    self.local_path = local_path
    tf.logging.set_verbosity(tf.logging.INFO)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

  def convert_dataset(self, split_name, photo_class, class_names_to_ids, num_shards=5):
    assert split_name in ['train', 'val']

    num_per_shard = int(math.ceil(len(photo_class) / float(num_shards)))
    with tf.Graph().as_default():
      image_reader = ImageReader()

      with tf.Session('') as sess:
        for shard_id in range(num_shards):
          # output_filename = self.training_configs['dataset_params']['dataset_dir'] + '%s_%05d-of-%05d.tfrecord' % (
          #   split_name, shard_id, num_shards)
          output_filename = os.path.join(self.local_path, '%s_%05d-of-%05d.tfrecord' % (
            split_name, shard_id, num_shards))

          with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(photo_class))
            for i in range(start_ndx, end_ndx):
              tf.logging.info('>> Converting image {}/{} shard {}'.format((i + 1), len(photo_class), shard_id))

              # Read the filename:
              print(photo_class[i][0])
              image_data = tf.gfile.FastGFile(photo_class[i][0], 'rb').read()
              height, width = image_reader.read_image_dims(sess, image_data)
              print('{}, {}'.format(width, height))
              class_name = photo_class[i][1]
              class_id = class_names_to_ids[class_name]

              example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
              tfrecord_writer.write(example.SerializeToString())

  def get_filenames_and_classes(self, annotation_file):
    photo_class = []
    class_names = []
    with open(annotation_file) as f:
      annotation_data = json.load(f)
      img_num = len(annotation_data)
      for i in range(img_num):
        folder = os.path.join(self.local_path, annotation_data[i]['folder'])
        path = os.path.join(folder, annotation_data[i]['filename'])
        if not os.path.exists(path):
          continue
        cname = annotation_data[i]['object']['label']      
        photo_class.append((path, cname))
        if cname not in class_names:
          class_names.append(cname)
    return photo_class, class_names

  def create_tf_data(self, annotation_file, ratio=0.7):
    photo_class, class_names = self.get_filenames_and_classes(annotation_file)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    random.seed(RANDOM_SEED)
    random.shuffle(photo_class)
    num_training = int(len(photo_class) * ratio)
    training_files = photo_class[:num_training]
    validation_files = photo_class[num_training:]

    self.num_classes = len(class_names)
    self.splits_to_sizes = {'train': len(training_files), 'val': len(validation_files)}
    self.items_to_descriptions = {'image': 'A color image of varying size.',
                                  'label': 'A single integer between 0 and %d' % (len(class_names) - 1)}

    self.convert_dataset('train', training_files, class_names_to_ids)
    self.convert_dataset('val', validation_files, class_names_to_ids)

    # create label map
    ids_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(ids_to_class_names, self.local_path)

  def get_dataset(self, split_name, dataset_dir):
    assert split_name in ['train', 'val']

    file_pattern = os.path.join(dataset_dir, '%s_*.tfrecord' % split_name)
    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir, 'label_map.txt'):
      labels_to_names = dataset_utils.read_label_file(dataset_dir, 'label_map.txt')

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=self.splits_to_sizes[split_name],
        items_to_descriptions=self.items_to_descriptions,
        num_classes=self.num_classes,
        labels_to_names=labels_to_names)

  def _configure_learning_rate(self, num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch / self.training_configs['dataset_params']['batch_size'] *
                      self.training_configs['learning_rate_params']['num_epochs_per_decay'])
    if self.training_configs['learning_rate_params']['sync_replicas']:
      decay_steps /= self.training_configs['learning_rate_params']['replicas_to_aggregate']

    if self.training_configs['learning_rate_params']['learning_rate_decay_type'] == 'exponential':
      return tf.train.exponential_decay(self.training_configs['learning_rate_params']['learning_rate'],
                                        global_step,
                                        decay_steps,
                                        self.training_configs['learning_rate_params']['learning_rate_decay_factor'],
                                        staircase=True,
                                        name='exponential_decay_learning_rate')
    elif self.training_configs['learning_rate_params']['learning_rate_decay_type'] == 'fixed':
      return tf.constant(self.training_configs['learning_rate_params']['learning_rate'], name='fixed_learning_rate')
    elif self.training_configs['learning_rate_params']['learning_rate_decay_type'] == 'polynomial':
      return tf.train.polynomial_decay(self.training_configs['learning_rate_params']['learning_rate'],
                                       global_step,
                                       decay_steps,
                                       self.training_configs['learning_rate_params']['end_learning_rate'],
                                       power=1.0,
                                       cycle=False,
                                       name='polynomial_decay_learning_rate')
    else:
      raise ValueError('learning_rate_decay_type [%s] was not recognized',
                       self.training_configs['learning_rate_params']['learning_rate_decay_type'])

  def _configure_optimizer(self, learning_rate):
    if self.training_configs['optimization_params']['optimizer'] == 'adadelta':
      optimizer = tf.train.AdadeltaOptimizer(
          learning_rate,
          rho=self.training_configs['optimization_params']['adadelta_rho'],
          epsilon=self.training_configs['optimization_params']['opt_epsilon'])
    elif self.training_configs['optimization_params']['optimizer'] == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(
          learning_rate,
          initial_accumulator_value=self.training_configs['optimization_params']['adagrad_initial_accumulator_value'])
    elif self.training_configs['optimization_params']['optimizer'] == 'adam':
      optimizer = tf.train.AdamOptimizer(
          learning_rate,
          beta1=self.training_configs['optimization_params']['adam_beta1'],
          beta2=self.training_configs['optimization_params']['adam_beta2'],
          epsilon=self.training_configs['optimization_params']['opt_epsilon'])
    elif self.training_configs['optimization_params']['optimizer'] == 'ftrl':
      optimizer = tf.train.FtrlOptimizer(
          learning_rate,
          learning_rate_power=self.training_configs['optimization_params']['ftrl_learning_rate_power'],
          initial_accumulator_value=self.training_configs['optimization_params']['ftrl_initial_accumulator_value'],
          l1_regularization_strength=self.training_configs['optimization_params']['ftrl_l1'],
          l2_regularization_strength=self.training_configs['optimization_params']['ftrl_l2'])
    elif self.training_configs['optimization_params']['optimizer'] == 'momentum':
      optimizer = tf.train.MomentumOptimizer(
          learning_rate,
          momentum=self.training_configs['optimization_params']['momentum'],
          name='Momentum')
    elif self.training_configs['optimization_params']['optimizer'] == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate,
          decay=self.training_configs['optimization_params']['rmsprop_decay'],
          momentum=self.training_configs['optimization_params']['rmsprop_momentum'],
          epsilon=self.training_configs['optimization_params']['opt_epsilon'])
    elif self.training_configs['optimization_params']['optimizer'] == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
      raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer

  def _get_init_fn(self):
    print(self.training_configs['fine_tuning_params']['checkpoint_path'])
    if self.training_configs['fine_tuning_params']['checkpoint_path'] is None:
      return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(self.training_configs['tf_configs']['train_dir']):
      tf.logging.info(
          'Ignoring --checkpoint_path because a checkpoint already exists in %s'
          % self.training_configs['tf_configs']['train_dir'])
      return None

    exclusions = []
    if self.training_configs['fine_tuning_params']['checkpoint_exclude_scopes']:
      exclusions = [scope.strip()
                    for scope in self.training_configs['fine_tuning_params']['checkpoint_exclude_scopes'].split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
      for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
          break
      else:
        variables_to_restore.append(var)

    if tf.gfile.IsDirectory(self.training_configs['fine_tuning_params']['checkpoint_path']):
      checkpoint_path = tf.train.latest_checkpoint(self.training_configs['fine_tuning_params']['checkpoint_path'])
    else:
      checkpoint_path = self.training_configs['fine_tuning_params']['checkpoint_path']

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=self.training_configs['fine_tuning_params']['ignore_missing_vars'])

  def _get_variables_to_train(self):
    if self.training_configs['fine_tuning_params']['trainable_scopes'] is None:
      return tf.trainable_variables()
    else:
      scopes = [scope.strip() for scope in self.training_configs['fine_tuning_params']['trainable_scopes'].split(',')]

    variables_to_train = []
    for scope in scopes:
      variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
      variables_to_train.extend(variables)
    return variables_to_train    

  def start(self):
    config = self.config
    if config is None:
      tf.logging.error('There is no input configurations.')
      return

    try:
      path1 = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
      path2 = os.path.join('wrappers/classification', 'training_configs.json')
      training_config_file = os.path.join(path1, path2) 
      with open(training_config_file) as f:
        training_configs = json.load(f)
      training_configs['tf_configs']['train_dir'] = config['train_dir']
      training_configs['tf_configs']['log_every_n_steps'] = int(config['log_every_n_steps'])
      training_configs['optimization_params']['optimizer'] = config['optimizer']
      training_configs['learning_rate_params']['learning_rate'] = float(config['learning_rate'])
      training_configs['dataset_params']['batch_size'] = int(config['batch_size'])
      training_configs['dataset_params']['model_name'] = config['pre-trained_model']
      training_configs['dataset_params']['dataset_dir'] = config['data_dir']
      training_configs['fine_tuning_params']['checkpoint_path'] = config['fine_tune_ckpt']
      if training_configs['fine_tuning_params']['checkpoint_path'] is not None:
        training_configs['fine_tuning_params']['checkpoint_exclude_scopes'] = \
        exclude_scopes_map[training_configs['dataset_params']['model_name']].format(\
        scope_map[training_configs['dataset_params']['model_name']], \
        scope_map[training_configs['dataset_params']['model_name']])
        training_configs['fine_tuning_params']['trainable_scopes'] = \
        exclude_scopes_map[training_configs['dataset_params']['model_name']].format(\
        scope_map[training_configs['dataset_params']['model_name']], \
        scope_map[training_configs['dataset_params']['model_name']])
      self.training_configs = training_configs

      with tf.Graph().as_default():  
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=training_configs['tf_configs']['num_clones'],
            clone_on_cpu=training_configs['tf_configs']['clone_on_cpu'],
            replica_id=training_configs['tf_configs']['task'],
            num_replicas=training_configs['tf_configs']['worker_replicas'],
            num_ps_tasks=training_configs['tf_configs']['num_ps_tasks'])

        # Create global_step
        with tf.device(deploy_config.variables_device()):
          global_step = slim.create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = self.get_dataset('train', self.local_path)

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            training_configs['dataset_params']['model_name'],
            num_classes=(dataset.num_classes - training_configs['dataset_params']['label_offset']),
            weight_decay=training_configs['optimization_params']['weight_decay'],
            is_training=True)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = training_configs['dataset_params']['preprocessing_name'] or training_configs['dataset_params']['model_name']
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
          provider = slim.dataset_data_provider.DatasetDataProvider(
              dataset,
              num_readers=training_configs['tf_configs']['num_readers'],
              common_queue_capacity=20 * training_configs['dataset_params']['batch_size'],
              common_queue_min=10 * training_configs['dataset_params']['batch_size'])
          [image, label] = provider.get(['image', 'label'])
          label -= training_configs['dataset_params']['label_offset']

          train_image_size = training_configs['dataset_params']['train_image_size'] or network_fn.default_image_size

          image = image_preprocessing_fn(image, train_image_size, train_image_size)

          images, labels = tf.train.batch(
              [image, label],
              batch_size=training_configs['dataset_params']['batch_size'],
              num_threads=training_configs['tf_configs']['num_preprocessing_threads'],
              capacity=5 * training_configs['dataset_params']['batch_size'])
          labels = slim.one_hot_encoding(
              labels, dataset.num_classes - training_configs['dataset_params']['label_offset'])
          batch_queue = slim.prefetch_queue.prefetch_queue(
              [images, labels], capacity=2 * deploy_config.num_clones)

        ####################
        # Define the model #
        ####################
        def clone_fn(batch_queue):
          """Allows data parallelism by creating multiple clones of network_fn."""
          images, labels = batch_queue.dequeue()
          logits, end_points = network_fn(images)

          #############################
          # Specify the loss function #
          #############################
          if 'AuxLogits' in end_points:
            slim.losses.softmax_cross_entropy(
                end_points['AuxLogits'], labels,
                label_smoothing=training_configs['learning_rate_params']['label_smoothing'], weights=0.4,
                scope='aux_loss')
          slim.losses.softmax_cross_entropy(
              logits, labels, label_smoothing=training_configs['learning_rate_params']['label_smoothing'], weights=1.0)
          return end_points

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
          x = end_points[end_point]
          summaries.add(tf.summary.histogram('activations/' + end_point, x))
          summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                          tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
          summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
          summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if training_configs['learning_rate_params']['moving_average_decay']:
          moving_average_variables = slim.get_model_variables()
          variable_averages = tf.train.ExponentialMovingAverage(
              training_configs['learning_rate_params']['moving_average_decay'], global_step)
        else:
          moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
          learning_rate = self._configure_learning_rate(dataset.num_samples, global_step)
          optimizer = self._configure_optimizer(learning_rate)
          summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if training_configs['learning_rate_params']['sync_replicas']:
          # If sync_replicas is enabled, the averaging will be done in the chief
          # queue runner.
          optimizer = tf.train.SyncReplicasOptimizer(
              opt=optimizer,
              replicas_to_aggregate=training_configs['learning_rate_params']['replicas_to_aggregate'],
              total_num_replicas=training_configs['tf_configs']['worker_replicas'],
              variable_averages=variable_averages,
              variables_to_average=moving_average_variables)
        elif training_configs['learning_rate_params']['moving_average_decay']:
          # Update ops executed locally by trainer.
          update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = self._get_variables_to_train()

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
          train_tensor = tf.identity(total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        train_dir = training_configs['tf_configs']['train_dir']
        if not os.path.exists(train_dir):
          os.makedirs(train_dir)
        weblog_dir = config['weblog_dir']
        if not os.path.exists(weblog_dir):
          os.makedirs(weblog_dir)

        logger = Logger('Training Monitor')      

        ###########################
        # Kicks off the training. #
        ###########################
        learning.train(
            train_tensor,
            logdir=train_dir,
            master=training_configs['tf_configs']['master'],
            is_chief=(training_configs['tf_configs']['task'] == 0),
            init_fn=self._get_init_fn(),
            summary_op=summary_op,
            log_every_n_steps=training_configs['tf_configs']['log_every_n_steps'],
            save_summaries_secs=training_configs['tf_configs']['save_summaries_secs'],
            save_interval_secs=training_configs['tf_configs']['save_interval_secs'],
            sync_optimizer=optimizer if training_configs['learning_rate_params']['sync_replicas'] else None,
            logger=logger,
            weblog_dir=weblog_dir)
    except:
      tf.logging.error('Unexpected error')

  def parse_dataset(self, annotation_file, ratio=0.7):    
    self.annotation_file = annotation_file
    self.create_tf_data(annotation_file, ratio)
