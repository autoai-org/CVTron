import os
import sys
import json
import numpy as np
import math
from skimage.draw import polygon
from PIL import Image
import tensorflow as tf
# from cvtron.thirdparty.deeplab import common
from deeplab import common
from cvtron.thirdparty.deeplab import model
from cvtron.thirdparty.deeplab.datasets import segmentation_dataset
from cvtron.thirdparty.deeplab.datasets import build_data
from cvtron.thirdparty.deeplab.utils import input_generator
from cvtron.thirdparty.deeplab.utils import train_utils
from cvtron.thirdparty.slim.deployment import model_deploy
from cvtron.utils.logger.Logger import Logger
from cvtron.wrappers.segmentation import learning

slim = tf.contrib.slim
dataset = slim.dataset
tfexample_decoder = slim.tfexample_decoder
prefetch_queue = slim.prefetch_queue

_NUM_SHARDS = 5
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

class DeepLabTrainer(object):
  def __init__(self, config=None, local_path=None):
    self.config = config
    self.local_path = local_path
    tf.logging.set_verbosity(tf.logging.INFO)

  def _create_tf_data(self, annotation_file, ratio):
    seg_mask_dir = os.path.join(self.local_path, 'mask')
    if not tf.gfile.IsDirectory(seg_mask_dir):
      tf.gfile.MakeDirs(seg_mask_dir) 

    with open(annotation_file) as f:
      annotation_data = json.load(f)

    categorty = []
    image_reader = build_data.ImageReader('jpeg', channels=3)
    img_num = len(annotation_data)
    img_num_train = int(img_num * ratio)
    num_per_shard = int(math.ceil(img_num_train / float(_NUM_SHARDS)))
    for shard_id in range(_NUM_SHARDS):
      output_filename = os.path.join(self.local_path, 
        '%s-%05d-of-%05d.tfrecord' % ('train', shard_id, _NUM_SHARDS))
      with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_idx = shard_id * num_per_shard
        end_idx = min((shard_id + 1) * num_per_shard, img_num_train)
        for img_idx in range(start_idx, end_idx):
          sys.stdout.write('\r>> Converting image %d/%d shard %d ' % (
            img_idx + 1, img_num_train, shard_id))
          sys.stdout.flush()
          folder = annotation_data[img_idx]['folder']
          filename = annotation_data[img_idx]['filename']
          image_filename = os.path.join(self.local_path, os.path.join(folder, filename))
          image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
          height, width = image_reader.read_image_dims(image_data)
          
          name, ext = os.path.splitext(os.path.basename(image_filename))
          seg_filename = os.path.join(seg_mask_dir, name + '.png')       
          seg_data = np.zeros((height, width), 'uint8')

          obj_num = len(annotation_data[img_idx]['object'])
          for obj_idx in range(obj_num):
            label = annotation_data[img_idx]['object'][obj_idx]['label']
            if label not in categorty:
              categorty.append(label)
            val = categorty.index(label) + 1
            polygon_str = annotation_data[img_idx]['object'][obj_idx]['polyline']
            poly = [int(i) for i in polygon_str.split(',')]
            poly_x = np.asarray(poly[0::2])
            poly_y = np.asarray(poly[1::2])
            rr, cc = polygon(poly_y, poly_x, seg_data.shape)
            seg_data[rr,cc] = val

          pil_image = Image.fromarray(seg_data)
          with tf.gfile.Open(seg_filename, mode='w') as f:
            pil_image.save(f, 'PNG')
          seg_data2 = tf.gfile.FastGFile(seg_filename, 'rb').read()
          # Convert to tf example.
          # example = build_data.image_seg_to_tfexample(
          #     image_data, filename, height, width, seg_data.tobytes())
          example = build_data.image_seg_to_tfexample(
              image_data, filename, height, width, seg_data2)
          tfrecord_writer.write(example.SerializeToString())

    return categorty

  def parse_dataset(self, annotation_file, ratio=1.0):
    self.annotation_file = annotation_file
    self._create_tf_data(annotation_file, ratio)

  def set_dataset_info(self, annotation_file, ratio=1.0):
    with open(annotation_file) as f:
      annotation_data = json.load(f)
    img_num = len(annotation_data)
    categorty = []
    for img_idx in range(img_num):
      obj_num = len(annotation_data[img_idx]['object'])
      for obj_idx in range(obj_num):
        label = annotation_data[img_idx]['object'][obj_idx]['label']
        if label not in categorty:
          categorty.append(label)
    
    self.splits_to_sizes = {
          'train': int(img_num * ratio),
          'val': img_num - int(img_num * ratio)
        }
    self.num_classes = len(categorty) + 1
    self.dataset_name = 'dataset'
    # self.splits_to_sizes = {
    #   'train': 1464,
    #   'val': 1449
    # }
    # self.num_classes = 21
    # self.dataset_name = 'pascal_voc_seg'

  def _get_dataset(self, split_name, dataset_dir):
    file_pattern = os.path.join(dataset_dir, '%s-*' % split_name)
    splits_to_sizes = self.splits_to_sizes
    ignore_label = 255
    num_classes = self.num_classes
    dataset_name = self.dataset_name

    # Specify how the TF-Examples are decoded.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/segmentation/class/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/segmentation/class/format': tf.FixedLenFeature(
            (), tf.string, default_value='png'),
    }
    items_to_handlers = {
        'image': tfexample_decoder.Image(
            image_key='image/encoded',
            format_key='image/format',
            channels=3),
        'image_name': tfexample_decoder.Tensor('image/filename'),
        'height': tfexample_decoder.Tensor('image/height'),
        'width': tfexample_decoder.Tensor('image/width'),
        'labels_class': tfexample_decoder.Image(
            image_key='image/segmentation/class/encoded',
            format_key='image/segmentation/class/format',
            channels=1),
    }

    decoder = tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=splits_to_sizes[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        ignore_label=ignore_label,
        num_classes=num_classes,
        name=dataset_name,
        multi_label=True)  

  def _build_deeplab(self, inputs_queue, outputs_to_num_classes, ignore_label):
    """Builds a clone of DeepLab.

    Args:
      inputs_queue: A prefetch queue for images and labels.
      outputs_to_num_classes: A map from output type to the number of classes.
        For example, for the task of semantic segmentation with 21 semantic
        classes, we would have outputs_to_num_classes['semantic'] = 21.
      ignore_label: Ignore label.

    Returns:
      A map of maps from output_type (e.g., semantic prediction) to a
        dictionary of multi-scale logits names to logits. For each output_type,
        the dictionary has keys which correspond to the scales and values which
        correspond to the logits. For example, if `scales` equals [1.0, 1.5],
        then the keys would include 'merged_logits', 'logits_1.00' and
        'logits_1.50'.
    """
    training_configs = self.training_configs

    samples = inputs_queue.dequeue()

    model_options = common.ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=training_configs['learning_params']['train_crop_size'],
        atrous_rates=training_configs['fine_tuning_params']['atrous_rates'],
        output_stride=training_configs['fine_tuning_params']['output_stride'])
    outputs_to_scales_to_logits = model.multi_scale_logits(
        samples[common.IMAGE],
        model_options=model_options,
        image_pyramid=training_configs['common']['image_pyramid'],
        weight_decay=training_configs['learning_params']['weight_decay'],
        is_training=True,
        fine_tune_batch_norm=training_configs['fine_tuning_params']['fine_tune_batch_norm'])

    for output, num_classes in outputs_to_num_classes.items():
      train_utils.add_softmax_cross_entropy_loss_for_each_scale(
          outputs_to_scales_to_logits[output],
          samples[common.LABEL],
          num_classes,
          ignore_label,
          loss_weight=1.0,
          upsample_logits=training_configs['learning_params']['upsample_logits'],
          scope=output)

    return outputs_to_scales_to_logits

  def start(self, notify_func=None, args=None):
    input_config = self.config
    if input_config is None:
      tf.logging.error('There is no input configurations.')
      return

    try:
      training_config_file = os.path.join(self.local_path, 'training_configs.json')
      print(training_config_file)
      with open(training_config_file) as f:
        self.training_configs = json.load(f)
      training_configs = self.training_configs
      training_configs['dataset_params']['dataset_dir'] = input_config['data_dir']
      training_configs['fine_tuning_params']['tf_initial_checkpoint'] = input_config['fine_tune_ckpt']
      common.FLAGS.min_resize_value = training_configs['common']['min_resize_value']
      common.FLAGS.max_resize_value = training_configs['common']['max_resize_value']
      common.FLAGS.resize_factor = training_configs['common']['resize_factor']
      common.FLAGS.logits_kernel_size = training_configs['common']['logits_kernel_size']
      common.FLAGS.model_variant = training_configs['common']['model_variant']
      common.FLAGS.image_pyramid = training_configs['common']['image_pyramid']
      common.FLAGS.add_image_level_feature = training_configs['common']['add_image_level_feature']
      common.FLAGS.aspp_with_batch_norm = training_configs['common']['aspp_with_batch_norm']
      common.FLAGS.aspp_with_separable_conv = training_configs['common']['aspp_with_separable_conv']
      common.FLAGS.multi_grid = training_configs['common']['multi_grid']
      common.FLAGS.depth_multiplier = training_configs['common']['depth_multiplier']
      common.FLAGS.decoder_output_stride = training_configs['common']['decoder_output_stride']
      common.FLAGS.decoder_use_separable_conv = training_configs['common']['decoder_use_separable_conv']
      common.FLAGS.merge_method = training_configs['common']['merge_method']
      
      # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
      config = model_deploy.DeploymentConfig(
          num_clones=training_configs['tf_configs']['num_clones'],
          clone_on_cpu=training_configs['tf_configs']['clone_on_cpu'],
          replica_id=training_configs['tf_configs']['task'],
          num_replicas=training_configs['tf_configs']['num_replicas'],
          num_ps_tasks=training_configs['tf_configs']['num_ps_tasks'])

      # Split the batch across GPUs.
      assert training_configs['learning_params']['train_batch_size'] % config.num_clones == 0, (
          'Training batch size not divisble by number of clones (GPUs).')

      clone_batch_size = int(training_configs['learning_params']['train_batch_size'] / config.num_clones)

      # Get dataset-dependent information.
      dataset = self._get_dataset(
          training_configs['dataset_params']['train_split'], 
          dataset_dir=training_configs['dataset_params']['dataset_dir'])

      train_dir = self.local_path
      training_configs['logging_configs']['train_logdir'] = train_dir

      with tf.Graph().as_default():
        with tf.device(config.inputs_device()):
          samples = input_generator.get(
              dataset,
              training_configs['learning_params']['train_crop_size'],
              clone_batch_size,
              min_resize_value=training_configs['common']['min_resize_value'],
              max_resize_value=training_configs['common']['max_resize_value'],
              resize_factor=training_configs['common']['resize_factor'],
              min_scale_factor=training_configs['fine_tuning_params']['min_scale_factor'],
              max_scale_factor=training_configs['fine_tuning_params']['max_scale_factor'],
              scale_factor_step_size=training_configs['fine_tuning_params']['scale_factor_step_size'],
              dataset_split=training_configs['dataset_params']['train_split'],
              is_training=True,
              model_variant=training_configs['common']['model_variant'])
          inputs_queue = prefetch_queue.prefetch_queue(
              samples, capacity=128 * config.num_clones)

        # Create the global step on the device storing the variables.
        with tf.device(config.variables_device()):
          global_step = tf.train.get_or_create_global_step()

          # Define the model and create clones.
          model_fn = self._build_deeplab
          model_args = (inputs_queue, {
              common.OUTPUT_TYPE: dataset.num_classes
          }, dataset.ignore_label)
          clones = model_deploy.create_clones(config, model_fn, args=model_args)

          # Gather update_ops from the first clone. These contain, for example,
          # the updates for the batch_norm variables created by model_fn.
          first_clone_scope = config.clone_scope(0)
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for model variables.
        for model_var in slim.get_model_variables():
          summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
          summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Build the optimizer based on the device specification.
        with tf.device(config.optimizer_device()):
          learning_rate = train_utils.get_model_learning_rate(
              training_configs['learning_params']['learning_policy'], 
              training_configs['learning_params']['base_learning_rate'],
              training_configs['learning_params']['learning_rate_decay_step'], 
              training_configs['learning_params']['learning_rate_decay_factor'],
              training_configs['learning_params']['training_number_of_steps'], 
              training_configs['learning_params']['learning_power'],
              training_configs['fine_tuning_params']['slow_start_step'], 
              training_configs['fine_tuning_params']['slow_start_learning_rate'])
          optimizer = tf.train.MomentumOptimizer(learning_rate, training_configs['learning_params']['momentum'])
          summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        startup_delay_steps = training_configs['tf_configs']['task'] * training_configs['tf_configs']['startup_delay_steps']
        for variable in slim.get_model_variables():
          summaries.add(tf.summary.histogram(variable.op.name, variable))

        with tf.device(config.variables_device()):
          total_loss, grads_and_vars = model_deploy.optimize_clones(
              clones, optimizer)
          total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
          summaries.add(tf.summary.scalar('total_loss', total_loss))

          # Modify the gradients for biases and last layer variables.
          last_layers = model.get_extra_layer_scopes(
            training_configs['fine_tuning_params']['last_layers_contain_logits_only'])
          grad_mult = train_utils.get_model_gradient_multipliers(
              last_layers, training_configs['learning_params']['last_layer_gradient_multiplier'])
          if grad_mult:
            grads_and_vars = slim.learning.multiply_gradients(
                grads_and_vars, grad_mult)

          # Create gradient update op.
          grad_updates = optimizer.apply_gradients(
              grads_and_vars, global_step=global_step)
          update_ops.append(grad_updates)
          update_op = tf.group(*update_ops)
          with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(
            tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries))

        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True

        weblog_dir = input_config['weblog_dir']
        if not os.path.exists(weblog_dir):
          os.makedirs(weblog_dir)
        logger = Logger('Training Monitor')

        init_fn = train_utils.get_model_init_fn(
                training_configs['logging_configs']['train_logdir'],
                training_configs['fine_tuning_params']['tf_initial_checkpoint'],
                training_configs['fine_tuning_params']['initialize_last_layer'],
                last_layers,
                ignore_missing_vars=True)

        # Start the training.
        learning.train(
            train_tensor,
            logdir=train_dir,
            log_every_n_steps=training_configs['logging_configs']['log_steps'],
            master=training_configs['tf_configs']['master'],
            number_of_steps=training_configs['learning_params']['training_number_of_steps'],
            is_chief=(training_configs['tf_configs']['task'] == 0),
            session_config=session_config,
            startup_delay_steps=startup_delay_steps,
            init_fn=init_fn,
            summary_op=summary_op,
            save_summaries_secs=training_configs['logging_configs']['save_summaries_secs'],
            save_interval_secs=training_configs['logging_configs']['save_interval_secs'],
            logger=logger,
            weblog_dir=weblog_dir,
            notify_func=notify_func,
            args=args)
    except:
      tf.logging.error('Unexpected error')
