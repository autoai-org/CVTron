#coding:utf-8

import functools
import hashlib
import io
import os
import json
import random
from shutil import copy
import PIL.Image
import tensorflow as tf

# from object_detection import trainer, evaluator
from cvtron.wrappers.detection import trainer
from cvtron.thirdparty.object_detection import evaluator
from cvtron.thirdparty.object_detection.builders import dataset_builder
from cvtron.thirdparty.object_detection.builders import model_builder
from cvtron.thirdparty.object_detection.utils import config_util
from cvtron.thirdparty.object_detection.utils import dataset_util
from cvtron.thirdparty.object_detection.utils import label_map_util

from google.protobuf import text_format
from cvtron.thirdparty.object_detection import exporter
from cvtron.thirdparty.object_detection.protos import pipeline_pb2

import numpy as np
from PIL import Image

class ObjectDetection(object):

  def __init__(self, config=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    self.config = config

  def create_tf_example(self, idx_tuple, annotation_data, label_map_dict):
    img_idx = idx_tuple[0]
    obj_idx_local = idx_tuple[1]

    img_path = os.path.join(annotation_data[img_idx]['folder'], annotation_data[img_idx]['filename'])
    if not os.path.exists(img_path):
      raise ValueError('Could not find image')
    with tf.gfile.GFile(img_path, 'rb') as fid:
      encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    try:
      image.verify()
    except Exception:
      raise ValueError('Corrupt JPEG')
    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(annotation_data[img_idx]['size']['width'])
    height = int(annotation_data[img_idx]['size']['height'])
    xmins = [float(annotation_data[img_idx]['boundbox'][obj_idx_local]['xmin']) / width]
    ymins = [float(annotation_data[img_idx]['boundbox'][obj_idx_local]['ymin']) / height]
    xmaxs = [float(annotation_data[img_idx]['boundbox'][obj_idx_local]['xmax']) / width]
    ymaxs = [float(annotation_data[img_idx]['boundbox'][obj_idx_local]['ymax']) / height]

    class_name = annotation_data[img_idx]['boundbox'][obj_idx_local]['label']
    classes_text = [class_name.encode('utf8')]
    classes = [label_map_dict[class_name]]
    truncated = [int(0)]
    difficult_obj = [int(0)]
    poses = ['Frontal'.encode('utf8')]

    feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
        annotation_data[img_idx]['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
        annotation_data[img_idx]['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

  def create_tf_record(self, output_file, annotation_data, label_map_dict, examples):
    writer = tf.python_io.TFRecordWriter(output_file)

    for idx in range(len(examples)):
        idx_tuple = examples[idx]

        try:
          tf_example = self.create_tf_example(idx_tuple, annotation_data, label_map_dict)
          writer.write(tf_example.SerializeToString())
        except ValueError:
          tf.logging.warning('Invalid example, ignoring.')

    writer.close()

  def create_tf_data(self, annotation_file, ratio=0.7):
    category = list()
    obj_list = list()
    with open(annotation_file) as f:
      annotation_data = json.load(f)
      img_num = len(annotation_data)
      for i in range(img_num):
        obj_num = len(annotation_data[i]['boundbox'])
        for j in range(obj_num):
          obj_idx_local = j
          img_idx = i
          obj_list.append((img_idx,  obj_idx_local))
          label = annotation_data[i]['boundbox'][j]['label']
          if label not in category:
              category.append(label)

    # create label map
    label_map_file = self.config['data_dir'] + 'label_map.pbtxt'
    with open(label_map_file, mode='w') as f:
      offset = 1
      for idx in range(len(category)):
        f.write('item { \n  id: %d\n  name: \'%s\'\n}\n\n' % (idx + offset, category[idx]))

    # shuffle samples
    random.seed(42)
    random.shuffle(obj_list)
    num_train = int(ratio * len(obj_list))
    train_examples = obj_list[:num_train]
    val_examples = obj_list[num_train:]

    label_map_dict = label_map_util.get_label_map_dict(label_map_file)
    train_output_path = os.path.join(self.config['data_dir'], 'train.record')
    val_output_path = os.path.join(self.config['data_dir'], 'val.record')

    self.create_tf_record(
        train_output_path,
        annotation_data,
        label_map_dict,
        train_examples)
    self.create_tf_record(
        val_output_path,
        annotation_data,
        label_map_dict,
        val_examples)

    return len(category)

  def create_pipeline_config(self, num_classes):
    old_pipeline_config_file = self.config['old_pipeline_config_file']
    new_pipeline_config_file = self.config['pipeline_config_dir'] + self.config['pre-trained_model'] + '.config'
    
    content = ''
    with open(old_pipeline_config_file, 'r+') as old_file:
      find_eval_input_reader = False
      for line in old_file:
        if line.find('num_classes:') != -1:
          (key, value) = line.split(':')
          line = line.replace(value, ' {}\n'.format(str(num_classes)))
        if line.find('batch_size:') != -1:
          (key, value) = line.split(':')
          line = line.replace(value, ' {}\n'.format(str(self.config['batch_size'])))
        if line.find('eval_input_reader:') != -1:
          find_eval_input_reader = True
        if line.find('input_path:') != -1 and (not find_eval_input_reader):
          (key, value) = line.split(':')
          line = line.replace(value, ' "{}"\n'.format(self.config['data_dir'] + 'train.record'))
        if line.find('input_path:') != -1 and find_eval_input_reader:
          (key, value) = line.split(':')
          line = line.replace(value, ' "{}"\n'.format(self.config['data_dir'] + 'val.record'))
        if line.find('label_map_path:') != -1:
          (key, value) = line.split(':')
          line = line.replace(value, ' "{}"\n'.format(self.config['data_dir'] + 'label_map.pbtxt'))
        if line.find('fine_tune_checkpoint:') != -1:
          (key, value) = line.split(':')
          line = line.replace(value, ' "{}"\n'.format(self.config['fine_tune_ckpt']))
        content += line
    with open(new_pipeline_config_file, 'w') as new_file:
      new_file.writelines(content)
    
    return new_pipeline_config_file

  def get_next(self, config):
    return dataset_util.make_initializable_iterator(
      dataset_builder.build(config)).get_next()

  def train(self):
    if self.config is None:
      tf.logging.error('No input configuration file.')
      return
    
    try:  
      # use only one gpu
      os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
      os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
      # from tensorflow.python.client import device_lib 
      # local_device_protos = device_lib.list_local_devices()
      # num = len([x.name for x in local_device_protos if x.device_type == 'GPU'])

      # create training data
      tf.logging.info('Generate training data')
      annotation_file = self.config['anno_file']
      num_classes = self.create_tf_data(annotation_file)
      copy(self.config['data_dir'] + 'label_map.pbtxt', self.config['train_dir']) ##

      # create(modify) pipeline config file
      train_pipeline_file = self.create_pipeline_config(num_classes)
      copy(train_pipeline_file, self.config['train_dir'])      

      tf.logging.info('Read config')
      configs = config_util.get_configs_from_pipeline_file(train_pipeline_file)
      model_config = configs['model']
      train_config = configs['train_config']
      input_config = configs['train_input_config']

      tf.logging.info('Build model')
      model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

      tf.logging.info('Create input dict')
      create_input_dict_fn = functools.partial(self.get_next, input_config)

      tf.logging.info('Set TensorFlow config')
      # env = json.loads(os.environ.get('TF_CONFIG', '{}'))
      # cluster_data = env.get('cluster', None)
      # cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
      # task_data = env.get('task', None) or {'type': 'master', 'index': 0}
      # task_info = type('TaskSpec', (object,), task_data)
      #
      # # Parameters for a single worker.
      # ps_tasks = 0
      # worker_replicas = 1
      # worker_job_name = 'lonely_worker'
      # task = 0
      # is_chief = True
      # master = ''
      #
      # if cluster_data and 'worker' in cluster_data:
      #   # Number of total worker replicas include "worker"s and the "master".
      #   worker_replicas = len(cluster_data['worker']) + 1
      # if cluster_data and 'ps' in cluster_data:
      #   ps_tasks = len(cluster_data['ps'])
      #
      # if worker_replicas > 1 and ps_tasks < 1:
      #   raise ValueError('At least 1 ps task is needed for distributed training.')
      #
      # if worker_replicas >= 1 and ps_tasks > 0:
      #   # Set up distributed training.
      #   server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
      #                            job_name=task_info.type,
      #                            task_index=task_info.index)
      #   if task_info.type == 'ps':
      #     server.join()
      #     return
      #
      #   worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
      #   task = task_info.index
      #   is_chief = (task_info.type == 'master')
      #   master = server.target      
      ps_tasks = 0
      worker_replicas = 1
      worker_job_name = 'lonely_worker'
      task = 0
      is_chief = True
      master = ''
      num_clones = 1
      clone_on_cpu = False

      tf.logging.info('Train model')
      trainer.train(create_input_dict_fn, model_fn, train_config, master, task,
                    num_clones, worker_replicas, clone_on_cpu, ps_tasks,
                    worker_job_name, is_chief, self.config)
    except:
      tf.logging.error('Unexcepted error')

  def eval(self):
    try:
      # use only one gpu
      gpu_id = 1
      os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
      os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

      tf.logging.info('Read config')
      eval_pipeline_file = os.path.join(self.config['train_dir'], self.config['pre-trained_model'] + '.config')
      configs = config_util.get_configs_from_pipeline_file(eval_pipeline_file)
      model_config = configs['model']
      eval_config = configs['eval_config']
      input_config = configs['eval_input_config']

      tf.logging.info('Build model')
      model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=False)

      tf.logging.info('Create input dict')
      create_input_dict_fn = functools.partial(self.get_next, input_config)
      label_map = label_map_util.load_labelmap(input_config.label_map_path)
      max_num_classes = max([item.id for item in label_map.item])
      categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes)

      # run_once = True
      # if run_once:
      #   eval_config.max_evals = 1

      ckpt_dir = self.config['train_dir']
      eval_dir = os.path.join(self.config['tmp_dir'], 'eval_dir')
      tf.logging.info('Run evaluation')
      evaluator.evaluate(create_input_dict_fn, model_fn, eval_config, categories,
                         ckpt_dir, eval_dir)
    except:
      tf.logging.error('Unexcepted error')

  def load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def run_inference_for_single_image(self, image, graph):
    with graph.as_default():
      with tf.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
        if 'detection_masks' in tensor_dict:
          # The following processing is only for single image
          detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
          detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
          # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
          real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
          detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
          detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
          detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              detection_masks, detection_boxes, image.shape[0], image.shape[1])
          detection_masks_reframed = tf.cast(
              tf.greater(detection_masks_reframed, 0.5), tf.uint8)
          # Follow the convention by adding back the batch dimension
          tensor_dict['detection_masks'] = tf.expand_dims(
              detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
          output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict

  def format_infer_result(self, results, im_width, im_height):
    label_map = label_map_util.load_labelmap(self.config['label_map'])
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    vis_results = list()
    min_score_thresh = 0.6
    for j in range(len(results)):
      output_dict = results[j]
      boxes = output_dict['detection_boxes']
      classes = output_dict['detection_classes']
      scores = output_dict['detection_scores']
      for i in range(boxes.shape[0]):
        if scores[i] > min_score_thresh:
          result = {}
          box = tuple(boxes[i].tolist())
          ymin, xmin, ymax, xmax = box
          result['x_min'] = xmin * im_width
          result['x_max'] = xmax * im_width
          result['y_min'] = ymin * im_height
          result['y_max'] = ymax * im_height
          if classes[i] in category_index.keys():
            class_name = category_index[classes[i]]['name']
          else:
            class_name = 'N/A'
          result['class_name'] = str(class_name)
          vis_results.append(result)
    return vis_results

  def export_inference_graph(self, ckpt_dir):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config_path = os.path.join(self.config['model_path'], self.config['model_name'] + '.config')

    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
      text_format.Merge(f.read(), pipeline_config)
    # text_format.Merge('', pipeline_config)
    # if FLAGS.input_shape:
    #   input_shape = [
    #       int(dim) if dim != '-1' else None
    #       for dim in FLAGS.input_shape.split(',')
    #   ]
    # else:
    #   input_shape = None
    input_shape = None
    input_type = 'image_tensor'
    trained_checkpoint_prefix = tf.train.latest_checkpoint(self.config['model_path'])
    output_directory = os.path.join(self.config['model_path'], 'exported')
    exporter.export_inference_graph(input_type, pipeline_config,
                                    trained_checkpoint_prefix,
                                    output_directory, input_shape)
    return output_directory

  def infer(self, image_paths):
    try:
      # use only one gpu
      gpu_id = 1
      os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
      os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

      tf.logging.info('Create model')
      ckpt_path = os.path.join(self.config['model_path'], 'exported/frozen_inference_graph.pb')
    
      if not os.path.isfile(ckpt_path):
        self.export_inference_graph(os.path.join(self.config['model_path'], 'exported'))

      detection_graph = tf.Graph()
      with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_path, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')

      detection_results = []
      for path in image_paths:
        image = Image.open(path)
        (im_width, im_height) = image.size
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np, detection_graph)
        detection_results.append(output_dict)
    except:
      tf.logging.error('Unexcepted error')
    return self.format_infer_result(detection_results, im_width, im_height)
