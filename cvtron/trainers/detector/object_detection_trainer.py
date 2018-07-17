import os
import io
import sys
import json
import hashlib
import random
import PIL.Image
import traceback
import functools
import tensorflow as tf
from object_detection.builders import dataset_builder
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection import trainer, evaluator
from object_detection.utils import config_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from cvtron.trainers.base.BaseTrainer import BaseTrainer
from cvtron.utils.logger.Logger import logger

class ObjectDetectionTrainer(BaseTrainer):
    def __init__(self, config = None, local_path = None):
        self.config = config
        self.local_path = local_path
    
    def _get_num_classes(self, annotation_file):
        category = []
        with open(annotation_file) as f:
            annotation_data = json.load(f)
            img_num = len(annotation_data)
            for i in range(img_num):
                obj_num = len(annotation_data[i]['boundbox'])
                for j in range(obj_num):
                    label = annotation_data[i]['boundbox'][j]['label']
                    if label not in category:
                        category.append(label)
        return len(category)

    def _create_tf_data(self, annotation_file, ratio=0.7):
        category = []
        obj_list = []

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
        label_map_file = os.path.join(self.local_path, 'label_map.pbtxt')
        with open(label_map_file, mode='w') as f:
            offset = 1
            for idx in range(len(category)):
                f.write('item { \n  id: %d\n  name: \'%s\'\n}\n\n' % (idx + offset, category[idx]))

        random.seed(42)
        random.shuffle(obj_list)
        num_train = int(ratio * len(obj_list))
        train_examples = obj_list[:num_train]
        val_examples = obj_list[num_train:]
        label_map_dict = label_map_util.get_label_map_dict(label_map_file)
        train_output_path = os.path.join(self.local_path, 'train.record')
        val_output_path = os.path.join(self.local_path, 'val.record')

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

    def create_tf_record(self, output_file, annotation_data, label_map_dict, examples):
        writer = tf.python_io.TFRecordWriter(output_file)
        for idx in range(len(examples)):
            idx_tuple = examples[idx]
            try:
                tf_example = self._create_tf_example(idx_tuple, annotation_data, label_map_dict)
                writer.write(tf_example.SerializeToString())
            except ValueError:
                logger.error('Invalid example, ignoring.')
                traceback.print_exc(file=sys.stdout)
        writer.close()

    def _create_tf_example(self, idx_tuple, annotation_data, label_map_dict):
        img_idx = idx_tuple[0]
        obj_idx_local = idx_tuple[1]
        img_path = os.path.join(self.local_path, annotation_data[img_idx]['folder'])
        img_path = os.path.join(img_path, annotation_data[img_idx]['filename'])
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
        width, height = image.size
        # width = int(annotation_data[img_idx]['size']['width'])
        # height = int(annotation_data[img_idx]['size']['height'])
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

    def parse_dataset(self, annotation_file, ratio=0.7):
        self.annotation_file = annotation_file
        self._create_tf_data(annotation_file, ratio)

    def set_annotation(self, annotation_file):
        self.annotation_file = annotation_file

    def get_next(self, config):
        return dataset_util.make_initializable_iterator(
            dataset_builder.build(config)).get_next()        

    def start(self):
        if self.config is None:
            logger.error('No Config Found')
            return
        train_pipeline_file = self.config['pipeline_config_file']
        configs = self._get_configs_from_pipeline_file(train_pipeline_file)
        model_config = configs['model']
        train_config = configs['train_config']
        input_config = configs['train_input_config']
        logger.info('Building Model')
        model_fn = functools.partial(
            model_builder.build,
            model_config = model_config,
            is_training=True
        )
        logger.info('creating input dict')
        create_input_dict_fn = functools.partial(self.get_next, input_config)
        ps_tasks = 0
        worker_replicas = 1
        worker_job_name = 'obj_detection_trainer'
        task = 0
        is_chief = True
        master = ''
        num_clones = 1
        clone_on_cpu = False
        try:
            logger.info('Training Started')
            trainer.train(create_input_dict_fn, model_fn, train_config, master, task,
                            num_clones, worker_replicas, clone_on_cpu, ps_tasks,
                            worker_job_name, is_chief, self.config)
        except:
            logger.error('Cannot Start Training')
            traceback.print_exc(file=sys.stdout)

    def evaluate(self, eval_pipeline_file, model_dir, eval_dir):
        configs = self._get_configs_from_pipeline_file(eval_pipeline_file)
        model_config = configs['model']
        eval_config = configs['eval_config']
        input_config = configs['eval_input_config']
        model_fn = functools.partial(
            model_builder.build,
            model_config=model_config,
            is_training=True)
        create_input_dict_fn = functools.partial(self.get_next, input_config)
        label_map = label_map_util.load_labelmap(input_config.label_map_path)
        max_num_classes = max([item.id for item in label_map.item])
        categories = label_map_util.convert_label_map_to_categories(
                        label_map, max_num_classes)
        evaluator.evaluate(create_input_dict_fn, model_fn, eval_config, categories,
                        model_dir, eval_dir)

    def _get_configs_from_pipeline_file(self, pipeline_config_path):
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.gfile.GFile(pipeline_config_path, "r") as f:
            proto_str = f.read()
            proto_str = proto_str.replace('PATH_TO_BE_CONFIGURED', self.local_path)
            text_format.Merge(proto_str, pipeline_config)
        configs = {}
        configs["model"] = pipeline_config.model
        configs["train_config"] = pipeline_config.train_config
        configs["train_input_config"] = pipeline_config.train_input_reader
        configs["eval_config"] = pipeline_config.eval_config
        configs["eval_input_config"] = pipeline_config.eval_input_reader
        return configs

    def override_pipeline_config(self, override_config, input_config_path):
        self.override = True
        self.default_config = self.get_train_config()
        self.override_config = override_config
        new_pipeline_path = os.path.join(self.local_path, 'pipeline.config')
        content = ''
        num_classes = self._get_num_classes(self.annotation_file)
        with open(input_config_path, 'r+') as old_file:
            find_eval_input_reader = False
            for line in old_file:
                if line.find('num_classes:') != -1:
                    (key, value) = line.split(':')
                    line = line.replace(value, ' {}\n'.format(str(num_classes)))
                if line.find('batch_size:') != -1:
                    (key, value) = line.split(':')
                    line = line.replace(value, ' {}\n'.format(str(override_config['batch_size'])))
                if line.find('initial_learning_rate:') != -1:
                    (key, value) = line.split(':')
                    line = line.replace(value, ' {}\n'.format(str(override_config['learning_rate'])))
                if line.find('eval_input_reader:') != -1:
                    find_eval_input_reader = True
                # if line.find('input_path:') != -1 and (not find_eval_input_reader):
                #    (key, value) = line.split(':')
                #    line = line.replace(value, ' "{}"\n'.format(os.path.join(override_config['data_dir'], 'train.record')))
                # if line.find('input_path:') != -1 and find_eval_input_reader:
                #     (key, value) = line.split(':')
                #     line = line.replace(value, ' "{}"\n'.format(os.path.join(override_config['data_dir'], 'val.record')))
                # if line.find('label_map_path:') != -1:
                #     (key, value) = line.split(':')
                #     line = line.replace(value, ' "{}"\n'.format(os.path.join(override_config['data_dir'], 'label_map.pbtxt')))
                # if line.find('fine_tune_checkpoint:') != -1:
                #     (key, value) = line.split(':')
                #     line = line.replace(value, ' "{}"\n'.format(override_config['fine_tune_ckpt']))
                if line.find('num_steps:') != -1:
                    (key, value) = line.split(':')
                    line = line.replace(value, ' {}\n'.format(str(override_config['num_steps'])))
                content += line
        with open(new_pipeline_path, 'w') as new_file:
            new_file.writelines(content)
    
        return new_pipeline_path

    def parse_config(self):
        train_pipeline_file = self.config['pipeline_config_file']
        configs = self._get_configs_from_pipeline_file(train_pipeline_file)

    def get_train_config(self):
        return {
            'num_steps':200000
        }