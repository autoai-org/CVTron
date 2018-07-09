import sys
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
        print(train_config)
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
        tf.logging.info('Train model')
        try:
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
            is_training=False)
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

    def parse_config(self):
        train_pipeline_file = self.config['pipeline_config_file']
        configs = self._get_configs_from_pipeline_file(train_pipeline_file)
