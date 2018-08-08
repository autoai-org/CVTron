#coding:utf-8
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from object_detection.utils import label_map_util

from cvtron.Base.decorator import singleton
from cvtron.utils.logger.Logger import logger
from cvtron.utils.image_loader import get_image_size
from cvtron.utils.image_loader import load_image_into_numpy_array
from cvtron.thirdparty.object_detection import exporter
from cvtron.thirdparty.object_detection.protos import pipeline_pb2
from google.protobuf import text_format


@singleton
class SlimObjectDetector(object):
    def __init__(self):
        self.hasInitialized = False
        

    def set_label_map(self, label_map_file):
        self.label_map = label_map_file

    def init(self, model_path):
        if not self.hasInitialized:
            self._init_model_(model_path)
        else:
            logger.info('model has initialized, skipping')

    def _init_model_(self, model_path):
        logger.info('initiating model')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph = detection_graph
        logger.info('model initialized')

    def format_output(self, results, im_width, im_height, threshold):
        label_map = label_map_util.load_labelmap(self.label_map)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        vis_results = []
        for i in range(len(results)):
            output_dict = results[i]
            boxes = output_dict['detection_boxes']
            classes = output_dict['detection_classes']
            scores = output_dict['detection_scores']
            for j in range(boxes.shape[0]):
                if scores[i] > threshold:
                    result = {}
                    ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                    result['x_min'] = xmin * im_width
                    result['x_max'] = xmax * im_width
                    result['y_min'] = ymin * im_height
                    result['y_max'] = ymax * im_height
                    result['score'] = float(scores[i])
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    result['class_name'] = str(class_name)
                    if result not in vis_results:
                        vis_results.append(result)
        return vis_results

    def export_latest_ckpt(self, train_dir, output_dir):
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        pipeline_config_path = os.path.join(train_dir, 'pipeline.config')

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
        trained_checkpoint_prefix = tf.train.latest_checkpoint(train_dir)
        exporter.export_inference_graph(input_type, pipeline_config,
                                    trained_checkpoint_prefix,
                                    output_dir, input_shape)       

    def detect(self, img_file, threshold = 0.7):
        # read image
        image_np = load_image_into_numpy_array(img_file)
        (im_width, im_height) = get_image_size(img_file)
        # Inference process
        with self.graph.as_default():
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            with tf.Session(config=sess_config) as sess:
                ## Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs
                }
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                ## Run inference
                output_dict = sess.run(tensor_dict, feed_dict={
                    image_tensor: np.expand_dims(image_np, 0)
                })
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                results = [output_dict]
                return self.format_output(results, im_width, im_height, threshold)