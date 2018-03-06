#coding:utf-8
import numpy as np
import tensorflow as tf

def print_prob(prob, limit=5):
    from cvtron.data_zoo.imagenet_classes import CLASS_NAMES
    synset = CLASS_NAMES
    num_classes = len(synset)
    pred = np.argsort(-prob)[::-1][:limit]
    prob = np.sort(-prob[0])
    topn = [(synset[pred[0][i]], -prob[i]) for i in range(limit)]
    print("Top "+str(limit)+": ", topn)
    return topn

def print_detect_result(result):
    print('xmin:'+str(result['xmin']))
    print('ymin:'+str(result['ymin']))
    print('xmax:'+str(result['xmax']))
    print('ymax:'+str(result['ymax']))
    print('class_num'+str(result['class_num']))
    return result

def report_hardware():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    devices = local_device_protos
    result = {
        'gpu':devices[1].physical_device_desc
    }
    print(result)
    return result