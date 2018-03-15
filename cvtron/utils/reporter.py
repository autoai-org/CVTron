# coding:utf-8
import numpy as np
import tensorflow as tf


def print_prob(prob, limit=5):
    from cvtron.data_zoo.imagenet_classes import CLASS_NAMES
    synset = CLASS_NAMES
    num_classes = len(synset)
    pred = np.argsort(-prob)[::-1][:limit]
    prob = np.sort(-prob[0])
    topn = [(synset[pred[0][i]], -prob[i]) for i in range(limit)]
    print("Top " + str(limit) + ": ", topn)
    return topn


def print_detect_result(result):
    print('xmin:' + str(result['xmin']))
    print('ymin:' + str(result['ymin']))
    print('xmax:' + str(result['xmax']))
    print('ymax:' + str(result['ymax']))
    print('class_num' + str(result['class_num']))
    return result


def report_hardware():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    result = {'cpu': [], 'gpu': []}
    for each in local_device_protos:
        if each.device_type == 'CPU':
            result['cpu'].append(each.name)
        else:
            descs = each.physical_device_desc.split(',')
            device = descs[0].split(':')[1].replace(' ', '')
            name = descs[1].split(':')[1].replace(' ', '')
            compute_capability = descs[3].split(':')[1].replace(' ', '')
            device_obj = {'device': device, 'name': name, 'compute_capability': compute_capability}
            result['gpu'].append(device_obj)
    return result
