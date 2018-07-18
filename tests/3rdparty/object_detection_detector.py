#coding:utf-8
import unittest
from cvtron.modeling.detector.slim_object_detector import SlimObjectDetector

class TestDetector(unittest.TestCase):
    def test_detector(self):
        sod = SlimObjectDetector()
        sod.set_label_map('/media/sfermi/Programming/project/web/cvtron/cvtron-serve/cvtron-serve/tmp/img_d_b79cc2ba/label_map.pbtxt')
        sod.init('/media/sfermi/Programming/project/web/cvtron/cvtron-serve/cvtron-serve/tmp/img_d_b79cc2ba/frozen_inference_graph.pb')
        result = sod.detect('tests/cat.jpg')
        print(result)
if __name__ == '__main__':
    unittest.main()