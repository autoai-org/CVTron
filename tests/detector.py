#coding:utf-8
import unittest

from cvtron.modeling.detector.api import simple_detect_api


class TestDetector(unittest.TestCase):
    def test_yolo_tiny(self):
        a = simple_detect_api("tests/cat.jpg")
        self.assertIsNotNone(a)

if __name__ == '__main__':
    unittest.main()
