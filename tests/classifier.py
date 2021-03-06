import math
import unittest

from cvtron.modeling.classifier.api import simple_classify_api


class TestClassifier(unittest.TestCase):
    def test_vgg(self):
        a=simple_classify_api("tests/tiger.jpeg")
        self.assertIsNotNone(a)
        self.assertFalse(math.isnan(a[0][0]))

    def test_inception(self):
        a=simple_classify_api("tests/tiger.jpeg",'inception_v3') 
        self.assertIsNotNone(a)
        self.assertFalse(math.isnan(a[0][0]))

if __name__ == '__main__':
    unittest.main()
