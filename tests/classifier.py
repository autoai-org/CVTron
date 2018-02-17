from cvtron.modeling.classifier.api import *
import unittest

class TestClassifier(unittest.TestCase):
    def test_vgg(self):
        a=simple_classify_api("tests/tiger.jpeg")
        self.assertIsInstance(a[0][0],float)

    def test_inception(self):
        a=simple_classify_api("tests/tiger.jpeg",'inception_v3') 
        self.assertIsInstance(a[0][0],float)

if __name__ == '__main__':
    unittest.main()