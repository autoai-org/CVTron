from cvtron.modeling.classifier.api import *

def test_vgg():
    a=simple_classify_api("tests/tiger.jpeg")

def test_inception():
    a=simple_classify_api("tests/tiger.jpeg",'inception_v3')