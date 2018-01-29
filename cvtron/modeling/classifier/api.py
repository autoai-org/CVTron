#coding:utf-8
import os
from cvtron.modeling.classifier.image_classifier import ImageClassifier
from cvtron.utils.reporter import print_prob
def simple_classify_api(img_file, 
                    model_name='vgg_19',
                    model_path=os.path.expanduser('~/.cvtron/model_zoo')):
    if model_name not in ['vgg_19','inception_v3']:
        raise ValueError('Only VGG 19 and Inception V3 are allowed')
    imageClassifier = ImageClassifier(model_name,model_path)
    prob = imageClassifier.classify(img_file)
    print_prob(prob,5)
    return prob
