#coding:utf-8
from cvtron.modeling.segmentor import api
from cvtron.utils.config_loader import MODEL_ZOO_PATH

imageSegmentor = api.get_segmentor()
pred = imageSegmentor.segment('tests/21.jpg')
