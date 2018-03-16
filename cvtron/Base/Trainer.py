#coding:utf-8


class Trainer(object):
    def __init__(self, config):
        self.config = config

    def preprocess(self):
        # Must be implemented by sub classes
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def getConfig(self):
        return self.config

    def getConfigKey(self):
        raise NotImplementedError
