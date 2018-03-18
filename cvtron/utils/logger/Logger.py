# coding:utf-8
import json
import os
import sys

from logbook import Logger as lg
from logbook import StreamHandler


class Logger(object):
    def __init__(self, loggerName):
        StreamHandler(sys.stdout).push_application()
        self.logger = lg(loggerName)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def fatal(self, message):
        self.logger.critical(message)

    def writeLog(self, log, filename):
        with open(filename, 'w') as f:
            json.dump(log, f)
