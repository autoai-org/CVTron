# coding:utf-8
<<<<<<< HEAD
from logbook import Logger, StreamHandler
=======
import json
import os
import sys

from logbook import Logger as lg
from logbook import StreamHandler

>>>>>>> 0b8133f63e104c13825aae3798841c45d0fe51f3

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
