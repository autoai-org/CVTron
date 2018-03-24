import unittest

from cvtron.utils.logger.Logger import Logger


class TestLogger(unittest.TestCase):
    def test_info(self):
        lg = Logger()
        lg.info('hello')
    def test_error(self):
        lg = Logger()
        lg.error('hello')
    def test_warn(self):
        lg = Logger()
        lg.error('hello')

if __name__ == '__main__':
    unittest.main()
