import unittest

test_loader = unittest.defaultTestLoader.discover('.')
test_runner = unittest.TextTestRunner(verbosity=2)
test_runner.run(test_loader)
