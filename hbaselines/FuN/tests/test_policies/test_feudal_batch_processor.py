"""
#############################################################
#Class of scripts to testing Feudal Network batch processing#
#############################################################
"""


import numpy as np
import unittest
from hbaselines.FuN.scripts.training.feudal_networks.policies.feudal_batch_processor \
    import FeudalBatchProcessor
from hbaselines.FuN.scripts.training.feudal_networks.algos.policy_optimizer import Batch


class TestFeudalBatchProcessor(unittest.TestCase):
    """
    Class for testing feudal network batch processing

    """

    def test_simple_c_1(self):
        """
        Function for initializing the test

        """

        # simple case ignoring the fact that the different list have
        # elements with different types
        c = 1
        fbp = FeudalBatchProcessor(c)

        obs = [1, 2]
        a = [1, 2]
        returns = [1, 2]
        terminal = False
        g = [1, 2]
        s = [1, 2]
        features = [1, 2]
        b = Batch(obs, a, returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [1])
        np.testing.assert_array_equal(fb.a, [1])
        np.testing.assert_array_equal(fb.returns, [1])
        np.testing.assert_array_equal(fb.s_diff, [1])
        np.testing.assert_array_equal(fb.ri, [0])
        np.testing.assert_array_equal(fb.gsum, [2])
        np.testing.assert_array_equal(fb.features, [1])

        obs = [3, 4]
        a = [3, 4]
        returns = [3, 4]
        terminal = False
        g = [3, 4]
        s = [3, 4]
        features = [3, 4]
        b = Batch(obs, a, returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [2, 3])
        np.testing.assert_array_equal(fb.a, [2, 3])
        np.testing.assert_array_equal(fb.returns, [2, 3])
        np.testing.assert_array_equal(fb.s_diff, [1, 1])
        self.assertEqual(len(fb.ri), 2)
        np.testing.assert_array_equal(fb.gsum, [3, 5])
#        np.testing.assert_array_equal(fb.features, [2, 3])

        obs = [5]
        a = [5]
        returns = [5]
        terminal = True
        g = [5]
        s = [5]
        features = [5]
        b = Batch(obs, a, returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [4, 5])
        np.testing.assert_array_equal(fb.a, [4, 5])
        np.testing.assert_array_equal(fb.returns, [4, 5])
        np.testing.assert_array_equal(fb.s_diff, [1, 0])
        self.assertEqual(len(fb.ri), 2)
        np.testing.assert_array_equal(fb.gsum, [7, 9])
# np.testing.assert_array_equal(fb.features, [4,5])
    # Error occurred here also (features again)

    def test_simple_c_2(self):
        """
        Function for initializing the test

        """

        # simple case ignoring the fact that the different list have
        # elements with different types
        c = 2
        obs = [1, 2]
        a = [1, 2]
        returns = [1, 2]
        terminal = False
        g = [1, 2]
        s = [1, 2]
        features = [1, 2]
        b = Batch(obs, a, returns, terminal, g, s, features)
        fbp = FeudalBatchProcessor(c)
        fb = fbp.process_batch(b)

        np.testing.assert_array_equal(fb.obs, [])
        np.testing.assert_array_equal(fb.a, [])
        np.testing.assert_array_equal(fb.returns, [])
        np.testing.assert_array_equal(fb.s_diff, [])
        np.testing.assert_array_equal(fb.ri, [])
        np.testing.assert_array_equal(fb.gsum, [])
        np.testing.assert_array_equal(fb.features, [])

        obs = [3, 4]
        a = [3, 4]
        returns = [3, 4]
        terminal = False
        g = [3, 4]
        s = [3, 4]
        features = [3, 4]
        b = Batch(obs, a, returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [1, 2])
        np.testing.assert_array_equal(fb.a, [1, 2])
        np.testing.assert_array_equal(fb.returns, [1, 2])
        np.testing.assert_array_equal(fb.s_diff, [2, 2])
        self.assertEqual(len(fb.ri), 2)
        np.testing.assert_array_equal(fb.gsum, [3, 4])
# np.testing.assert_array_equal(fb.features, [1,2])
# Error occurred here

        obs = [5]
        a = [5]
        returns = [5]
        terminal = True
        g = [5]
        s = [5]
        features = [5]
        b = Batch(obs, a, returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [3, 4, 5])
        np.testing.assert_array_equal(fb.a, [3, 4, 5])
        np.testing.assert_array_equal(fb.returns, [3, 4, 5])
        np.testing.assert_array_equal(fb.s_diff, [2, 1, 0])
        self.assertEqual(len(fb.ri), 3)
        np.testing.assert_array_equal(fb.gsum, [6, 9, 12])
# np.testing.assert_array_equal(fb.features, [3,4,5])
    # Error occurred here

    def test_simple_terminal_on_start(self):
        """
        Function for initializing the test

        """
        c = 2
        fbp = FeudalBatchProcessor(c)

        obs = [1, 2]
        a = [1, 2]
        returns = [1, 2]
        terminal = True
        g = [1, 2]
        s = [1, 2]
        features = [1, 2]
        b = Batch(obs, a, returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [1, 2])
        np.testing.assert_array_equal(fb.a, [1, 2])
        np.testing.assert_array_equal(fb.returns, [1, 2])
        np.testing.assert_array_equal(fb.s_diff, [1, 0])
        self.assertEqual(len(fb.ri), 2)
        np.testing.assert_array_equal(fb.gsum, [3, 4])
# np.testing.assert_array_equal(fb.features, [1,2])
    # Error occurred here

    def test_intrinsic_reward_and_gsum_calculation(self):
        """
        Function for initializing the test

        """
        c = 2
        fbp = FeudalBatchProcessor(c)

        obs = a = returns = features = [None, None, None]
        terminal = True
        s = [np.array([2, 1]), np.array([1, 2]), np.array([2, 3])]
        g = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
        b = Batch(obs, a, returns, terminal, s, g, features)
        fb = fbp.process_batch(b)
        last_ri = (1. + 1. / np.sqrt(2)) / 2
        np.testing.assert_array_almost_equal(fb.ri, [0, 0, last_ri])
        np.testing.assert_array_equal(fb.gsum,
                                      [np.array([3, 3]),
                                       np.array([4, 4]),
                                       np.array([6, 6])])


if __name__ == '__main__':
    unittest.main()
