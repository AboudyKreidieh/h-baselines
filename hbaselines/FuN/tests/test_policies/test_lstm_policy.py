"""
#################################################
#Class of scripts to testing LSTM Network policy#
#################################################
"""


import unittest
# from FuN.scripts.training.feudal_networks.policies.lstm_policy \
#     import LSTMPolicy
# import tensorflow as tf


class TestLSTMPolicy(unittest.TestCase):
    """
    Class for testing LSTM network policy

    """

    def test_init(self):
        """
        Function for initializing the test

        """
        # global_step = tf.get_variable("global_step", [], tf.int32,
        #                               initializer=tf.constant_initializer(
        #                                   0, dtype=tf.int32),
        #                               trainable=False)
        # lstm_pi = LSTMPolicy((80, 80, 3), 4, global_step)


if __name__ == '__main__':
    unittest.main()
