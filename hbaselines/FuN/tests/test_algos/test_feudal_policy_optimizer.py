"""
#############################################################
#Class of scripts to testing Feudal Network policy optimizer#
#############################################################
"""


# import gym
import unittest
# import tensorflow as tf
# from FuN.scripts.training.feudal_networks.algos.feudal_policy_optimizer \
#     import FeudalPolicyOptimizer


class TestFeudalPolicyOptimizer(unittest.TestCase):
    """
    Class for testing feudal policy optimizer

    """

    def test_init(self):
        """
        Function for initializing the test

        """
        # env = gym.make('OneRoundDeterministicRewardBoxObs-v0')
        # with tf.Session() as session:
        #     feudal_opt = FeudalPolicyOptimizer(env, 0, 'feudal', False)


if __name__ == '__main__':
    unittest.main()
