"""
###################################################
#Class of scripts to testing Feudal Network policy#
###################################################
"""


import numpy as np
import unittest
from hbaselines.FuN.scripts.training.feudal_networks.policies.feudal_policy import \
    FeudalPolicy
import tensorflow as tf

np.set_printoptions(suppress=True, precision=6)


class TestFeudalPolicy(unittest.TestCase):
    """
    Class for testing feudal network policy

    """

    def setUp(self):
        """
        Function for initializing the test

        """

        # reset graph before each test case
        tf.reset_default_graph()

    def test_init(self):
        """
        Function for initializing the test

        """

        # global_step = tf.get_variable("global_step", [], tf.int32,
        #                               initializer=tf.constant_initializer(
        #                                   0, dtype=tf.int32),
        #                               trainable=False)
        # feudal = FeudalPolicy((80, 80, 3), 4, global_step)

    def test_fit_simple_dataset(self):
        """
        Function for initializing the test

        """

        with tf.Session() as session:
            global_step = tf.get_variable("global_step", [], tf.int32,
                                          initializer=tf.constant_initializer(
                                              0, dtype=tf.int32),
                                          trainable=False)
            obs_space = (80, 80, 3)
            act_space = 2
            lr = 1e-5
            g_dim = 256
            # worker_hid_dim = 32
            # manager_hid_dim = 256
            pi = FeudalPolicy(obs_space, act_space, global_step)

            grads = tf.gradients(pi.loss, pi.var_list)

            prints = []
            for g in grads:
                prints.append(g.op.name)
                prints.append(g)
            # grads[0] = tf.Print(grads[0],prints)
            grads, _ = tf.clip_by_global_norm(grads, 40)
            grads_and_vars = list(zip(grads, pi.var_list))
            opt = tf.train.AdamOptimizer(lr)
            train_op = opt.apply_gradients(grads_and_vars)

            # train_op = tf.train.AdamOptimizer(lr).minimize(
            # pi.loss,var_list=pi.var_list)
            session.run(tf.global_variables_initializer())

            obs = [np.zeros(obs_space), np.zeros(obs_space)]
            a = [[1, 0], [0, 1]]
            returns = [0, 1]
            s_diff = [np.ones(g_dim), np.ones(g_dim)]
            gsum = [np.zeros((1, g_dim)), np.ones((1, g_dim))]
            ri = [0, 0]

            _, features = pi.get_initial_features()
            worker_features = features[0:2]
            manager_features = features[2:]

            feed_dict = {
                pi.obs: obs,
                pi.ac: a,
                pi.r: returns,
                pi.s_diff: s_diff,
                pi.prev_g: gsum,
                pi.ri: ri,
                pi.state_in[0]: worker_features[0],
                pi.state_in[1]: worker_features[1],
                pi.state_in[2]: manager_features[0],
                pi.state_in[3]: manager_features[1]
            }

            n_updates = 1000
            verbose = True
            for i in range(n_updates):
                loss, vf, policy, _ = session.run([pi.loss,
                                                   pi.manager_vf,
                                                   pi.pi, train_op],
                                                  feed_dict=feed_dict)
                if verbose:
                    print('loss: {}\npolicy: {}\nvalue: {}\n-------'.format(
                        loss, policy, vf))

    def test_simple_manager_behavior(self):
        """
        Function for initializing the test

        """

        with tf.Session() as session:
            global_step = tf.get_variable("global_step", [], tf.int32,
                                          initializer=tf.constant_initializer(
                                              0, dtype=tf.int32),
                                          trainable=False)
            obs_space = (80, 80, 3)
            act_space = 2
            lr = 5e-4
            g_dim = 256
            # worker_hid_dim = 32
            # manager_hid_dim = 256
            pi = FeudalPolicy(obs_space, act_space, global_step)
            train_op = tf.train.AdamOptimizer(lr).minimize(pi.loss)

            worker_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            worker_vars = [v for v in worker_vars if 'worker' in v.name]
            worker_assign = tf.group(*[tf.assign(v, tf.zeros_like(v))
                                       for v in worker_vars])

            session.run(tf.global_variables_initializer())

            obs = [np.zeros(obs_space), np.zeros(obs_space)]
            a = [[1, 0], [0, 1]]
            returns = [0, 1]
            s_diff = [np.ones(g_dim), np.ones(g_dim)]
            gsum = [np.zeros((1, g_dim)), np.ones((1, g_dim))]
            ri = [0, 0]

            _, features = pi.get_initial_features()
            worker_features = features[0:2]
            manager_features = features[2:]

            feed_dict = {
                pi.obs: obs,
                pi.ac: a,
                pi.r: returns,
                pi.s_diff: s_diff,
                pi.prev_g: gsum,
                pi.ri: ri,
                pi.state_in[0]: worker_features[0],
                pi.state_in[1]: worker_features[1],
                pi.state_in[2]: manager_features[0],
                pi.state_in[3]: manager_features[1]
            }

            n_updates = 1000
            verbose = True
            for i in range(n_updates):
                loss, vf, policy, _, _ = session.run(
                    [pi.loss, pi.manager_vf, pi.pi, train_op, worker_assign],
                    feed_dict=feed_dict)
                if verbose:
                    print('loss: {}\npolicy: {}\nvalue: {}\n-------'.format(
                        loss, policy, vf))

                worker_var_values = session.run(worker_vars)
                print(worker_var_values)
                U = session.run(pi.U, feed_dict=feed_dict)
                print(U)
                input()


if __name__ == '__main__':
    unittest.main()
