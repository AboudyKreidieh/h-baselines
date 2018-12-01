"""Contains tests for the model abstractions and different models."""
import unittest
import tensorflow as tf

from hbaselines.models import build_linear


class TestModels(unittest.TestCase):
    """A series of tests to check the different neural network models."""

    def setUp(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5])

    def test_fcnet(self):
        pass

    def test_lstm(self):
        pass

    def test_linear_zeros(self):
        # try an empty initialization (all zeros)
        output = build_linear(
            inputs=self.input_ph,
            num_outputs=1,
            scope='test',
            reuse=False,
            nonlinearity=None,
            weights_initializer=tf.constant_initializer(0))

        self.sess.run(tf.global_variables_initializer())

        self.assertTrue(
            all(self.sess.run(output,
                              feed_dict={self.input_ph: [[0, 0, 0, 1, 0],
                                                         [1, 1, 1, 1, 1]]})
                == [[0], [0]])
        )

    def test_linear_nonzeros(self):
        # try with a non-empty initialization
        output = build_linear(
            inputs=self.input_ph,
            num_outputs=1,
            scope='test',
            reuse=False,
            nonlinearity=None,
            weights_initializer=tf.constant_initializer([[1],
                                                         [1],
                                                         [1],
                                                         [1],
                                                         [1]]),
        )

        self.sess.run(tf.global_variables_initializer())

        self.assertTrue(
            all(self.sess.run(output,
                              feed_dict={self.input_ph: [[0, 0, 0, -1, 0],
                                                         [1, 1, 1, 1, 1]]})
                == [[-1], [5]])
        )

    def test_linear_nonlinearity(self):
        # try with an non-empty initialization and a non-linearity
        output = build_linear(
            inputs=self.input_ph,
            num_outputs=1,
            scope='test',
            reuse=False,
            nonlinearity=tf.nn.relu,
            weights_initializer=tf.constant_initializer([[1],
                                                         [1],
                                                         [1],
                                                         [1],
                                                         [1]]))

        self.sess.run(tf.global_variables_initializer())

        self.assertTrue(
            all(self.sess.run(output,
                              feed_dict={self.input_ph: [[0, 0, 0, -1, 0],
                                                         [1, 1, 1, 1, 1]]})
                == [[0], [5]])
        )


if __name__ == '__main__':
    unittest.main()
