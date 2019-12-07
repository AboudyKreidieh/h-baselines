import tensorflow as tf

from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.tf_util import adjust_shape


class PolicyWithValue(object):

    def __init__(self, env, observations, latent, vf_latent, sess=None):
        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution
        # type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        self.vf = fc(vf_latent, 'vf', 1)
        self.vf = self.vf[:, 0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and \
                        inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, obs, **extra_feed):
        """Compute next action(s) given the observation(s).

        Parameters
        ----------
        obs : array_like
            observation data (either single or a batch)

        Returns
        -------
        array_like
            action
        array_like
            value estimate
        array_like
            next state
        array_like
            negative log likelihood of the action under current policy
            parameters) tuple
        """
        a, v, state, neglogp = self._evaluate(
            [self.action, self.vf, self.state, self.neglogp],
            obs,
            **extra_feed
        )
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, obs, *args, **kwargs):
        """Compute value estimate(s) given the observation(s).

        Parameters
        ----------
        obs : array_like
            observation data (either single or a batch)

        Returns
        -------
        array_like
            value estimate
        """
        return self._evaluate(self.vf, obs, *args, **kwargs)
