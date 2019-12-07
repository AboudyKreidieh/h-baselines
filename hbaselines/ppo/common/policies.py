import tensorflow as tf
import numpy as np

from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype


class PolicyWithValue(object):

    def __init__(self, env, obs_ph, **network_kwargs):
        self.obs_ph = obs_ph

        # Create the actor network.
        policy_latent = self._create_latent(obs_ph, 'pi', **network_kwargs)
        policy_latent = tf.layers.flatten(policy_latent)

        # Create a separate value function.
        vf_latent = self._create_latent(obs_ph, 'vf', **network_kwargs)
        vf_latent = tf.layers.flatten(vf_latent)

        # Based on the action space, will select what probability distribution
        # type
        self.pdtype = make_pdtype(env.action_space)

        pd, pi = self.pdtype.pdfromlatent(policy_latent, init_scale=0.01)
        self.pd, self.pi = pd, pi

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)

        # TODO: add duel vf
        self.vf = fc(vf_latent, 'vf', 1)
        self.vf = self.vf[:, 0]

    @staticmethod
    def _create_latent(obs_ph, scope, **network_kwargs):
        layers = network_kwargs['layers']
        act_fun = network_kwargs['act_fun']
        layer_norm = network_kwargs['layer_norm']

        with tf.compat.v1.variable_scope(scope, reuse=False):
            latent = tf.layers.flatten(obs_ph)

            # Create the hidden layers.
            for i, layer_size in enumerate(layers):
                latent = fc(
                    latent,
                    scope='fc' + str(i),
                    nh=layer_size,
                    init_scale=np.sqrt(2)
                )
                if layer_norm:
                    latent = tf.contrib.layers.layer_norm(
                        latent, center=True, scale=True)
                latent = act_fun(latent)

        return latent
