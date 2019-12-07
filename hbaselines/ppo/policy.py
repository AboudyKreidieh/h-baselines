import tensorflow as tf
import numpy as np

from hbaselines.ppo.common.policies import PolicyWithValue


class Model(object):
    """PPO policy model.

    Performs the step, value, and training operations.

    Attributes
    ----------
    sess : tf.compat.v1.Session
        the tensorflow session
    ac_ph : tf.compat.v1.placeholder
        placeholder for the actions
    adv_ph : tf.compat.v1.placeholder
        placeholder for the advantages
    ret_ph : tf.compat.v1.placeholder
        placeholder for the returns
    old_neglogpac : tf.compat.v1.placeholder
        placeholder for the negative log-probability of actions in the previous
        step policy
    old_vpred : tf.compat.v1.placeholder
        placeholder for the predicted values from the previous step policy
    learning_rate : tf.compat.v1.placeholder
        placeholder for the current learning rate
    clip_range : tf.compat.v1.placeholder
        placeholder for the current clip range to the gradients
    """
    def __init__(self,
                 sess,
                 env,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 duel_vf,
                 **network_kwargs):
        """Instantiate the model object.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the tensorflow session
        env : gym.Env
            the training environment
        ent_coef : float
            policy entropy coefficient in the optimization objective
        vf_coef : float
            value function loss coefficient in the optimization objective
        max_grad_norm : float or None
            gradient norm clipping coefficient
        duel_vf : bool
            whether to use duel value functions for the value estimates
        """
        self.sess = sess

        # Get state space and action space
        ob_space = env.observation_space
        ac_space = env.action_space

        # =================================================================== #
        # Part 1. Create the placeholders.                                    #
        # =================================================================== #

        self.ob_ph = tf.compat.v1.placeholder(
            tf.float32, [None, ob_space.shape[0]])
        self.act_ob_ph = tf.compat.v1.placeholder(
            tf.float32, [None, ob_space.shape[0]])
        self.ac_ph = tf.compat.v1.placeholder(
            tf.float32, [None, ac_space.shape[0]])
        self.adv_ph = tf.compat.v1.placeholder(tf.float32, [None])
        self.ret_ph = tf.compat.v1.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.old_neglogpac = old_neglogpac = tf.compat.v1.placeholder(
            tf.float32, [None])
        # Keep track of old critic
        self.old_vpred = old_vpred = tf.compat.v1.placeholder(
            tf.float32, [None])
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, [])
        self.clip_range = tf.compat.v1.placeholder(tf.float32, [])
        self.update_actor_ph = tf.compat.v1.placeholder(tf.float32, [])

        # =================================================================== #
        # Part 2. Create the policies.                                        #
        # =================================================================== #

        with tf.compat.v1.variable_scope('ppo_model', reuse=tf.AUTO_REUSE):
            # act_model that is used for sampling
            act_model = PolicyWithValue(
                ac_space, self.act_ob_ph, duel_vf, **network_kwargs)

            # Train model for training
            train_model = PolicyWithValue(
                ac_space, self.ob_ph, duel_vf, **network_kwargs)

        # =================================================================== #
        # Part 3. Calculate the loss.                                         #
        # =================================================================== #
        # Total loss = policy gradient loss - entropy * entropy coefficient   #
        #              + Value coefficient * value loss                       #
        # =================================================================== #

        neglogpac = train_model.pd.neglogp(self.ac_ph)

        # Calculate the entropy. Entropy is used to improve exploration by
        # limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = old_vpred + tf.clip_by_value(
            train_model.vf - old_vpred, - self.clip_range, self.clip_range)
        # Unclipped value
        vf_losses1 = tf.square(vpred - self.ret_ph)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - self.ret_ph)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(old_neglogpac - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -self.adv_ph * ratio

        pg_losses2 = -self.adv_ph * tf.clip_by_value(
            ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - old_neglogpac))
        clipfrac = tf.reduce_mean(
            tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range)))

        # Total loss
        loss = self.update_actor_ph * (pg_loss - entropy * ent_coef) \
            + vf_loss * vf_coef

        # =================================================================== #
        # Part 4. Create the parameter update procedure.                      #
        # =================================================================== #

        # 1. Get the model parameters
        params = tf.trainable_variables('ppo_model')

        # 2. Build our trainer
        self.trainer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=1e-5)

        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
        grads_and_var = list(zip(grads, var))

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy',
                           'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        self.train_model = train_model
        self.act_model = act_model

        # Create a saver object.
        self.saver = tf.compat.v1.train.Saver(params)

        # =================================================================== #
        # Part 5. Initialize all parameter.                                   #
        # =================================================================== #

        self.sess.run(tf.global_variables_initializer())

    def step(self, obs):
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
        if len(obs.shape) == 1:
            obs = np.array([obs])

        return self.sess.run(
            [self.act_model.action, self.act_model.vf, self.act_model.neglogp],
            feed_dict={self.act_model.obs_ph: obs}
        )

    def value(self, obs):
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
        if len(obs.shape) == 1:
            obs = np.array([obs])

        return self.sess.run(self.act_model.vf, {self.act_model.obs_ph: obs})

    def train(self,
              lr,
              clip_range,
              obs,
              returns,
              actions,
              values,
              neglogpacs,
              update_actor=1):
        """Perform the training operation.

        Parameters
        ----------
        lr : float
            the current learning rate
        clip_range : float
            the current clip range for the gradients
        obs : array_like
            (batch_size, obs_dim) matrix of observations
        returns : array_like
            (batch_size,) vector of returns
        actions : array_like
            (batch_size, ac_dim) matrix of actions
        values : array_like
            (batch_size,) vector of values
        neglogpacs : array_like
            TODO
        update_actor : float
            whether to update the actor policy

        Returns
        -------
        TODO
            TODO
        """
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.ob_ph: obs,
            self.ac_ph: actions,
            self.adv_ph: advs,
            self.ret_ph: returns,
            self.learning_rate: lr,
            self.clip_range: clip_range,
            self.old_neglogpac: neglogpacs,
            self.old_vpred: values,
            self.update_actor_ph: update_actor,
        }

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]

    def save(self, save_path):
        """Save the parameters of a tensorflow model.

        Parameters
        ----------
        save_path : str
            Prefix of filenames created for the checkpoint
        """
        self.saver.save(self.sess, save_path)

    def load(self, load_path):
        """Load model parameters from a checkpoint.

        Parameters
        ----------
        load_path : str
            location of the checkpoint
        """
        self.saver.restore(self.sess, load_path)
