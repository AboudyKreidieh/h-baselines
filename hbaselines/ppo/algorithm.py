import tensorflow as tf
import random
import os
import time
import numpy as np
from collections import deque
import csv

from hbaselines.ppo.util import ensure_dir, explained_variance
from hbaselines.ppo.runner import Runner
from hbaselines.ppo.policy import Model
from hbaselines.ppo.tf_util import get_session


DEFAULT_NETWORK_KWARGS = dict(
    # The size of the Neural network for the policy
    layers=[64, 64],
    # the activation function to use in the neural network.
    act_fun=tf.nn.tanh,
    # TODO
    normalize_observations=False,
    # TODO
    layer_norm=False,
)


class PPO(object):
    """Proximal Policy Optimization algorithm.

    Attributes
    ----------
    """

    def __init__(self,
                 env,
                 eval_env=None,
                 nsteps=2048,
                 ent_coef=0.0,
                 lr=3e-4,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 gamma=0.99,
                 lam=0.95,
                 nminibatches=4,
                 noptepochs=4,
                 cliprange=0.2,
                 **network_kwargs):
        """Instantiate the algorithm object.

        Parameters
        ----------
        network : TODO
            TODO
        env : str or gym.Env
            the training environment
        eval_env : str or gym.Env
            the evaluation environment
        nsteps : int
            number of steps of the vectorized environment per update (i.e.
            batch size is nsteps * nenv where nenv is number of environment
            copies simulated in parallel)
        ent_coef : float
            policy entropy coefficient in the optimization objective
        lr : float or function
            learning rate, constant or a schedule function [0,1] -> R+ where 1
            is beginning of the training and 0 is the end of the training.
        vf_coef : float
            value function loss coefficient in the optimization objective
        max_grad_norm : float or None
            gradient norm clipping coefficient
        gamma : float
            discounting factor
        lam : float
            advantage estimation discounting factor (lambda in the paper)
        nminibatches : int
            number of training minibatches per update. For recurrent policies,
            should be smaller or equal than number of environments run in
            parallel.
        noptepochs : int
            number of training epochs per update
        cliprange : float or function
            clipping range, constant or schedule function [0,1] -> R+ where 1
            is beginning of the training and 0 is the end of the training
        load_path : str
            path to load the model from
        network_kwargs : dict
            keyword arguments to the policy / network builder. See baselines.
            common/policies.py/build_policy and arguments to a particular type
            of network. For instance, 'mlp' network architecture has arguments
            num_hidden and num_layers.
        """
        print([env,
               eval_env,
               nsteps,
               ent_coef,
               lr,
               vf_coef,
               max_grad_norm,
               gamma,
               lam,
               nminibatches,
               noptepochs,
               cliprange,
               network_kwargs])
        self.env = env
        self.eval_env = eval_env
        self.nsteps = nsteps
        self.ent_coef = ent_coef
        self.lr = self._get_function(lr)
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.cliprange = self._get_function(cliprange)
        self.network_kwargs = network_kwargs
        # self.network_kwargs.update(DEFAULT_NETWORK_KWARGS)

        # Create the tensorflow session.
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        self.sess = get_session(config)

        # Calculate the batch_size
        self.nbatch = self.nsteps
        self.nbatch_train = self.nbatch // self.nminibatches

        # Instantiate the model object (that creates act_model and train_model)
        self.policy_tf = Model(
            sess=self.sess,
            env=env,
            nbatch_train=self.nbatch_train,
            nsteps=nsteps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            network_kwargs=network_kwargs
        )

        # Instantiate the runner objects.
        self.runner = Runner(
            env=self.env,
            model=self.policy_tf,
            nsteps=self.nsteps,
            gamma=self.gamma,
            lam=self.lam
        )
        if self.eval_env is not None:
            self.eval_runner = Runner(
                env=self.eval_env,
                model=self.policy_tf,
                nsteps=self.nsteps,
                gamma=self.gamma,
                lam=self.lam
            )

        # additional attributes
        self.epinfobuf = deque(maxlen=100)
        self.eval_epinfobuf = deque(maxlen=100)

    def learn(self,
              total_timesteps,
              log_dir=None,
              seed=None,
              log_interval=10,
              save_interval=0):
        """Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347).

        Parameters
        ----------
        total_timesteps : int
            number of timesteps (number of actions taken in the environment)
        log_dir : str
            the directory where the training and evaluation statistics, as well
            as the tensorboard log, should be stored
        seed : int or None
            the initial seed for training, if None: keep current seed
        log_interval : int
            number of timesteps between logging events
        save_interval : int
            number of timesteps between saving events
        """
        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(log_dir)
        ensure_dir(os.path.join(log_dir, "checkpoints"))

        # Create a tensorboard object for logging.
        # save_path = os.path.join(log_dir, "tb_log")
        # writer = tf.compat.v1.summary.FileWriter(save_path)

        # file path for training and evaluation results
        log_filepath = os.path.join(log_dir, "results.csv")

        # Setup the seed value.
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        total_timesteps = int(total_timesteps)

        # Start total timer
        tfirststart = time.perf_counter()

        nupdates = total_timesteps // self.nbatch
        for update in range(1, nupdates+1):
            assert self.nbatch % self.nminibatches == 0
            # Start timer
            tstart = time.perf_counter()
            frac = 1.0 - (update - 1.0) / nupdates
            # Calculate the learning rate
            lrnow = self.lr(frac)
            # Calculate the cliprange
            cliprangenow = self.cliprange(frac)

            if update % log_interval == 0:
                print('Stepping environment...')

            # Get minibatch
            obs, returns, masks, actions, values, neglogpacs, states, epinfos \
                = self.runner.run()
            # Add to the episode info buffer.
            self.epinfobuf.extend(epinfos)

            # Run the evaluation procedure, if needed.
            if self.eval_env is not None:
                eval_obs, eval_returns, eval_masks, eval_actions, \
                    eval_values, eval_neglogpacs, eval_states, eval_epinfos \
                    = self.eval_runner.run()
                # Add to the episode info buffer.
                self.eval_epinfobuf.extend(eval_epinfos)

            if update % log_interval == 0:
                print('Done.')

            # Here what we're going to do is for each minibatch calculate the
            # loss and append it.
            mblossvals = []

            # create the indices array (index of each element of batch_size)
            inds = np.arange(self.nbatch)
            for _ in range(self.noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, self.nbatch, self.nbatch_train):
                    end = start + self.nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (
                        obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(
                        self.policy_tf.train(lrnow, cliprangenow, *slices))

            # Log training statistics.
            if update % log_interval == 0 or update == 1:
                self._log_results(update, tfirststart, tstart, mblossvals,
                                  values, returns, log_filepath)

            # Save a checkpoint of the model.
            if save_interval and (update % save_interval == 0 or update == 1):
                ckpt_path = os.path.join(log_dir, "checkpoints/itr")
                print('Saving to {}'.format(ckpt_path))
                self.policy_tf.save(ckpt_path)

        return self.policy_tf

    @staticmethod
    def _get_function(val):
        """Return a function from a constant or function.

        If `val` is a constant, it is converted to a function that simply
        returns the value. Otherwise, this method asserts `val` is callable and
        returns it.
        """
        if isinstance(val, float):
            def constfn(v):
                def f(_):
                    return v
                return f

            val = constfn(val)
        else:
            assert callable(val)

        return val

    def _collect_samples(self, env, steps):
        pass

    def _log_results(self,
                     update,
                     tfirststart,
                     tstart,
                     mblossvals,
                     values,
                     returns,
                     file_path):
        """Log training and evaluation statistics.

        Parameters
        ----------
        update : int
            the training iteration
        tfirststart : float
            time when the training procedure started
        tstart : float
            time when the current training iteration started
        mblossvals : array_like
            list of mini-batch losses
        values : array_like
            list of training iteration value estimates
        returns : array_like
            list of training iteration returns
        file_path : str
            the path where the training and evaluation statistics should be
            stored
        """
        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(self.nbatch / (tnow - tstart))
        # Calculate if value function is a good predictor of the returns
        # (ev > 1) or if it's just worse than predicting nothing (ev =< 0).
        ev = explained_variance(values, returns)

        log_statistics = {
            "misc/serial_timesteps": update * self.nsteps,
            "misc/nupdates": update,
            "misc/total_timesteps": update * self.nbatch,
            "fps": fps,
            "misc/explained_variance": float(ev),
            "eprewmean": safemean([epinfo['r'] for epinfo in self.epinfobuf]),
            'eplenmean': safemean([epinfo['l'] for epinfo in self.epinfobuf]),
            "misc/time_elapsed": tnow - tfirststart,
        }
        # Add evaluation statistics.
        if self.eval_env is not None:
            log_statistics.update({
                'eval_eprewmean': safemean(
                    [epinfo['r'] for epinfo in self.eval_epinfobuf]),
                'eval_eplenmean': safemean(
                    [epinfo['l'] for epinfo in self.eval_epinfobuf])
            })
        # Add loss statistics.
        for (lossval, lossname) in zip(lossvals, self.policy_tf.loss_names):
            log_statistics['loss/' + lossname] = lossval

        # Save statistics in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                w = csv.DictWriter(f, fieldnames=log_statistics.keys())
                if not exists:
                    w.writeheader()
                w.writerow(log_statistics)

        # Print statistics.
        print("-" * 57)
        for key in sorted(log_statistics.keys()):
            val = log_statistics[key]
            print("| {:<25} | {:<25.3g} |".format(key, float(val)))
        print("-" * 57)
        print('')


# Avoid division error when calculate the mean (in our case if epinfo is empty
# returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
