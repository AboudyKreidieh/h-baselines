import tensorflow as tf
import random
import os
import time
import numpy as np
import os.path as osp
from hbaselines.ppo import logger
from collections import deque
from hbaselines.ppo.common import explained_variance
from hbaselines.ppo.common.policies import build_policy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from hbaselines.ppo.runner import Runner
from baselines.ppo2.model import Model


def constfn(val):
    def f(_):
        return val
    return f


class PPO(object):

    def __init__(self,
                 network,
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
                 load_path=None,
                 model_fn=None,
                 update_fn=None,
                 init_fn=None,
                 mpi_rank_weight=1,
                 comm=None,
                 **network_kwargs):
        """Instantiate the algorithm object.

        Parameters
        ----------
        network : TODO
            policy network architecture. Either string (mlp, lstm, lnlstm,
            cnn_lstm, cnn, cnn_small, conv_only, see baselines.common/models.py
            for full list) specifying the standard network architecture, or a
            function that takes tensorflow tensor as input and returns tuple
            (output_tensor, extra_feed) where output tensor is the last network
            layer output, extra_feed is None for feed-forward neural nets, and
            extra_feed is a dictionary describing how to feed state into the
            network for recurrent neural nets. See common/models.py/lstm for
            more details on using recurrent nets in policies
        env : baselines.common.vec_env.VecEnv
            the environment. Needs to be vectorized for parallel environment
            simulation. The environments produced by gym.make can be wrapped
            using baselines.common.vec_env.DummyVecEnv class.
        eval_env : TODO
            TODO
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
        model_fn : TODO
            TODO
        update_fn : TODO
            TODO
        init_fn : TODO
            TODO
        mpi_rank_weight : TODO
            TODO
        comm : TODO
            TODO
        network_kwargs : dict
            keyword arguments to the policy / network builder. See baselines.
            common/policies.py/build_policy and arguments to a particular type
            of network. For instance, 'mlp' network architecture has arguments
            num_hidden and num_layers.
        """
        self.network = network
        self.env = env
        self.eval_env = eval_env
        self.nsteps = nsteps
        self.ent_coef = ent_coef
        self.lr = lr
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.cliprange = cliprange
        self.load_path = load_path
        self.model_fn = model_fn or Model
        self.update_fn = update_fn
        self.init_fn = init_fn
        self.mpi_rank_weight = mpi_rank_weight
        self.comm = comm
        self.network_kwargs = network_kwargs

        # TODO: create shared method
        if isinstance(lr, float):
            self.lr = constfn(lr)
        else:
            assert callable(lr)
            self.lr = lr

        # TODO: create shared method
        if isinstance(cliprange, float):
            self.cliprange = constfn(cliprange)
        else:
            assert callable(cliprange)
            self.cliprange = cliprange

        # TODO: see what this is
        if init_fn is not None:
            init_fn()

        # create the policy object
        self.policy = build_policy(env, network, **network_kwargs)

        # Get the nb of env
        nenvs = self.env.num_envs

        # Get state_space and action_space
        ob_space = self.env.observation_space
        ac_space = self.env.action_space

        # Calculate the batch_size
        self.nbatch = nenvs * self.nsteps
        self.nbatch_train = self.nbatch // self.nminibatches

        # instantiate the model object (that creates act_model and train_model)
        self.model = self.model_fn(
            policy=self.policy,
            ob_space=ob_space,
            ac_space=ac_space,
            nbatch_act=nenvs,
            nbatch_train=self.nbatch_train,
            nsteps=nsteps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            comm=comm,
            mpi_rank_weight=mpi_rank_weight
        )

        # load the pre-defined model parameters, if needed  TODO: remove
        if load_path is not None:
            self.model.load(load_path)

        # Instantiate the runner objects
        self.runner = Runner(
            env=self.env,
            model=self.model,
            nsteps=self.nsteps,
            gamma=self.gamma,
            lam=self.lam
        )
        if self.eval_env is not None:
            self.eval_runner = Runner(
                env=self.eval_env,
                model=self.model,
                nsteps=self.nsteps,
                gamma=self.gamma,
                lam=self.lam
            )

        # additional attributes
        self.epinfobuf = deque(maxlen=100)
        self.eval_epinfobuf = deque(maxlen=100)

    def learn(self,
              total_timesteps,
              seed=None,
              log_interval=10,
              save_interval=0):
        """Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347).

        Parameters
        ----------
        total_timesteps : int
            number of timesteps (number of actions taken in the environment)
        seed : int
            TODO
        log_interval : int
            number of timesteps between logging events
        save_interval : int
            number of timesteps between saving events
        """
        # set global seeds
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        total_timesteps = int(total_timesteps)

        is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

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

            if update % log_interval == 0 and is_mpi_root:
                logger.info('Stepping environment...')

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

            if update % log_interval == 0 and is_mpi_root:
                logger.info('Done.')

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
                        self.model.train(lrnow, cliprangenow, *slices))

            # Feedforward --> get losses --> update
            lossvals = np.mean(mblossvals, axis=0)
            # End timer
            tnow = time.perf_counter()
            # Calculate the fps (frame per second)
            fps = int(self.nbatch / (tnow - tstart))

            if self.update_fn is not None:
                self.update_fn(update)

            if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predictor of the
                # returns (ev > 1) or if it's just worse than predicting
                # nothing (ev =< 0)
                ev = explained_variance(values, returns)
                logger.logkv("misc/serial_timesteps", update * self.nsteps)
                logger.logkv("misc/nupdates", update)
                logger.logkv("misc/total_timesteps", update * self.nbatch)
                logger.logkv("fps", fps)
                logger.logkv("misc/explained_variance", float(ev))
                logger.logkv('eprewmean',
                             safemean([epinfo['r']
                                       for epinfo in self.epinfobuf]))
                logger.logkv('eplenmean',
                             safemean([epinfo['l']
                                       for epinfo in self.epinfobuf]))
                if self.eval_env is not None:
                    logger.logkv('eval_eprewmean', safemean(
                        [epinfo['r'] for epinfo in self.eval_epinfobuf]))
                    logger.logkv('eval_eplenmean', safemean(
                        [epinfo['l'] for epinfo in self.eval_epinfobuf]))
                logger.logkv('misc/time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals,
                                               self.model.loss_names):
                    logger.logkv('loss/' + lossname, lossval)

                logger.dumpkvs()
            if save_interval and (update % save_interval == 0 or update == 1) \
                    and logger.get_dir() and is_mpi_root:
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i' % update)
                print('Saving to', savepath)
                self.model.save(savepath)

        return self.model


# Avoid division error when calculate the mean (in our case if epinfo is empty
# returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
