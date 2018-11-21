"""A runner script for fcnet models.

This run script used to test the performance of DDPG and DQN with fully
connected network models on various environments.

Arguments
---------
env_name : str
    name of the gym environment. This environment must either be registered in
    gym, be available in the computation framework Flow, or be available within
    the hbaselines/envs folder.
seed : int
    seed used during training
blank : int  # TODO: add argument for hyperparameters
    blank
"""
import os
import numpy as np
import datetime
import csv

from hbaselines.utils.logger import ensure_dir
from hbaselines.algs import DDPG, DQN
from stable_baselines.deepq.policies import LnMlpPolicy as DQNPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy as DDPGPolicy

# total timesteps that training will be executed
TOTAL_TIMESTEPS = 1e7

# hyperparameters for the DQN algorithm
HP_DQN = dict(

)

# hyperparameters for the DDPG algorithm
HP_DDPG = dict(
    gamma=0.995,                             # 0.99
    memory_policy=None,
    nb_train_steps=50,                     # 50
    nb_rollout_steps=100,
    param_noise=None,
    action_noise=None,
    normalize_observations=True,           # False
    tau=0.0677,                               # 0.001
    batch_size=128,                         # 128
    param_noise_adaption_interval=50,
    normalize_returns=False,                # False
    enable_popart=False,
    observation_range=(-5, 5),
    critic_l2_reg=0.,
    return_range=(-np.inf, np.inf),
    actor_lr=1e-4,                          # 1e-4
    critic_lr=1e-4,                          # 1e-3
    clip_norm=None,
    reward_scale=1.,
    render=False,
    memory_limit=10000,
    verbose=2,
    tensorboard_log=None,
    _init_setup_model=True
)

seed = None
dir_name = "DDPG_{}".format(datetime.datetime.now().time())


if __name__ == '__main__':
    # create the save directory folder (if it doesn't exist)
    ensure_dir(dir_name)

    # if the environment is in Flow or h-baselines, register it
    env = "Hopper-v2"

    # determine whether the env is discrete or continuous in the action space
    discrete = False

    # add the hyperparameters to the folder
    with open(os.path.join(dir_name, "hyperparameters.csv"), 'w') as f:
        hp = HP_DQN if discrete else HP_DDPG
        w = csv.DictWriter(f, fieldnames=hp.keys())
        w.writeheader()
        w.writerow(HP_DDPG)

    # initialize the algorithm
    if discrete:
        # if discrete, use DQN
        alg = DQN(policy=DQNPolicy, env=env, **HP_DQN, verbose=2)
    else:
        # if continuous, use DDPG
        alg = DDPG(policy=DDPGPolicy, env=env, **HP_DDPG)

    # perform training
    trained_model = alg.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=10,
        seed=seed,
        dir_name=dir_name)
