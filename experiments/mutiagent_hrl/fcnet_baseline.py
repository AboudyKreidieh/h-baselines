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
from hbaselines.algs import DDPG, DQN
from stable_baselines.deepq.policies import LnMlpPolicy as DQNPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy as DDPGPolicy

# total timesteps that training will be executed
TOTAL_TIMESTEPS = 100000

# hyperparameters for the DQN algorithm
HP_DQN = {}

# hyperparameters for the DDPG algorithm
HP_DDPG = {}

seed = None


if __name__ == '__main__':
    # if the environment is in Flow or h-baselines, register it
    env = "HalfCheetah-v2"

    # determine whether the env is discrete or continuous in the action space
    discrete = False

    # initialize the algorithm
    if discrete:
        # if discrete, use DQN
        alg = DQN(policy=DQNPolicy, env=env, **HP_DQN, verbose=2)
    else:
        # if continuous, use DDPG
        alg = DDPG(policy=DDPGPolicy, env=env, **HP_DDPG, verbose=2)

    # perform training
    trained_model = alg.learn(total_timesteps=TOTAL_TIMESTEPS,
                              log_interval=10,
                              seed=seed)
