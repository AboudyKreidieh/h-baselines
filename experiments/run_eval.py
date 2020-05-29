"""An evaluator script for pre-trained policies."""
import os
import sys
import argparse
import random
import numpy as np
import tensorflow as tf
import json
import time
from copy import deepcopy

from flow.core.util import emission_to_csv

from hbaselines.algorithms import OffPolicyRLAlgorithm
from hbaselines.fcnet.td3 import FeedForwardPolicy \
    as TD3FeedForwardPolicy
from hbaselines.fcnet.sac import FeedForwardPolicy \
    as SACFeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy \
    as TD3GoalConditionedPolicy
from hbaselines.goal_conditioned.sac import GoalConditionedPolicy \
    as SACGoalConditionedPolicy


# dictionary that maps policy names to policy objects
POLICY_DICT = {
    "FeedForwardPolicy": {
        "TD3": TD3FeedForwardPolicy,
        "SAC": SACFeedForwardPolicy,
    },
    "GoalConditionedPolicy": {
        "TD3": TD3GoalConditionedPolicy,
        "SAC": SACGoalConditionedPolicy,
    },
}

# name of Flow environments. These are rendered differently
FLOW_ENV_NAMES = [
    "ring",
    "ring_small",
    "figureeight0",
    "figureeight1",
    "figureeight2",
    "merge0",
    "merge1",
    "merge2",
    "highway-single"
]


def parse_options(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Run evaluation episodes of a given checkpoint.',
        epilog='python run_eval "/path/to/dir_name" ckpt_num')

    # required input parameters
    parser.add_argument(
        'dir_name', type=str, help='the path to the checkpoints folder')

    # optional arguments
    parser.add_argument(
        '--ckpt_num', type=int, default=None,
        help='the checkpoint number. If not specified, the last checkpoint is '
             'used.')
    parser.add_argument(
        '--num_rollouts', type=int, default=1,
        help='number of eval episodes')
    parser.add_argument(
        '--no_render', action='store_true',
        help='shuts off rendering')
    parser.add_argument(
        '--random_seed', action='store_true',
        help='whether to run the simulation on a random seed. If not added, '
             'the original seed is used.')

    flags, _ = parser.parse_known_args(args)

    return flags


def get_hyperparameters_from_dir(ckpt_path):
    """Collect the algorithm-specific hyperparameters from the checkpoint.

    Parameters
    ----------
    ckpt_path : str
        the path to the checkpoints folder

    Returns
    -------
    str
        environment name
    hbaselines.goal_conditioned.*
        policy object
    dict
        algorithm and policy hyperparaemters
    int
        the seed value
    """
    # import the dictionary of hyperparameters
    with open(os.path.join(ckpt_path, 'hyperparameters.json'), 'r') as f:
        hp = json.load(f)

    # collect the policy object
    policy_name = hp['policy_name']
    alg_name = hp['algorithm']
    policy = POLICY_DICT[policy_name][alg_name]

    # collect the environment name
    env_name = hp['env_name']

    # collect the seed value
    seed = hp['seed']

    # remove unnecessary features from hp dict
    hp = hp.copy()
    del hp['policy_name'], hp['env_name'], hp['seed']
    del hp['algorithm'], hp['date/time']

    return env_name, policy, hp, seed


def main(args):
    """Execute multiple training operations."""
    flags = parse_options(args)

    # get the hyperparameters
    env_name, policy, hp, seed = get_hyperparameters_from_dir(flags.dir_name)
    hp['render'] = not flags.no_render  # to visualize the policy

    # create the algorithm object. We will be using the eval environment in
    # this object to perform the rollout.
    alg = OffPolicyRLAlgorithm(
        policy=policy,
        env=env_name,
        **hp
    )

    # setup the seed value
    if not flags.random_seed:
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

    # get the checkpoint number
    if flags.ckpt_num is None:
        filenames = os.listdir(os.path.join(flags.dir_name, "checkpoints"))
        metafiles = [f[:-5] for f in filenames if f[-5:] == ".meta"]
        metanum = [int(f.split("-")[-1]) for f in metafiles]
        ckpt_num = max(metanum)
    else:
        ckpt_num = flags.ckpt_num

    # location to the checkpoint
    ckpt = os.path.join(flags.dir_name, "checkpoints/itr-{}".format(ckpt_num))

    # restore the previous checkpoint
    alg.saver = tf.compat.v1.train.Saver(alg.trainable_vars)
    alg.load(ckpt)

    # some variables that will be needed when replaying the rollout
    policy = alg.policy_tf
    env = alg.sampler.env

    # Perform the evaluation procedure.
    episdoe_rewards = []

    # Add an emission path to Flow environments.
    if env_name in FLOW_ENV_NAMES:
        sim_params = deepcopy(env.wrapped_env.sim_params)
        sim_params.emission_path = "./flow_results"
        env.wrapped_env.restart_simulation(
            sim_params, render=not flags.no_render)

    for episode_num in range(flags.num_rollouts):
        # Run a rollout.
        obs = env.reset()
        total_reward = 0
        while True:
            context = [env.current_context] \
                if hasattr(env, "current_context") else None
            action = policy.get_action(
                np.asarray([obs]),
                context=context,
                apply_noise=False,
                random_actions=False,
            )
            obs, reward, done, _ = env.step(action[0])
            if not flags.no_render:
                env.render()
            total_reward += reward
            if done:
                break

        # Print total returns from a given episode.
        episdoe_rewards.append(total_reward)
        print("Round {}, return: {}".format(episode_num, total_reward))

    # Print total statistics.
    print("Average, std return: {}, {}".format(
        np.mean(episdoe_rewards), np.std(episdoe_rewards)))

    if env_name in FLOW_ENV_NAMES:
        # wait a short period of time to ensure the xml file is readable
        time.sleep(0.1)

        # collect the location of the emission file
        dir_path = env.wrapped_env.sim_params.emission_path
        emission_filename = "{0}-emission.xml".format(
            env.wrapped_env.network.name)
        emission_path = os.path.join(dir_path, emission_filename)

        # convert the emission file into a csv
        emission_to_csv(emission_path)

        # Delete the .xml version of the emission file.
        os.remove(emission_path)


if __name__ == '__main__':
    main(sys.argv[1:])
