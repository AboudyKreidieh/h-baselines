"""An evaluator script for pre-trained policies."""
import os
import sys
import argparse
import random
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

from hbaselines.algorithms import OffPolicyRLAlgorithm
from hbaselines.fcnet.td3 import FeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy


# dictionary that maps policy names to policy objects
POLICY_DICT = {
    "FeedForwardPolicy": FeedForwardPolicy,
    "GoalConditionedPolicy": GoalConditionedPolicy,
}

"""

python run_analysis.py --dir_name data/goal-conditioned/AntMaze/2020-06-05-08:11:54  data/goal-conditioned/AntMaze/2020-06-05-08:12:55  data/goal-conditioned/AntMaze/2020-06-05-17:07:44  data/goal-conditioned/AntMaze/2020-06-06-11:37:20 data/goal-conditioned/AntMaze/2020-06-05-08:12:23  data/goal-conditioned/AntMaze/2020-06-05-17:07:14  data/goal-conditioned/AntMaze/2020-06-05-17:07:52  data/goal-conditioned/AntMaze/2020-06-06-11:37:29 data/goal-conditioned/AntMaze/2020-06-05-08:12:47  data/goal-conditioned/AntMaze/2020-06-05-17:07:24  data/goal-conditioned/AntMaze/2020-06-06-11:37:13  data/goal-conditioned/AntMaze/2020-06-06-11:37:45 --name "CHER(0.001)" "HRL" "CHER(0.1)" "HRL" "CHER(0.1)" "HIRO" "CHER(0.001)" "CHER(0.1)" "HIRO" "HRL" "HIRO" "CHER(0.001)"

"""


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
        '--dir_name', type=str, nargs='+',
        help='the path to the checkpoints folder')
    parser.add_argument(
        '--name', type=str, nargs='+',
        help='the name of the experiment that we are inspecting.')

    # optional arguments
    parser.add_argument(
        '--num_batches', type=int, default=100,
        help='number of batches ot sample from the replay buffer')
    parser.add_argument(
        '--metric', type=str, default='kl', help='kl divergence')

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
    policy = POLICY_DICT[policy_name]

    # collect the environment name
    env_name = hp['env_name']

    # collect the seed value
    seed = hp['seed']

    # remove unnecessary features from hp dict
    hp = hp.copy()
    del hp['policy_name'], hp['env_name'], hp['seed']

    return env_name, policy, hp, seed


def main(args):
    """Execute multiple training operations."""
    flags = parse_options(args)

    data = {
        'name': [],
        'step': [],
        'distance': [],
    }

    for dir_name, name in zip(flags.dir_name, flags.name):

        # get the hyperparameters
        env_name, policy, hp, seed = get_hyperparameters_from_dir(dir_name)

        print(hp.keys())
        del hp['algorithm']
        del hp['date/time']

        # create the algorithm object. We will be using the eval environment in
        # this object to perform the rollout.
        alg = OffPolicyRLAlgorithm(
            policy=policy, env=env_name, eval_env=env_name, **hp)

        # setup the seed value
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        filenames = os.listdir(os.path.join(dir_name, "checkpoints"))
        metafiles = [f[:-5] for f in filenames if f[-5:] == ".meta"]
        metanum = list(sorted([int(f.split("-")[-1]) for f in metafiles]))[:-1]

        # get the checkpoint number
        ckpt_num = max(metanum)

        # location to the checkpoint
        ckpt = os.path.join(dir_name, "checkpoints/itr-{}".format(ckpt_num))

        # restore the previous checkpoint
        alg.saver = tf.compat.v1.train.Saver(alg.trainable_vars)
        alg.load(ckpt)

        # some variables that will be needed when replaying the rollout
        policy = alg.policy_tf

        batches = []
        for b in range(flags.num_batches):
            worker_obs0 = policy.replay_buffer.sample(with_additional=False)[5]
            batches.append(worker_obs0)

        for ckpt_num_one, ckpt_num_two in zip(metanum[1:], metanum[:-1]):

            # get the checkpoint number
            ckpt_num = ckpt_num_one

            # location to the checkpoint
            ckpt = os.path.join(dir_name, "checkpoints/itr-{}".format(ckpt_num))

            # restore the previous checkpoint
            alg.load(ckpt)

            # some variables that will be needed when replaying the rollout
            policy = alg.policy_tf

            mean_one = []
            for b in batches:
                a = policy.policy[-1].get_action(b, None, False, False)
                mean_one.append(a)

            # get the checkpoint number
            ckpt_num = ckpt_num_two

            # location to the checkpoint
            ckpt = os.path.join(dir_name, "checkpoints/itr-{}".format(ckpt_num))

            # restore the previous checkpoint
            alg.load(ckpt)

            # some variables that will be needed when replaying the rollout
            policy = alg.policy_tf

            mean_two = []
            for b in batches:
                a = policy.policy[-1].get_action(b, None, False, False)
                mean_two.append(a)

            # compute a distance metric between the policies
            mean_one = np.concatenate(mean_one, axis=0)
            mean_two = np.concatenate(mean_two, axis=0)
            kl = np.sum((mean_one - mean_two) ** 2, axis=1).mean()
            print("{},{},{},{}".format(name, ckpt_num_one, ckpt_num_two, kl))

            data['name'].append(name)
            data['step'].append(ckpt_num_one)
            data['distance'].append(kl)

    df = pd.DataFrame(data, columns=['name', 'step', 'distance'])
    plt.title("Manager MDP Non-Stationarity")
    ax = sns.lineplot(x='step', y='distance', hue='name', data=df)
    plt.savefig('ns.png')


if __name__ == '__main__':
    main(sys.argv[1:])
