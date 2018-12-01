"""A runner script for LSTM models.

This run script used to test the performance of DDPG and DQN with LSTM models
on various environments.
"""
import os
import datetime
import csv
from threading import Thread

from hbaselines.utils.logger import ensure_dir
from hbaselines.utils.train import create_parser, get_hyperparameters
from hbaselines.algs.ddpg import DDPG
from hbaselines.algs.dqn import DQN
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from hbaselines.policies.ddpg import LSTMPolicy as DDPGPolicy

EXAMPLE_USAGE = 'python lstm_baseline.py "HalfCheetah-v2" --gamma 0.995'


def run_exp(env, discrete, hp, steps, dir_name, i):
    # initialize the algorithm
    if discrete:
        # if discrete, use DQN
        alg = DQN(policy=DQNPolicy, env=env, recurrent=True, **hp)
    else:
        # if continuous, use DDPG
        alg = DDPG(policy=DDPGPolicy, env=env, recurrent=True, **hp)

    # perform training
    alg.learn(
        total_timesteps=steps,
        log_interval=10,
        file_path=os.path.join(dir_name, "results_{}.csv".format(i)))

    return None


def main():
    parser = create_parser(
        description='Test the performance of DDPG and DQN with LSTM models '
                    'on various environments.',
        example_usage=EXAMPLE_USAGE)
    args = parser.parse_args()

    # create a save directory folder (if it doesn't exist)
    dir_name = 'data/lstm/{}'.format(datetime.datetime.now().time())
    dir_name = os.path.join(os.getcwd(), dir_name)
    ensure_dir(dir_name)

    # if the environment is in Flow or h-baselines, register it
    env = args.env_name

    # determine whether the env is discrete or continuous in the action space
    discrete = False

    # get the hyperparameters
    hp = get_hyperparameters(args, discrete)

    # add the hyperparameters to the folder
    with open(os.path.join(dir_name, 'hyperparameters.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=hp.keys())
        w.writeheader()
        w.writerow(hp)

    for i in range(args.n_training):
        thread = Thread(target=run_exp,
                        args=[env, discrete, hp, args.steps, dir_name, i])
        thread.start()


if __name__ == '__main__':
    main()
