"""Perform hyperparameter tuning on an environment.

Tuning is performed using "n" random combinations of hyperparams from the range
of tunable parameters described below.

Hyper-parameters were chosen compliments of this paper:
https://arxiv.org/pdf/1705.05035.pdf

Example Usage
-------------
    python fcnet_baseline.py "HalfCheetah-v2" "HIROPolicy"
"""
import numpy as np
import random
import argparse
import os
import subprocess
import ray
import ray.tune as tune

# range of values for different hyper-parameters
LEARNING_RATE = [1e-5, 1e-3]
GAMMA = [0.95, 0.99, 0.995, 0.999]
REWARD_SCALE = [0.0005, 0.001, 0.01, 0.015, 0.1, 1]
TARGET_UPDATE = [1e-3, 1e-1]
GRADIENT_CLIPPING = [0, 10]
NUM_CPUS = 6
# TODO: look into nb_train_steps and nb_rollout_steps


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Perform hyperparameter tuning on an environment.',
        epilog='python fcnet_baseline.py "HalfCheetah-v2" "HIROPolicy"')

    # required input parameters
    parser.add_argument(
        'env', type=str,
        help='Name of the gym environment. This environment must either be '
             'registered in gym, be available in the computation framework '
             'Flow, or be available within the hbaselines/envs folder.')
    parser.add_argument(
        'policy', type=str,
        help='Type of policy to use. Must be one of: {"FullyConnectedPolicy", '
             '"LSTMPolicy", "FeudalPolicy", "HIROPolicy".')

    # optional input parameters
    parser.add_argument(
        '-n', type=int, default=20,
        help='number of hyperparameters to test, defaults to 20')
    parser.add_argument(
        '-s',  type=int, default=6,
        help='number of seeds to perform, defaults to 6')
    parser.add_argument(
        '--plot',  action='store_true',
        help='specifies whether to plot the results at the end of the tuning '
             'procedure')

    return parser

def call_baseline(config, reporter):
    # retrieve hyperparameters
    learning_rate = config['learning_rate']
    gamma = config['gamma']
    batch_size = config['batch_size']
    reward_scale = config['reward_scale']
    target_update = config['target_update']
    gradient_clipping = config['gradient_clipping']
    print('Chosen hyperparameters:')
    print(' - learning rate: {}'.format(learning_rate))
    print(' - gamma:         {}'.format(gamma))
    print(' - batch size:    {}'.format(batch_size))
    print(' - reward scale:  {}'.format(reward_scale))
    print(' - target update: {}'.format(target_update))
    print(' - grad clipping: {}'.format(gradient_clipping))
    print()

    # execute training on the chosen hyper-parameters
    subprocess.call([
        'python', '{}'.format(runner), '{}'.format(args.env),
        '--n_training', '{}'.format(args.s),
        '--actor_lr', '{}'.format(learning_rate),
        '--critic_lr', '{}'.format(learning_rate),
        '--gamma', '{}'.format(gamma),
        '--batch_size', '{}'.format(batch_size),
        '--reward_scale', '{}'.format(reward_scale),
        '--tau', '{}'.format(target_update),
        '--clip_norm', '{}'.format(gradient_clipping),
    ])

    reporter(done=True)

if __name__ == '__main__':
    p = create_parser()
    args = p.parse_args()

    if args.policy == 'FullyConnectedPolicy':
        runner = os.path.join(os.getcwd(), 'fcnet_baseline.py')
    elif args.policy == 'LSTMPolicy':
        runner = os.path.join(os.getcwd(), 'lstm_baseline_ray.py')
    elif args.policy == 'FeudalPolicy':
        runner = os.path.join(os.getcwd(), 'feudal_baseline.py')
    elif args.policy == 'HIROPolicy':
        runner = os.path.join(os.getcwd(), 'hiro_baseline.py')
    else:
        raise AssertionError('policy must be one of: {"FullyConnectedPolicy", '
                             '"LSTMPolicy", "FeudalPolicy", "HIROPolicy"}')

    # initialize ray
    ray.init(num_cpus=NUM_CPUS)

    # choose a set of hyper-parameters
    config = {}
    config['learning_rate'] = 10 ** random.uniform(
        np.log10(LEARNING_RATE[0]), np.log10(LEARNING_RATE[1]))
    config['gamma'] = 0.99  # keeping this constant for now
    config['batch_size'] = 256  # keeping this constant for now
    config['reward_scale'] = 0.01  # keeping this constant for now
    config['target_update'] = 10 ** random.uniform(
        np.log10(TARGET_UPDATE[0]), np.log10(TARGET_UPDATE[1]))
    config['gradient_clipping'] = random.uniform(
        GRADIENT_CLIPPING[0], GRADIENT_CLIPPING[1])
    trials = tune.run_experiments({
        'lstm_baseline': {
            'trial_resources': {'cpu': NUM_CPUS},
            'run': call_baseline,
            'config': {**config},
            'stop': {'done': True},
            'num_samples': args.n,
        },
    })
