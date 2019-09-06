"""A runner script for goal-directed hierarchical models.

This run script used to test the performance of DDPG and DQN with the  neural
network model "HIRO" on various environments.

See: TODO: add paper
"""
import os
from time import strftime
import csv
import sys

from hbaselines.common.train import ensure_dir
from hbaselines.common.train import parse_options, get_hyperparameters
from hbaselines.hiro import TD3, GoalDirectedPolicy

EXAMPLE_USAGE = 'python hiro_baseline.py "HalfCheetah-v2" --n_cpus 3'


def run_exp(env, hp, steps, dir_name, evaluate, i):
    """Run a single training procedure.

    Parameters
    ----------
    env : str or gym.Env
        the training/testing environment
    hp : dict
        additional algorithm hyper-parameters
    steps : int
        total number of training steps
    dir_name : str
        the location the results files are meant to be stored
    evaluate : bool
        whether to include an evaluation environment
    i : int
        an increment term, used for logging purposes
    """
    eval_env = env if evaluate else None
    alg = TD3(policy=GoalDirectedPolicy, env=env, eval_env=eval_env, **hp)

    # perform training
    alg.learn(
        total_timesteps=steps,
        log_dir=dir_name,
        log_interval=2000,
        seed=None,
        exp_num=i
    )

    return None


def main(args, base_dir):
    """Execute multiple training operations."""
    # create a save directory folder (if it doesn't exist)
    dir_name = os.path.join(
        base_dir, '{}/{}'.format(args.env_name, strftime("%Y-%m-%d-%H:%M:%S")))
    ensure_dir(dir_name)

    # get the hyperparameters
    hp = get_hyperparameters(args)

    # add the hyperparameters to the folder
    with open(os.path.join(dir_name, 'hyperparameters.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=hp.keys())
        w.writeheader()
        w.writerow(hp)

    for i in range(args.n_training):
        run_exp(args.env_name, hp, args.steps, dir_name, args.evaluate, i)


if __name__ == '__main__':
    # collect arguments
    args = parse_options(
        description='Test the performance of TD3 with goal-directed '
                    'hierarchical models on various environments.',
        example_usage=EXAMPLE_USAGE,
        args=sys.argv[1:]
    )

    # execute the training procedure
    main(args, 'data/goal-directed')

    # exit from the process
    os._exit(1)
