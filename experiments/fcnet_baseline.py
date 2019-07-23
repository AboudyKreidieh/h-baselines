"""A runner script for fcnet models.

This run script used to test the performance of TD3 with fully connected
network models on various environments.
"""
import os
import csv
from time import strftime
import ray
import sys

from hbaselines.common.train import ensure_dir
from hbaselines.common.train import parse_options, get_hyperparameters
from hbaselines.hiro import TD3, FeedForwardPolicy

EXAMPLE_USAGE = 'python fcnet_baseline.py "HalfCheetah-v2" --n_cpus 3'


@ray.remote
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
    alg = TD3(policy=FeedForwardPolicy, env=env, eval_env=eval_env, **hp)

    # perform training
    alg.learn(
        total_timesteps=steps,
        log_dir=dir_name,
        log_interval=1000,
        seed=None,
        exp_num=i
    )

    return None


def main(args):
    """Execute multiple training operations."""
    args = parse_options(
        description='Test the performance of TD3 with fully connected network '
                    'models on various environments.',
        example_usage=EXAMPLE_USAGE,
        args=args
    )

    # if the environment is in Flow or h-baselines, register it
    env = args.env_name

    # create a save directory folder (if it doesn't exist)
    dir_name = 'data/fcnet/{}/{}'.format(env, strftime("%Y-%m-%d-%H:%M:%S"))
    ensure_dir(dir_name)

    # get the hyperparameters
    hp = get_hyperparameters(args)

    # add the hyperparameters to the folder
    with open(os.path.join(dir_name, 'hyperparameters.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=hp.keys())
        w.writeheader()
        w.writerow(hp)

    ray.init(num_cpus=args.n_cpus)
    ray.get([run_exp.remote(env, hp, args.steps, dir_name, args.evaluate, i)
             for i in range(args.n_training)])
    ray.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])
    os._exit(1)
