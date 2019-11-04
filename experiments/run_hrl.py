"""A runner script for goal-directed hierarchical models.

This run script used to test the performance of DDPG and DQN with the neural
network model "HIRO" on various environments.

See: TODO: add paper
"""
import os
import json
from time import strftime
import sys

from hbaselines.utils.misc import ensure_dir
from hbaselines.utils.train import parse_options, get_hyperparameters
from hbaselines.goal_conditioned import TD3, GoalConditionedPolicy

EXAMPLE_USAGE = 'python run_hrl.py "HalfCheetah-v2" --n_cpus 3'


def run_exp(env, hp, steps, dir_name, evaluate, seed):
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
    seed : int
        specified the random seed for numpy, tensorflow, and random
    """
    eval_env = env if evaluate else None
    alg = TD3(policy=GoalConditionedPolicy, env=env, eval_env=eval_env, **hp)

    # perform training
    alg.learn(
        total_timesteps=steps,
        log_dir=dir_name,
        log_interval=2000,
        seed=seed,
    )

    return None


def main(args, base_dir):
    """Execute multiple training operations."""
    # create a save directory folder (if it doesn't exist)
    dir_name = os.path.join(
        base_dir, '{}/{}'.format(args.env_name, strftime("%Y-%m-%d-%H:%M:%S")))
    ensure_dir(dir_name)

    # get the hyperparameters
    hp = get_hyperparameters(args, GoalConditionedPolicy)

    # add the seed for logging purposes
    params_with_seed = hp.copy()
    params_with_seed['seed'] = args.seed

    # add the hyperparameters to the folder
    with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
        json.dump(params_with_seed, f, sort_keys=True, indent=4)

    run_exp(args.env_name, hp, args.total_steps, dir_name, args.evaluate,
            args.seed)


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
