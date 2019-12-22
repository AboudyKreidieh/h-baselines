"""A runner script for fcnet models."""
import os
import json
from time import strftime
import sys

from hbaselines.utils.misc import ensure_dir
from hbaselines.utils.train import parse_options, get_hyperparameters
from hbaselines.algorithms import OffPolicyRLAlgorithm
from hbaselines.fcnet.td3 import FeedForwardPolicy

EXAMPLE_USAGE = 'python run_fcnet.py "HalfCheetah-v2" --total_steps 1e6'


def run_exp(env,
            hp,
            steps,
            dir_name,
            evaluate,
            seed,
            eval_interval,
            log_interval,
            save_interval):
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
    eval_interval : int
        number of simulation steps in the training environment before an
        evaluation is performed
    log_interval : int
        the number of training steps before logging training results
    save_interval : int
        number of simulation steps in the training environment before the model
        is saved
    """
    eval_env = env if evaluate else None

    alg = OffPolicyRLAlgorithm(
        policy=FeedForwardPolicy,
        env=env,
        eval_env=eval_env,
        **hp
    )

    # perform training
    alg.learn(
        total_timesteps=steps,
        log_dir=dir_name,
        log_interval=log_interval,
        eval_interval=eval_interval,
        save_interval=save_interval,
        seed=seed,
    )


def main(args, base_dir):
    """Execute multiple training operations."""
    for i in range(args.n_training):
        # value of the next seed
        seed = args.seed + i

        # create a save directory folder (if it doesn't exist)
        dir_name = os.path.join(
            base_dir,
            '{}/{}'.format(args.env_name, strftime("%Y-%m-%d-%H:%M:%S")))
        ensure_dir(dir_name)

        # get the hyperparameters
        hp = get_hyperparameters(args, FeedForwardPolicy)

        # add the seed for logging purposes
        params_with_extra = hp.copy()
        params_with_extra['seed'] = seed
        params_with_extra['env_name'] = args.env_name
        params_with_extra['policy_name'] = "FeedForwardPolicy"

        # add the hyperparameters to the folder
        with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
            json.dump(params_with_extra, f, sort_keys=True, indent=4)

        run_exp(env=args.env_name,
                hp=hp,
                steps=args.total_steps,
                dir_name=dir_name,
                evaluate=args.evaluate,
                seed=seed,
                eval_interval=args.eval_interval,
                log_interval=args.log_interval,
                save_interval=args.save_interval)


if __name__ == '__main__':
    # collect arguments
    args = parse_options(
        description='Test the performance of TD3 with fully connected network '
                    'models on various environments.',
        example_usage=EXAMPLE_USAGE,
        args=sys.argv[1:]
    )

    # execute the training procedure
    main(args, 'data/fcnet')
