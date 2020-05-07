"""A runner script for imitating on an expert policy."""
import os
import json
from time import strftime
import sys

from hbaselines.utils.misc import ensure_dir
from hbaselines.utils.train import parse_imitation_options
from hbaselines.utils.train import get_imitation_hyperparameters
from hbaselines.algorithms import DAggerAlgorithm

EXAMPLE_USAGE = 'python run_imitation.py "ring-imitation" --total_steps 1e6'


def run_exp(env,
            policy,
            hp,
            steps,
            dir_name,
            seed,
            log_interval,
            save_interval,
            initial_sample_steps):
    """Run a single imitation procedure.

    Parameters
    ----------
    env : str or gym.Env
        the training/testing environment
    policy : type [ hbaselines.base_policies.ImitationLearningPolicy ]
        the policy class to use
    hp : dict
        additional algorithm hyper-parameters
    steps : int
        total number of training steps
    dir_name : str
        the location the results files are meant to be stored
    seed : int
        specified the random seed for numpy, tensorflow, and random
    log_interval : int
        the number of training steps before logging training results
    save_interval : int
        number of simulation steps in the training environment before the model
        is saved
    initial_sample_steps : int
        the number of steps to initialize the replay buffer with before
        beginning training
    """
    alg = DAggerAlgorithm(
        policy=policy,
        env=env,
        **hp
    )

    # perform training
    alg.learn(
        total_timesteps=steps,
        log_dir=dir_name,
        log_interval=log_interval,
        save_interval=save_interval,
        initial_sample_steps=initial_sample_steps,
        seed=seed,
    )


def main(args, base_dir):
    """Execute multiple training operations."""
    for i in range(args.n_training):
        # value of the next seed
        seed = args.seed + i

        # The time when the current experiment started.
        now = strftime("%Y-%m-%d-%H:%M:%S")

        # Create a save directory folder (if it doesn't exist).
        dir_name = os.path.join(base_dir, '{}/{}'.format(args.env_name, now))
        ensure_dir(dir_name)

        # Get the policy class.
        from hbaselines.fcnet.imitation import FeedForwardPolicy

        # Get the hyperparameters.
        hp = get_imitation_hyperparameters(args, FeedForwardPolicy)

        # Add the seed for logging purposes.
        params_with_extra = hp.copy()
        params_with_extra['seed'] = seed
        params_with_extra['env_name'] = args.env_name
        params_with_extra['policy_name'] = "FeedForwardPolicy"
        params_with_extra['algorithm'] = 'DAgger'
        params_with_extra['date/time'] = now

        # Add the hyperparameters to the folder.
        with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
            json.dump(params_with_extra, f, sort_keys=True, indent=4)

        run_exp(
            env=args.env_name,
            policy=FeedForwardPolicy,
            hp=hp,
            steps=args.total_steps,
            dir_name=dir_name,
            seed=seed,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            initial_sample_steps=args.initial_sample_steps,
        )


if __name__ == '__main__':
    # collect arguments
    flags = parse_imitation_options(
        description='Perform an imitation learning operation.',
        example_usage=EXAMPLE_USAGE,
        args=sys.argv[1:]
    )

    # execute the training procedure
    main(flags, 'data/imitation')
