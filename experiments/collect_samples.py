"""A script for collecting initial samples for imitation learning."""
import numpy as np
import sys
import os
import argparse
import json
from time import strftime
from tqdm import tqdm

from hbaselines.utils.env_util import create_env
from hbaselines.utils.misc import ensure_dir


def parse_args(args):
    """Parse script-specific arguments."""
    parser = argparse.ArgumentParser()

    # required input parameters
    parser.add_argument(
        'env_name', type=str,
        help='Name of the gym environment. This environment must either be '
             'registered in gym, be available in the computation framework '
             'Flow, or be available within the hbaselines/envs folder.')

    # optional input parameters
    parser.add_argument(
        '--render', action='store_true',
        help='enable rendering of the environment')
    parser.add_argument(
        '--samples', type=int, default=100000,
        help='Number of samples to collect.')
    parser.add_argument(
        '--log_dir', type=str, default='./imitation_sample',
        help='Number of samples to collect.')

    # parse the arguments
    flags, _ = parser.parse_known_args(args)
    return flags


def main(args):
    """Perform the sampling operation."""
    # Some variables to store sample data.
    states = []
    contexts = []
    next_stats = []
    next_contexts = []
    actions = []

    # Parse the arguments.
    flags = parse_args(args)

    # The path that the csv data will be stored in.
    log_dir = os.path.join(flags.log_dir, flags.env_name)
    ensure_dir(log_dir)
    log_dir = os.path.join(
        log_dir, "{}.json".format(strftime("%Y-%m-%d-%H:%M:%S")))

    # Create the environment.
    env = create_env(
        flags.env_name, flags.render,
        shared=False,
        maddpg=False,
        evaluate=False
    )
    obs = env.reset()

    for _ in tqdm(range(flags.samples)):
        # Collect the contextual term. None if it is not passed.
        context = [env.current_context] if hasattr(env, "current_context") \
            else None

        # Add the contextual term to the observation.
        if context is not None:
            obs = np.concatenate((obs, context), axis=0)

        # Query the environment for the expert action.
        action = env.query_expert(obs)

        # Update the environment.
        next_obs, _, done, _ = env.step(action)

        # Store the next sample in the initial variables.
        states.append(list(obs))
        next_stats.append(list(next_obs))
        actions.append(action)

        # Some book-keeping.
        obs = next_obs.copy()

        if done:
            # Reset the environment.
            obs = env.reset()

    # Store the data in the json file.
    samples = {
        'states': states,
        'next_states': next_stats,
        'actions': actions
    }
    with open(log_dir, 'w') as fp:
        json.dump(samples, fp)


if __name__ == "__main__":
    main(sys.argv[1:])
