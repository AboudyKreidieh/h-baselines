"""A runner script for goal-conditioned hierarchical models."""
import os
import json
from time import strftime
import sys

from hbaselines.utils.misc import ensure_dir
from hbaselines.utils.train import parse_options
from hbaselines.utils.train import get_hyperparameters
from hbaselines.utils.train import run_exp


def main(args, base_dir):
    """Execute multiple training operations."""
    for i in range(args.n_training):
        # value of the next seed
        seed = args.seed + i

        # The time when the current experiment started.
        now = strftime("%Y-%m-%d-%H:%M:%S")

        # Create a save directory folder (if it doesn't exist).
        if args.log_dir is not None:
            dir_name = args.log_dir
        else:
            dir_name = os.path.join(base_dir, '{}/{}'.format(
                args.env_name, now))
        ensure_dir(dir_name)

        # Get the policy class.
        if args.alg == "TD3":
            from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy
        elif args.alg == "SAC":
            from hbaselines.goal_conditioned.sac import GoalConditionedPolicy
        else:
            raise ValueError("Unknown algorithm: {}".format(args.alg))

        # Get the hyperparameters.
        hp = get_hyperparameters(args, GoalConditionedPolicy)

        # Add the seed for logging purposes.
        params_with_extra = hp.copy()
        params_with_extra['seed'] = seed
        params_with_extra['env_name'] = args.env_name
        params_with_extra['policy_name'] = "GoalConditionedPolicy"
        params_with_extra['algorithm'] = args.alg
        params_with_extra['date/time'] = now

        # Add the hyperparameters to the folder.
        with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
            json.dump(params_with_extra, f, sort_keys=True, indent=4)

        run_exp(
            env=args.env_name,
            policy=GoalConditionedPolicy,
            hp=hp,
            dir_name=dir_name,
            evaluate=args.evaluate,
            seed=seed,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            initial_exploration_steps=args.initial_exploration_steps,
            ckpt_path=args.ckpt_path,
        )


if __name__ == '__main__':
    main(
        parse_options(
            description='Test the performance of goal-conditioned '
                        'hierarchical models on various environments.',
            example_usage='python run_hrl.py "HalfCheetah-v2" '
                          '--meta_period 10',
            args=sys.argv[1:],
            hierarchical=True,
            multiagent=False,
        ),
        'data/goal-conditioned'
    )
