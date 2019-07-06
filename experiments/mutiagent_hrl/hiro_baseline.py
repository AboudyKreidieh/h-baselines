"""A runner script for HIRO models.

This run script used to test the performance of DDPG and DQN with the  neural
network model "HIRO" on various environments.

See: TODO: add paper
"""
import os
import time
import csv
import ray

from hbaselines.common.train import ensure_dir
from hbaselines.common.train import create_parser, get_hyperparameters
from hbaselines.hiro import DDPG, HIROPolicy

EXAMPLE_USAGE = 'python hiro_baseline.py "HalfCheetah-v2" --gamma 0.995'
NUM_CPUS = 3


@ray.remote
def run_exp(env, hp, steps, dir_name, i):
    # TODO: make evaluation an option
    alg = DDPG(policy=HIROPolicy, env=env, eval_env=env, **hp)

    # perform training
    alg.learn(
        total_timesteps=steps,
        log_dir=dir_name,
        log_interval=10000,
        seed=None,
        exp_num=i
    )

    return None


def main():
    parser = create_parser(
        description='Test the performance of DDPG and DQN with LSTM models '
                    'on various environments.',
        example_usage=EXAMPLE_USAGE)
    args = parser.parse_args()

    # create a save directory folder (if it doesn't exist)
    dir_name = 'data/hiro/{}'.format(time.strftime("%Y-%m-%d-%H:%M:%S"))
    ensure_dir(dir_name)

    # get the hyperparameters
    hp = get_hyperparameters(args)

    # add the hyperparameters to the folder
    with open(os.path.join(dir_name, 'hyperparameters.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=hp.keys())
        w.writeheader()
        w.writerow(hp)

    ray.init(num_cpus=NUM_CPUS)
    ray.get([run_exp.remote(args.env_name, hp, args.steps, dir_name, i)
             for i in range(args.n_training)])
    ray.shutdown()


if __name__ == '__main__':
    main()
    os._exit(1)
