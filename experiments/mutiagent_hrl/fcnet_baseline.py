"""A runner script for fcnet models.

This run script used to test the performance of DDPG and DQN with fully
connected network models on various environments.
"""
import os
import csv
from time import strftime

from hbaselines.utils.train import ensure_dir
from hbaselines.utils.train import create_parser, get_hyperparameters
from hbaselines.algs.ddpg import DDPG
from hbaselines.hiro.policy import FeedForwardPolicy

EXAMPLE_USAGE = 'python fcnet_baseline.py "HalfCheetah-v2" --gamma 0.995'
NUM_CPUS = 3


@ray.remote
def run_exp(env, hp, steps, dir_name, i):
    # use DDPG
    alg = DDPG(policy=FeedForwardPolicy, env=env, **hp)

    # perform training
    alg.learn(
        total_timesteps=steps,
        log_dir=os.path.join(dir_name, "results_{}.csv".format(i)),
        log_interval=10,
        callback=None,
        seed=None,
        tb_log_name=dir_name,
        reset_num_timesteps=True,
    )

    return None


def main():
    parser = create_parser(
        description='Test the performance of DDPG and DQN with fully connected'
                    ' network models on various environments.',
        example_usage=EXAMPLE_USAGE)
    args = parser.parse_args()

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

    ray.init(num_cpus=NUM_CPUS)
    ray.get([run_exp.remote(env, hp, args.steps, dir_name, i)
             for i in range(args.n_training)])
    # [run_exp(env, hp, args.steps, dir_name, i)
    #  for i in range(args.n_training)]
    ray.shutdown()


if __name__ == '__main__':
    main()
    os._exit(1)
