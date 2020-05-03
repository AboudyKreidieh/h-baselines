"""An evaluator script for pre-trained policies."""
import os
import sys
import argparse
import random
import numpy as np
import tensorflow as tf
import json
from skvideo.io import FFmpegWriter

from hbaselines.algorithms import OffPolicyRLAlgorithm
from hbaselines.fcnet.td3 import FeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy


# dictionary that maps policy names to policy objects
POLICY_DICT = {
    "FeedForwardPolicy": FeedForwardPolicy,
    "GoalConditionedPolicy": GoalConditionedPolicy,
}


def parse_options(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Run evaluation episodes of a given checkpoint.',
        epilog='python run_eval "/path/to/dir_name" ckpt_num')

    # required input parameters
    parser.add_argument(
        'dir_name', type=str, help='the path to the checkpoints folder')

    # optional arguments
    parser.add_argument(
        '--ckpt_num', type=int, default=None,
        help='the checkpoint number. If not specified, the last checkpoint is '
             'used.')
    parser.add_argument(
        '--num_rollouts', type=int, default=1, help='number of eval episodes')
    parser.add_argument(
        '--no_render', action='store_true', help='shuts off rendering')
    parser.add_argument(
        '--video', type=str, default='output.mp4',
        help='path to the video to render')

    flags, _ = parser.parse_known_args(args)

    return flags


def get_hyperparameters_from_dir(ckpt_path):
    """Collect the algorithm-specific hyperparameters from the checkpoint.

    Parameters
    ----------
    ckpt_path : str
        the path to the checkpoints folder

    Returns
    -------
    str
        environment name
    hbaselines.goal_conditioned.*
        policy object
    dict
        algorithm and policy hyperparaemters
    int
        the seed value
    """
    # import the dictionary of hyperparameters
    with open(os.path.join(ckpt_path, 'hyperparameters.json'), 'r') as f:
        hp = json.load(f)

    # collect the policy object
    policy_name = hp['policy_name']
    policy = POLICY_DICT[policy_name]

    # collect the environment name
    env_name = hp['env_name']

    # collect the seed value
    seed = hp['seed']

    # remove unnecessary features from hp dict
    hp = hp.copy()
    del hp['policy_name'], hp['env_name'], hp['seed']

    return env_name, policy, hp, seed


def main(args):
    """Execute multiple training operations."""
    flags = parse_options(args)

    # get the hyperparameters
    env_name, policy, hp, seed = get_hyperparameters_from_dir(flags.dir_name)
    hp['render'] = not flags.no_render  # to visualize the policy

    print(hp.keys())
    del hp['algorithm']
    del hp['date/time']

    # create the algorithm object. We will be using the eval environment in
    # this object to perform the rollout.
    alg = OffPolicyRLAlgorithm(
        policy=policy,
        env=env_name,
        eval_env=env_name,
        **hp
    )

    # setup the seed value
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # get the checkpoint number
    if flags.ckpt_num is None:
        filenames = os.listdir(os.path.join(flags.dir_name, "checkpoints"))
        metafiles = [f[:-5] for f in filenames if f[-5:] == ".meta"]
        metanum = [int(f.split("-")[-1]) for f in metafiles]
        ckpt_num = max(metanum)
    else:
        ckpt_num = flags.ckpt_num

    # location to the checkpoint
    ckpt = os.path.join(flags.dir_name, "checkpoints/itr-{}".format(ckpt_num))

    # restore the previous checkpoint
    alg.saver = tf.compat.v1.train.Saver(alg.trainable_vars)
    alg.load(ckpt)

    # some variables that will be needed when replaying the rollout
    policy = alg.policy_tf
    env = alg.eval_env

    # Perform the evaluation procedure.
    episdoe_rewards = []
    if not flags.no_render:
        out = FFmpegWriter(flags.video)

    if not isinstance(env, list):
        env_list = [env]
    else:
        env_list = env

    for env in env_list:

        env.images_are_rgb = True
        env.boundary_dist = 4.

        for episode_num in range(flags.num_rollouts):
            # Run a rollout.
            obs = env.reset()

            # predict a trajectory using the dynamics
            # if dynamics are present
            try:
                import matplotlib.pyplot as plt

                goal_dim = env.current_context.shape[0]
                goal = env.current_context - obs[:goal_dim]

                o = np.concatenate([obs, goal], 0).astype(np.float32)
                states, actions = policy.predict_trajectory(o[np.newaxis, :])
                states = states[0]
                actions = actions[0]
                plt.figure()
                ax = plt.subplot(111)
                env.plot_trajectory(
                    ax, states, actions, goal=env.current_context)
                plt.savefig('episode{}.png'.format(episode_num))

            except AttributeError:
                print("Skipping trajectory prediction")

            total_reward = 0
            while True:
                context = [env.current_context] \
                    if hasattr(env, "current_context") else None
                action = policy.get_action(
                    np.asarray([obs]),
                    context=context,
                    apply_noise=False,
                    random_actions=False,
                )
                obs, reward, done, _ = env.step(action[0])
                if not flags.no_render:
                    frame = env.render(mode='rgb_array', height=64, width=64)
                    #frame = np.flip(frame, axis=0)
                    #out.writeFrame(frame)
                total_reward += reward
                if done:
                    break

            # Print total returns from a given episode.
            episdoe_rewards.append(total_reward)
            print("Round {}, return: {}".format(episode_num, total_reward))

    if not flags.no_render:
        out.close()

    # Print total statistics.
    print("Average, std return: {}, {}".format(
        np.mean(episdoe_rewards), np.std(episdoe_rewards)))


if __name__ == '__main__':
    main(sys.argv[1:])
