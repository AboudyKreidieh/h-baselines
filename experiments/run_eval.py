"""An evaluator script for pre-trained policies."""
import os
import sys
import random
import numpy as np
import tensorflow as tf
from copy import deepcopy
from skvideo.io import FFmpegWriter

from hbaselines.algorithms import RLAlgorithm
from hbaselines.utils.eval import parse_options
from hbaselines.utils.eval import get_hyperparameters_from_dir
from hbaselines.utils.eval import TrajectoryLogger

# name of Flow environments. These are rendered differently
FLOW_ENV_NAMES = [
    "ring-v0",
    "ring-v0-fast",
    "ring-v1-fast",
    "ring-v2-fast",
    "ring-v3-fast",
    "ring-v4-fast",
    "merge-v0",
    "merge-v1",
    "merge-v2",
    "highway-v0",
    "highway-v1",
    "highway-v2",
    "i210-v0",
    "i210-v1",
    "i210-v2",
]


def main(args):
    """Execute multiple training operations."""
    flags = parse_options(args)

    # Run assertions.
    assert not (flags.no_render and flags.save_video), \
        "If saving the rendering, no_render cannot be set to True."

    # get the hyperparameters
    env_name, policy, hp, seed = get_hyperparameters_from_dir(flags.dir_name)
    hp['num_envs'] = 1
    hp['render_eval'] = not flags.no_render  # to visualize the policy
    multiagent = env_name.startswith("multiagent")

    # create the algorithm object. We will be using the eval environment in
    # this object to perform the rollout.
    alg = RLAlgorithm(
        policy=policy,
        env=env_name,
        eval_env=env_name,
        **hp
    )

    # setup the seed value
    if not flags.random_seed:
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

    if flags.save_trajectory:
        logger = TrajectoryLogger(env_name)
    else:
        logger = None

    # Perform the evaluation procedure.
    episode_rewards = []

    # Add an emission path to Flow environments.
    if env_name in FLOW_ENV_NAMES:
        sim_params = deepcopy(env.wrapped_env.sim_params)
        sim_params.emission_path = "./flow_results"
        env.wrapped_env.restart_simulation(
            sim_params, render=not flags.no_render)

    if not isinstance(env, list):
        env_list = [env]
    else:
        env_list = env

    for env_num, env in enumerate(env_list):
        for episode_num in range(flags.num_rollouts):
            if not flags.no_render and env_name not in FLOW_ENV_NAMES:
                out = FFmpegWriter("{}_{}_{}.mp4".format(
                    flags.video, env_num, episode_num))
            else:
                out = None

            obs, total_reward = env.reset(), 0

            if flags.save_trajectory:
                logger.reset(env)

            while True:
                context = [env.current_context] \
                    if hasattr(env, "current_context") else None

                if multiagent:
                    processed_obs = {
                        key: np.array([obs[key]]) for key in obs.keys()}
                else:
                    processed_obs = np.asarray([obs])

                action = policy.get_action(
                    obs=processed_obs,
                    context=context,
                    apply_noise=False,
                    random_actions=False,
                )

                # Flatten the actions to pass to step.
                if multiagent:
                    action = {key: action[key][0] for key in action.keys()}
                else:
                    action = action[0]

                # Visualize the sub-goals of the hierarchical policy.
                if hasattr(policy, "meta_action") \
                        and policy.meta_action is not None \
                        and hasattr(env, "set_goal"):
                    goal = np.array([
                        policy.meta_action[0][i] +
                        (obs[policy.goal_indices]
                         if policy.relative_goals else 0)
                        for i in range(policy.num_levels - 1)
                    ])
                    env.set_goal(goal)

                # Advance the simulation by one step.
                new_obs, reward, done, info = env.step(action)

                if flags.save_trajectory:
                    logger.log_sample(new_obs, policy)

                # Render the new step.
                if not flags.no_render:
                    if flags.save_video:
                        if alg.env_name == "AntGather":
                            out.writeFrame(env.render(mode='rgb_array'))
                        else:
                            out.writeFrame(env.render(
                                mode='rgb_array', height=1024, width=1024))
                    else:
                        env.render()

                if multiagent:
                    if (isinstance(done, dict) and done["__all__"]) \
                            or done is True:
                        break
                    obs0_transition = {
                        key: np.array(obs[key]) for key in obs.keys()}
                    obs1_transition = {
                        key: np.array(new_obs[key]) for key in new_obs.keys()}
                    total_reward += sum(
                        reward[key] for key in reward.keys())
                else:
                    if done:
                        break
                    obs0_transition = obs
                    obs1_transition = new_obs
                    total_reward += reward

                policy.store_transition(
                    obs0=obs0_transition,
                    context0=context[0] if context is not None else None,
                    action=action,
                    reward=reward,
                    obs1=obs1_transition,
                    context1=context[0] if context is not None else None,
                    done=done,
                    is_final_step=done,
                    evaluate=True,
                )

                obs = new_obs

            # Print total returns from a given episode.
            episode_rewards.append(total_reward)
            print("Round {}, return: {}".format(episode_num, total_reward))
            for key in info.keys():
                print("Round {}, {}: {}".format(episode_num, key, info[key]))

            # Save the video.
            if not flags.no_render and env_name not in FLOW_ENV_NAMES \
                    and flags.save_video:
                out.close()

            # Save logged trajectory data.
            if flags.save_trajectory:
                logger.save(
                    "log_{}_{}".format(env_num, episode_num), plot=True)

    # Print total statistics.
    print("Average, std return: {}, {}".format(
        np.mean(episode_rewards), np.std(episode_rewards)))


if __name__ == '__main__':
    main(sys.argv[1:])
