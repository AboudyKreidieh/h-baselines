"""Contains tests for the model abstractions and different models."""
import unittest
import tensorflow as tf
import numpy as np
import random
from gym.spaces import Box

from hbaselines.utils.train import parse_options
from hbaselines.utils.train import get_hyperparameters
from hbaselines.utils.reward_fns import negative_distance
from hbaselines.utils.env_util import get_meta_ac_space
from hbaselines.utils.env_util import get_state_indices
from hbaselines.utils.env_util import import_flow_env
from hbaselines.utils.tf_util import layer
from hbaselines.utils.tf_util import conv_layer
from hbaselines.utils.tf_util import apply_squashing_func
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import gaussian_likelihood
from hbaselines.fcnet.td3 import FeedForwardPolicy \
    as TD3FeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy \
    as TD3GoalConditionedPolicy
from hbaselines.multiagent.td3 import MultiFeedForwardPolicy \
    as TD3MultiFeedForwardPolicy
from hbaselines.multiagent.h_td3 import MultiGoalConditionedPolicy \
    as TD3MultiGoalConditionedPolicy
from hbaselines.fcnet.sac import FeedForwardPolicy \
    as SACFeedForwardPolicy
from hbaselines.fcnet.ppo import FeedForwardPolicy \
    as PPOFeedForwardPolicy
from hbaselines.algorithms.rl_algorithm import TD3_PARAMS
from hbaselines.algorithms.rl_algorithm import SAC_PARAMS
from hbaselines.algorithms.rl_algorithm import PPO_PARAMS
from hbaselines.algorithms.rl_algorithm import FEEDFORWARD_PARAMS
from hbaselines.algorithms.rl_algorithm import MULTIAGENT_PARAMS
from hbaselines.algorithms.rl_algorithm import GOAL_CONDITIONED_PARAMS


class TestTrain(unittest.TestCase):
    """A simple test to get Travis running."""

    def test_parse_options_td3(self):
        """Test the parse_options and get_hyperparameters methods for TD3.

        This is done for the following cases:

        1. hierarchical = False, multiagent = False
           a. default arguments
           b. custom  arguments
        2. hierarchical = True,  multiagent = False
           a. default arguments
           b. custom  arguments
        3. hierarchical = False, multiagent = True
           a. default arguments
           b. custom  arguments
        4. hierarchical = True,  multiagent = True
           a. default arguments
           b. custom  arguments
        """
        self.maxDiff = None
        model_params = FEEDFORWARD_PARAMS["model_params"]

        # =================================================================== #
        # test case 1.a                                                       #
        # =================================================================== #

        args = parse_options(
            "", "", args=["AntMaze"], multiagent=False, hierarchical=False)
        self.assertDictEqual(vars(args), {
            'env_name': 'AntMaze',
            'alg': 'TD3',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_dir': None,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'save_replay_buffer': False,
            'num_envs': 1,
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:layers': None,
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': False,
            'model_params:model_type': 'fcnet',
            'model_params:strides': None,
            'use_huber': False,
            'noise': TD3_PARAMS['noise'],
            'target_policy_noise': TD3_PARAMS['target_policy_noise'],
            'target_noise_clip': TD3_PARAMS['target_noise_clip'],
            'buffer_size': TD3_PARAMS['buffer_size'],
            'batch_size': TD3_PARAMS['batch_size'],
            'actor_lr': TD3_PARAMS['actor_lr'],
            'critic_lr': TD3_PARAMS['critic_lr'],
            'tau': TD3_PARAMS['tau'],
            'gamma': TD3_PARAMS['gamma'],
        })

        hp = get_hyperparameters(args, TD3FeedForwardPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 1000000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'num_envs': 1,
            'save_replay_buffer': False,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': TD3_PARAMS['buffer_size'],
                'batch_size': TD3_PARAMS['batch_size'],
                'actor_lr': TD3_PARAMS['actor_lr'],
                'critic_lr': TD3_PARAMS['critic_lr'],
                'tau': TD3_PARAMS['tau'],
                'gamma': TD3_PARAMS['gamma'],
                'noise': TD3_PARAMS['noise'],
                'target_policy_noise': TD3_PARAMS['target_policy_noise'],
                'target_noise_clip': TD3_PARAMS['target_noise_clip'],
                'use_huber': TD3_PARAMS['use_huber'],
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'model_type': model_params["model_type"],
                    'layers': model_params["layers"],
                    'layer_norm': model_params["layer_norm"],
                    'filters': model_params["filters"],
                    'ignore_flat_channels': model_params[
                        "ignore_flat_channels"],
                    'ignore_image': model_params["ignore_image"],
                    'image_channels': model_params["image_channels"],
                    'image_height': model_params["image_height"],
                    'image_width': model_params["image_width"],
                    'kernel_sizes': model_params["kernel_sizes"],
                    'strides': model_params["strides"],
                },
            }
        })

        # =================================================================== #
        # test case 1.b                                                       #
        # =================================================================== #

        args = parse_options(
            "", "",
            args=[
                "AntMaze",
                '--evaluate',
                '--save_replay_buffer',
                '--n_training', '1',
                '--total_steps', '2',
                '--seed', '3',
                '--log_dir', 'custom_dir',
                '--log_interval', '4',
                '--eval_interval', '5',
                '--save_interval', '6',
                '--nb_train_steps', '7',
                '--nb_rollout_steps', '8',
                '--nb_eval_episodes', '9',
                '--reward_scale', '10',
                '--render',
                '--render_eval',
                '--verbose', '11',
                '--actor_update_freq', '12',
                '--meta_update_freq', '13',
                '--buffer_size', '14',
                '--batch_size', '15',
                '--actor_lr', '16',
                '--critic_lr', '17',
                '--tau', '18',
                '--gamma', '19',
                '--noise', '20',
                '--num_envs', '21',
                '--target_policy_noise', '22',
                '--target_noise_clip', '23',
                '--use_huber',
                '--l2_penalty', '1',
                '--model_params:model_type', 'model_type',
                '--model_params:layers', '24', '25',
                '--model_params:layer_norm',
            ],
            multiagent=False,
            hierarchical=False,
        )
        self.assertDictEqual(vars(args), {
            'actor_lr': 16.0,
            'actor_update_freq': 12,
            'alg': 'TD3',
            'batch_size': 15,
            'buffer_size': 14,
            'critic_lr': 17.0,
            'env_name': 'AntMaze',
            'eval_interval': 5,
            'evaluate': True,
            'gamma': 19.0,
            'initial_exploration_steps': 10000,
            'log_dir': 'custom_dir',
            'log_interval': 4,
            'meta_update_freq': 13,
            'l2_penalty': 1,
            'model_params:layers': [24, 25],
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': True,
            'model_params:model_type': 'model_type',
            'model_params:strides': None,
            'n_training': 1,
            'nb_eval_episodes': 9,
            'nb_rollout_steps': 8,
            'nb_train_steps': 7,
            'noise': 20.0,
            'num_envs': 21,
            'render': True,
            'render_eval': True,
            'reward_scale': 10.0,
            'save_interval': 6,
            'save_replay_buffer': True,
            'seed': 3,
            'target_noise_clip': 23.0,
            'target_policy_noise': 22.0,
            'tau': 18.0,
            'total_steps': 2,
            'use_huber': True,
            'verbose': 11,
        })

        hp = get_hyperparameters(args, TD3FeedForwardPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 2,
            '_init_setup_model': True,
            'render': True,
            'render_eval': True,
            'reward_scale': 10.0,
            'save_replay_buffer': True,
            'verbose': 11,
            'actor_update_freq': 12,
            'meta_update_freq': 13,
            'nb_eval_episodes': 9,
            'nb_rollout_steps': 8,
            'nb_train_steps': 7,
            'num_envs': 21,
            'policy_kwargs': {
                'actor_lr': 16.0,
                'batch_size': 15,
                'buffer_size': 14,
                'critic_lr': 17.0,
                'gamma': 19.0,
                'l2_penalty': 1,
                'model_params': {
                    'layers': [24, 25],
                    'filters': [16, 16, 16],
                    'ignore_flat_channels': [],
                    'ignore_image': False,
                    'image_channels': 3,
                    'image_height': 32,
                    'image_width': 32,
                    'kernel_sizes': [5, 5, 5],
                    'layer_norm': True,
                    'model_type': 'model_type',
                    'strides': [2, 2, 2]
                },
                'noise': 20.0,
                'target_noise_clip': 23.0,
                'target_policy_noise': 22.0,
                'tau': 18.0,
                'use_huber': True
            },
        })

        # =================================================================== #
        # test case 2.a                                                       #
        # =================================================================== #

        args = parse_options(
            "", "", args=["AntMaze"], multiagent=False, hierarchical=True)
        self.assertDictEqual(vars(args), {
            'env_name': 'AntMaze',
            'alg': 'TD3',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_dir': None,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'save_replay_buffer': False,
            'num_envs': 1,
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:layers': None,
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': False,
            'model_params:model_type': 'fcnet',
            'model_params:strides': None,
            'use_huber': False,
            'noise': TD3_PARAMS['noise'],
            'target_policy_noise': TD3_PARAMS['target_policy_noise'],
            'target_noise_clip': TD3_PARAMS['target_noise_clip'],
            'buffer_size': TD3_PARAMS['buffer_size'],
            'batch_size': TD3_PARAMS['batch_size'],
            'actor_lr': TD3_PARAMS['actor_lr'],
            'critic_lr': TD3_PARAMS['critic_lr'],
            'tau': TD3_PARAMS['tau'],
            'gamma': TD3_PARAMS['gamma'],
            'cg_weights': GOAL_CONDITIONED_PARAMS['cg_weights'],
            'cg_delta': GOAL_CONDITIONED_PARAMS['cg_delta'],
            'cooperative_gradients': False,
            'pretrain_ckpt': GOAL_CONDITIONED_PARAMS['pretrain_ckpt'],
            'pretrain_path': GOAL_CONDITIONED_PARAMS['pretrain_path'],
            'pretrain_worker': False,
            'hindsight': False,
            'intrinsic_reward_scale': GOAL_CONDITIONED_PARAMS[
                'intrinsic_reward_scale'],
            'intrinsic_reward_type': GOAL_CONDITIONED_PARAMS[
                'intrinsic_reward_type'],
            'meta_period': GOAL_CONDITIONED_PARAMS['meta_period'],
            'num_levels': GOAL_CONDITIONED_PARAMS['num_levels'],
            'off_policy_corrections': False,
            'relative_goals': False,
            'subgoal_testing_rate': GOAL_CONDITIONED_PARAMS[
                'subgoal_testing_rate'],
        })

        hp = get_hyperparameters(args, TD3GoalConditionedPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 1000000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'num_envs': 1,
            'save_replay_buffer': False,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': TD3_PARAMS['buffer_size'],
                'batch_size': TD3_PARAMS['batch_size'],
                'actor_lr': TD3_PARAMS['actor_lr'],
                'critic_lr': TD3_PARAMS['critic_lr'],
                'tau': TD3_PARAMS['tau'],
                'gamma': TD3_PARAMS['gamma'],
                'noise': TD3_PARAMS['noise'],
                'target_policy_noise': TD3_PARAMS['target_policy_noise'],
                'target_noise_clip': TD3_PARAMS['target_noise_clip'],
                'use_huber': TD3_PARAMS['use_huber'],
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'model_type': model_params["model_type"],
                    'layers': model_params["layers"],
                    'layer_norm': model_params["layer_norm"],
                    'filters': model_params["filters"],
                    'ignore_flat_channels': model_params[
                        "ignore_flat_channels"],
                    'ignore_image': model_params["ignore_image"],
                    'image_channels': model_params["image_channels"],
                    'image_height': model_params["image_height"],
                    'image_width': model_params["image_width"],
                    'kernel_sizes': model_params["kernel_sizes"],
                    'strides': model_params["strides"],
                },
                'cg_weights': GOAL_CONDITIONED_PARAMS['cg_weights'],
                'cg_delta': GOAL_CONDITIONED_PARAMS['cg_delta'],
                'cooperative_gradients': False,
                'pretrain_ckpt': GOAL_CONDITIONED_PARAMS['pretrain_ckpt'],
                'pretrain_path': GOAL_CONDITIONED_PARAMS['pretrain_path'],
                'pretrain_worker': False,
                'hindsight': False,
                'intrinsic_reward_scale': GOAL_CONDITIONED_PARAMS[
                    'intrinsic_reward_scale'],
                'intrinsic_reward_type': GOAL_CONDITIONED_PARAMS[
                    'intrinsic_reward_type'],
                'meta_period': GOAL_CONDITIONED_PARAMS['meta_period'],
                'num_levels': GOAL_CONDITIONED_PARAMS['num_levels'],
                'off_policy_corrections': False,
                'relative_goals': False,
                'subgoal_testing_rate': GOAL_CONDITIONED_PARAMS[
                    'subgoal_testing_rate'],
            }
        })

        # =================================================================== #
        # test case 2.b                                                       #
        # =================================================================== #

        args = parse_options(
            "", "",
            args=[
                "AntMaze",
                "--num_levels", "1",
                "--meta_period", "2",
                "--intrinsic_reward_type", "3",
                "--intrinsic_reward_scale", "4",
                "--relative_goals",
                "--off_policy_corrections",
                "--hindsight",
                "--subgoal_testing_rate", "6",
                "--cooperative_gradients",
                "--cg_weights", "7",
                "--cg_delta", "10",
                "--pretrain_ckpt", "8",
                "--pretrain_path", "9",
                "--pretrain_worker",
            ],
            multiagent=False,
            hierarchical=True,
        )

        self.assertDictEqual(vars(args), {
            'env_name': 'AntMaze',
            'alg': 'TD3',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_dir': None,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'save_replay_buffer': False,
            'num_envs': 1,
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:layers': None,
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': False,
            'model_params:model_type': 'fcnet',
            'model_params:strides': None,
            'use_huber': False,
            'noise': TD3_PARAMS['noise'],
            'target_policy_noise': TD3_PARAMS['target_policy_noise'],
            'target_noise_clip': TD3_PARAMS['target_noise_clip'],
            'buffer_size': TD3_PARAMS['buffer_size'],
            'batch_size': TD3_PARAMS['batch_size'],
            'actor_lr': TD3_PARAMS['actor_lr'],
            'critic_lr': TD3_PARAMS['critic_lr'],
            'tau': TD3_PARAMS['tau'],
            'gamma': TD3_PARAMS['gamma'],
            'cg_weights': 7,
            'cg_delta': 10,
            'cooperative_gradients': True,
            'pretrain_ckpt': 8,
            'pretrain_path': "9",
            'pretrain_worker': True,
            'hindsight': True,
            'intrinsic_reward_scale': 4,
            'intrinsic_reward_type': "3",
            'meta_period': 2,
            'num_levels': 1,
            'off_policy_corrections': True,
            'relative_goals': True,
            'subgoal_testing_rate': 6,
        })

        hp = get_hyperparameters(args, TD3GoalConditionedPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 1000000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'num_envs': 1,
            'save_replay_buffer': False,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': TD3_PARAMS['buffer_size'],
                'batch_size': TD3_PARAMS['batch_size'],
                'actor_lr': TD3_PARAMS['actor_lr'],
                'critic_lr': TD3_PARAMS['critic_lr'],
                'tau': TD3_PARAMS['tau'],
                'gamma': TD3_PARAMS['gamma'],
                'noise': TD3_PARAMS['noise'],
                'target_policy_noise': TD3_PARAMS['target_policy_noise'],
                'target_noise_clip': TD3_PARAMS['target_noise_clip'],
                'use_huber': TD3_PARAMS['use_huber'],
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'model_type': model_params["model_type"],
                    'layers': model_params["layers"],
                    'layer_norm': model_params["layer_norm"],
                    'filters': model_params["filters"],
                    'ignore_flat_channels': model_params[
                        "ignore_flat_channels"],
                    'ignore_image': model_params["ignore_image"],
                    'image_channels': model_params["image_channels"],
                    'image_height': model_params["image_height"],
                    'image_width': model_params["image_width"],
                    'kernel_sizes': model_params["kernel_sizes"],
                    'strides': model_params["strides"],
                },
                'cg_weights': 7,
                'cg_delta': 10,
                'cooperative_gradients': True,
                'pretrain_ckpt': 8,
                'pretrain_path': "9",
                'pretrain_worker': True,
                'hindsight': True,
                'intrinsic_reward_scale': 4,
                'intrinsic_reward_type': "3",
                'meta_period': 2,
                'num_levels': 1,
                'off_policy_corrections': True,
                'relative_goals': True,
                'subgoal_testing_rate': 6,
            }
        })

        # =================================================================== #
        # test case 3.a                                                       #
        # =================================================================== #

        args = parse_options(
            "", "", args=["AntMaze"], multiagent=True, hierarchical=False)
        self.assertDictEqual(vars(args), {
            'env_name': 'AntMaze',
            'alg': 'TD3',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_dir': None,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'save_replay_buffer': False,
            'num_envs': 1,
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:layers': None,
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': False,
            'model_params:model_type': 'fcnet',
            'model_params:strides': None,
            'use_huber': False,
            'noise': TD3_PARAMS['noise'],
            'target_policy_noise': TD3_PARAMS['target_policy_noise'],
            'target_noise_clip': TD3_PARAMS['target_noise_clip'],
            'buffer_size': TD3_PARAMS['buffer_size'],
            'batch_size': TD3_PARAMS['batch_size'],
            'actor_lr': TD3_PARAMS['actor_lr'],
            'critic_lr': TD3_PARAMS['critic_lr'],
            'tau': TD3_PARAMS['tau'],
            'gamma': TD3_PARAMS['gamma'],
            'shared': False,
            'maddpg': False,
            'n_agents': MULTIAGENT_PARAMS["n_agents"],
        })

        hp = get_hyperparameters(args, TD3MultiFeedForwardPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 1000000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'num_envs': 1,
            'save_replay_buffer': False,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': TD3_PARAMS['buffer_size'],
                'batch_size': TD3_PARAMS['batch_size'],
                'actor_lr': TD3_PARAMS['actor_lr'],
                'critic_lr': TD3_PARAMS['critic_lr'],
                'tau': TD3_PARAMS['tau'],
                'gamma': TD3_PARAMS['gamma'],
                'noise': TD3_PARAMS['noise'],
                'target_policy_noise': TD3_PARAMS['target_policy_noise'],
                'target_noise_clip': TD3_PARAMS['target_noise_clip'],
                'use_huber': TD3_PARAMS['use_huber'],
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'model_type': model_params["model_type"],
                    'layers': model_params["layers"],
                    'layer_norm': model_params["layer_norm"],
                    'filters': model_params["filters"],
                    'ignore_flat_channels': model_params[
                        "ignore_flat_channels"],
                    'ignore_image': model_params["ignore_image"],
                    'image_channels': model_params["image_channels"],
                    'image_height': model_params["image_height"],
                    'image_width': model_params["image_width"],
                    'kernel_sizes': model_params["kernel_sizes"],
                    'strides': model_params["strides"],
                },
                'shared': False,
                'maddpg': False,
                'n_agents': MULTIAGENT_PARAMS["n_agents"],
            }
        })

        # =================================================================== #
        # test case 3.b                                                       #
        # =================================================================== #

        args = parse_options(
            "", "",
            args=["AntMaze", "--shared", "--maddpg", "--n_agents", "2"],
            multiagent=True,
            hierarchical=False,
        )

        self.assertDictEqual(vars(args), {
            'env_name': 'AntMaze',
            'alg': 'TD3',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_dir': None,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'save_replay_buffer': False,
            'num_envs': 1,
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:layers': None,
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': False,
            'model_params:model_type': 'fcnet',
            'model_params:strides': None,
            'use_huber': False,
            'noise': TD3_PARAMS['noise'],
            'target_policy_noise': TD3_PARAMS['target_policy_noise'],
            'target_noise_clip': TD3_PARAMS['target_noise_clip'],
            'buffer_size': TD3_PARAMS['buffer_size'],
            'batch_size': TD3_PARAMS['batch_size'],
            'actor_lr': TD3_PARAMS['actor_lr'],
            'critic_lr': TD3_PARAMS['critic_lr'],
            'tau': TD3_PARAMS['tau'],
            'gamma': TD3_PARAMS['gamma'],
            'shared': True,
            'maddpg': True,
            'n_agents': 2,
        })

        hp = get_hyperparameters(args, TD3MultiFeedForwardPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 1000000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'num_envs': 1,
            'save_replay_buffer': False,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': TD3_PARAMS['buffer_size'],
                'batch_size': TD3_PARAMS['batch_size'],
                'actor_lr': TD3_PARAMS['actor_lr'],
                'critic_lr': TD3_PARAMS['critic_lr'],
                'tau': TD3_PARAMS['tau'],
                'gamma': TD3_PARAMS['gamma'],
                'noise': TD3_PARAMS['noise'],
                'target_policy_noise': TD3_PARAMS['target_policy_noise'],
                'target_noise_clip': TD3_PARAMS['target_noise_clip'],
                'use_huber': TD3_PARAMS['use_huber'],
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'model_type': model_params["model_type"],
                    'layers': model_params["layers"],
                    'layer_norm': model_params["layer_norm"],
                    'filters': model_params["filters"],
                    'ignore_flat_channels': model_params[
                        "ignore_flat_channels"],
                    'ignore_image': model_params["ignore_image"],
                    'image_channels': model_params["image_channels"],
                    'image_height': model_params["image_height"],
                    'image_width': model_params["image_width"],
                    'kernel_sizes': model_params["kernel_sizes"],
                    'strides': model_params["strides"],
                },
                'shared': True,
                'maddpg': True,
                'n_agents': 2,
            }
        })

        # =================================================================== #
        # test case 4.a                                                       #
        # =================================================================== #

        args = parse_options(
            "", "", args=["AntMaze"], multiagent=True, hierarchical=True)
        self.assertDictEqual(vars(args), {
            'env_name': 'AntMaze',
            'alg': 'TD3',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_dir': None,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'save_replay_buffer': False,
            'num_envs': 1,
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:layers': None,
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': False,
            'model_params:model_type': 'fcnet',
            'model_params:strides': None,
            'use_huber': False,
            'noise': TD3_PARAMS['noise'],
            'target_policy_noise': TD3_PARAMS['target_policy_noise'],
            'target_noise_clip': TD3_PARAMS['target_noise_clip'],
            'buffer_size': TD3_PARAMS['buffer_size'],
            'batch_size': TD3_PARAMS['batch_size'],
            'actor_lr': TD3_PARAMS['actor_lr'],
            'critic_lr': TD3_PARAMS['critic_lr'],
            'tau': TD3_PARAMS['tau'],
            'gamma': TD3_PARAMS['gamma'],
            'cg_weights': GOAL_CONDITIONED_PARAMS['cg_weights'],
            'cg_delta': GOAL_CONDITIONED_PARAMS['cg_delta'],
            'cooperative_gradients': False,
            'pretrain_ckpt': GOAL_CONDITIONED_PARAMS['pretrain_ckpt'],
            'pretrain_path': GOAL_CONDITIONED_PARAMS['pretrain_path'],
            'pretrain_worker': False,
            'hindsight': False,
            'intrinsic_reward_scale': GOAL_CONDITIONED_PARAMS[
                'intrinsic_reward_scale'],
            'intrinsic_reward_type': GOAL_CONDITIONED_PARAMS[
                'intrinsic_reward_type'],
            'meta_period': GOAL_CONDITIONED_PARAMS['meta_period'],
            'num_levels': GOAL_CONDITIONED_PARAMS['num_levels'],
            'off_policy_corrections': False,
            'relative_goals': False,
            'subgoal_testing_rate': GOAL_CONDITIONED_PARAMS[
                'subgoal_testing_rate'],
            'shared': False,
            'maddpg': False,
            'n_agents': MULTIAGENT_PARAMS["n_agents"],
        })

        hp = get_hyperparameters(args, TD3MultiFeedForwardPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 1000000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'num_envs': 1,
            'save_replay_buffer': False,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': TD3_PARAMS['buffer_size'],
                'batch_size': TD3_PARAMS['batch_size'],
                'actor_lr': TD3_PARAMS['actor_lr'],
                'critic_lr': TD3_PARAMS['critic_lr'],
                'tau': TD3_PARAMS['tau'],
                'gamma': TD3_PARAMS['gamma'],
                'noise': TD3_PARAMS['noise'],
                'target_policy_noise': TD3_PARAMS['target_policy_noise'],
                'target_noise_clip': TD3_PARAMS['target_noise_clip'],
                'use_huber': TD3_PARAMS['use_huber'],
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'model_type': model_params["model_type"],
                    'layers': model_params["layers"],
                    'layer_norm': model_params["layer_norm"],
                    'filters': model_params["filters"],
                    'ignore_flat_channels': model_params[
                        "ignore_flat_channels"],
                    'ignore_image': model_params["ignore_image"],
                    'image_channels': model_params["image_channels"],
                    'image_height': model_params["image_height"],
                    'image_width': model_params["image_width"],
                    'kernel_sizes': model_params["kernel_sizes"],
                    'strides': model_params["strides"],
                },
                'shared': False,
                'maddpg': False,
                'n_agents': MULTIAGENT_PARAMS["n_agents"],
            }
        })

        # =================================================================== #
        # test case 4.b                                                       #
        # =================================================================== #

        args = parse_options(
            "", "",
            args=[
                "AntMaze",
                "--num_levels", "1",
                "--meta_period", "2",
                "--intrinsic_reward_type", "3",
                "--intrinsic_reward_scale", "4",
                "--relative_goals",
                "--off_policy_corrections",
                "--hindsight",
                "--subgoal_testing_rate", "6",
                "--cooperative_gradients",
                "--cg_weights", "7",
                "--cg_delta", "9",
                "--shared",
                "--maddpg",
                "--n_agents", "8",
            ],
            multiagent=True,
            hierarchical=True,
        )

        self.assertDictEqual(vars(args), {
            'env_name': 'AntMaze',
            'alg': 'TD3',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_dir': None,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'save_replay_buffer': False,
            'num_envs': 1,
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:layers': None,
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': False,
            'model_params:model_type': 'fcnet',
            'model_params:strides': None,
            'use_huber': False,
            'noise': TD3_PARAMS['noise'],
            'target_policy_noise': TD3_PARAMS['target_policy_noise'],
            'target_noise_clip': TD3_PARAMS['target_noise_clip'],
            'buffer_size': TD3_PARAMS['buffer_size'],
            'batch_size': TD3_PARAMS['batch_size'],
            'actor_lr': TD3_PARAMS['actor_lr'],
            'critic_lr': TD3_PARAMS['critic_lr'],
            'tau': TD3_PARAMS['tau'],
            'gamma': TD3_PARAMS['gamma'],
            'cg_weights': 7,
            'cg_delta': 9,
            'cooperative_gradients': True,
            'pretrain_ckpt': GOAL_CONDITIONED_PARAMS['pretrain_ckpt'],
            'pretrain_path': GOAL_CONDITIONED_PARAMS['pretrain_path'],
            'pretrain_worker': False,
            'hindsight': True,
            'intrinsic_reward_scale': 4,
            'intrinsic_reward_type': "3",
            'meta_period': 2,
            'num_levels': 1,
            'off_policy_corrections': True,
            'relative_goals': True,
            'subgoal_testing_rate': 6,
            'shared': True,
            'maddpg': True,
            'n_agents': 8,
        })

        hp = get_hyperparameters(args, TD3MultiGoalConditionedPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 1000000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'num_envs': 1,
            'save_replay_buffer': False,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': TD3_PARAMS['buffer_size'],
                'batch_size': TD3_PARAMS['batch_size'],
                'actor_lr': TD3_PARAMS['actor_lr'],
                'critic_lr': TD3_PARAMS['critic_lr'],
                'tau': TD3_PARAMS['tau'],
                'gamma': TD3_PARAMS['gamma'],
                'noise': TD3_PARAMS['noise'],
                'target_policy_noise': TD3_PARAMS['target_policy_noise'],
                'target_noise_clip': TD3_PARAMS['target_noise_clip'],
                'use_huber': TD3_PARAMS['use_huber'],
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'model_type': model_params["model_type"],
                    'layers': model_params["layers"],
                    'layer_norm': model_params["layer_norm"],
                    'filters': model_params["filters"],
                    'ignore_flat_channels': model_params[
                        "ignore_flat_channels"],
                    'ignore_image': model_params["ignore_image"],
                    'image_channels': model_params["image_channels"],
                    'image_height': model_params["image_height"],
                    'image_width': model_params["image_width"],
                    'kernel_sizes': model_params["kernel_sizes"],
                    'strides': model_params["strides"],
                },
                'cg_weights': 7,
                'cg_delta': 9,
                'cooperative_gradients': True,
                'pretrain_ckpt': GOAL_CONDITIONED_PARAMS['pretrain_ckpt'],
                'pretrain_path': GOAL_CONDITIONED_PARAMS['pretrain_path'],
                'pretrain_worker': False,
                'hindsight': True,
                'intrinsic_reward_scale': 4,
                'intrinsic_reward_type': "3",
                'meta_period': 2,
                'num_levels': 1,
                'off_policy_corrections': True,
                'relative_goals': True,
                'subgoal_testing_rate': 6,
                'shared': True,
                'maddpg': True,
                'n_agents': 8,
            }
        })

    def test_parse_options_sac(self):
        """Test the parse_options and get_hyperparameters methods for SAC.

        This is done for the following cases:

        1. hierarchical = False, multiagent = False
           a. default arguments
           b. custom  arguments
        2. hierarchical = True,  multiagent = False
           a. default arguments
           b. custom  arguments
        3. hierarchical = False, multiagent = True
           a. default arguments
           b. custom  arguments
        4. hierarchical = True,  multiagent = True
           a. default arguments
           b. custom  arguments
        """
        self.maxDiff = None
        model_params = FEEDFORWARD_PARAMS["model_params"]

        # =================================================================== #
        # test case 1.a                                                       #
        # =================================================================== #

        args = parse_options(
            "", "", args=["AntMaze", "--alg", "SAC"],
            multiagent=False, hierarchical=False)
        self.assertDictEqual(vars(args), {
            'env_name': 'AntMaze',
            'alg': 'SAC',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_dir': None,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'save_replay_buffer': False,
            'num_envs': 1,
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:layers': None,
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': False,
            'model_params:model_type': 'fcnet',
            'model_params:strides': None,
            'use_huber': False,
            'target_entropy': SAC_PARAMS['target_entropy'],
            'buffer_size': SAC_PARAMS['buffer_size'],
            'batch_size': SAC_PARAMS['batch_size'],
            'actor_lr': SAC_PARAMS['actor_lr'],
            'critic_lr': SAC_PARAMS['critic_lr'],
            'tau': SAC_PARAMS['tau'],
            'gamma': SAC_PARAMS['gamma'],
        })

        hp = get_hyperparameters(args, SACFeedForwardPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 1000000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'num_envs': 1,
            'save_replay_buffer': False,
            '_init_setup_model': True,
            'policy_kwargs': {
                'buffer_size': SAC_PARAMS['buffer_size'],
                'batch_size': SAC_PARAMS['batch_size'],
                'actor_lr': SAC_PARAMS['actor_lr'],
                'critic_lr': SAC_PARAMS['critic_lr'],
                'tau': SAC_PARAMS['tau'],
                'gamma': SAC_PARAMS['gamma'],
                'target_entropy': SAC_PARAMS['target_entropy'],
                'use_huber': SAC_PARAMS['use_huber'],
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'model_type': model_params["model_type"],
                    'layers': model_params["layers"],
                    'layer_norm': model_params["layer_norm"],
                    'filters': model_params["filters"],
                    'ignore_flat_channels': model_params[
                        "ignore_flat_channels"],
                    'ignore_image': model_params["ignore_image"],
                    'image_channels': model_params["image_channels"],
                    'image_height': model_params["image_height"],
                    'image_width': model_params["image_width"],
                    'kernel_sizes': model_params["kernel_sizes"],
                    'strides': model_params["strides"],
                },
            }
        })

        # =================================================================== #
        # test case 1.b                                                       #
        # =================================================================== #

        args = parse_options(
            "", "",
            args=[
                "AntMaze",
                "--alg", "SAC",
                '--evaluate',
                '--save_replay_buffer',
                '--n_training', '1',
                '--total_steps', '2',
                '--seed', '3',
                '--log_dir', 'custom_dir',
                '--log_interval', '4',
                '--eval_interval', '5',
                '--save_interval', '6',
                '--nb_train_steps', '7',
                '--nb_rollout_steps', '8',
                '--nb_eval_episodes', '9',
                '--reward_scale', '10',
                '--render',
                '--render_eval',
                '--verbose', '11',
                '--actor_update_freq', '12',
                '--meta_update_freq', '13',
                '--buffer_size', '14',
                '--batch_size', '15',
                '--actor_lr', '16',
                '--critic_lr', '17',
                '--tau', '18',
                '--gamma', '19',
                '--target_entropy', '20',
                '--num_envs', '21',
                '--use_huber',
                '--model_params:model_type', 'model_type',
                '--model_params:layer_norm',
                '--model_params:layers', '22', '23',
            ],
            multiagent=False,
            hierarchical=False,
        )
        self.assertDictEqual(vars(args), {
            'actor_lr': 16.0,
            'actor_update_freq': 12,
            'alg': 'SAC',
            'batch_size': 15,
            'buffer_size': 14,
            'critic_lr': 17.0,
            'env_name': 'AntMaze',
            'eval_interval': 5,
            'evaluate': True,
            'gamma': 19.0,
            'initial_exploration_steps': 10000,
            'log_dir': 'custom_dir',
            'log_interval': 4,
            'meta_update_freq': 13,
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:layers': [22, 23],
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': True,
            'model_params:model_type': 'model_type',
            'model_params:strides': None,
            'n_training': 1,
            'nb_eval_episodes': 9,
            'nb_rollout_steps': 8,
            'nb_train_steps': 7,
            'target_entropy': 20.0,
            'num_envs': 21,
            'render': True,
            'render_eval': True,
            'reward_scale': 10.0,
            'save_interval': 6,
            'save_replay_buffer': True,
            'seed': 3,
            'tau': 18.0,
            'total_steps': 2,
            'use_huber': True,
            'verbose': 11,
        })

        hp = get_hyperparameters(args, SACFeedForwardPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 2,
            '_init_setup_model': True,
            'render': True,
            'render_eval': True,
            'reward_scale': 10.0,
            'save_replay_buffer': True,
            'verbose': 11,
            'actor_update_freq': 12,
            'meta_update_freq': 13,
            'nb_eval_episodes': 9,
            'nb_rollout_steps': 8,
            'nb_train_steps': 7,
            'num_envs': 21,
            'policy_kwargs': {
                'actor_lr': 16.0,
                'batch_size': 15,
                'buffer_size': 14,
                'critic_lr': 17.0,
                'gamma': 19.0,
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'layers': [22, 23],
                    'filters': [16, 16, 16],
                    'ignore_flat_channels': [],
                    'ignore_image': False,
                    'image_channels': 3,
                    'image_height': 32,
                    'image_width': 32,
                    'kernel_sizes': [5, 5, 5],
                    'layer_norm': True,
                    'model_type': 'model_type',
                    'strides': [2, 2, 2]
                },
                'target_entropy': 20.0,
                'tau': 18.0,
                'use_huber': True
            },
        })

    def test_parse_options_PPO(self):
        """Test the parse_options and get_hyperparameters methods for PPO.

        This is done for the following cases:

        1. hierarchical = False, multiagent = False
           a. default arguments
           b. custom  arguments

        All other variants should work as well (tested by a different methods).
        """
        self.maxDiff = None
        model_params = FEEDFORWARD_PARAMS["model_params"]

        # =================================================================== #
        # test case 1.a                                                       #
        # =================================================================== #

        args = parse_options(
            "", "", args=["AntMaze", "--alg", "PPO"],
            multiagent=False, hierarchical=False)
        self.assertDictEqual(vars(args), {
            'env_name': 'AntMaze',
            'alg': 'PPO',
            'evaluate': False,
            'n_training': 1,
            'total_steps': 1000000,
            'seed': 1,
            'log_dir': None,
            'log_interval': 2000,
            'eval_interval': 50000,
            'save_interval': 50000,
            'initial_exploration_steps': 10000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'save_replay_buffer': False,
            'num_envs': 1,
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:layers': None,
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': False,
            'model_params:model_type': 'fcnet',
            'model_params:strides': None,
            'cliprange': PPO_PARAMS['cliprange'],
            'cliprange_vf': PPO_PARAMS['cliprange_vf'],
            'ent_coef': PPO_PARAMS['ent_coef'],
            'gamma': PPO_PARAMS['gamma'],
            'lam': PPO_PARAMS['lam'],
            'learning_rate': PPO_PARAMS['learning_rate'],
            'max_grad_norm': PPO_PARAMS['max_grad_norm'],
            'n_minibatches': PPO_PARAMS['n_minibatches'],
            'n_opt_epochs': PPO_PARAMS['n_opt_epochs'],
            'vf_coef': PPO_PARAMS['vf_coef'],
        })

        hp = get_hyperparameters(args, PPOFeedForwardPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 1000000,
            'nb_train_steps': 1,
            'nb_rollout_steps': 1,
            'nb_eval_episodes': 50,
            'reward_scale': 1,
            'render': False,
            'render_eval': False,
            'verbose': 2,
            'actor_update_freq': 2,
            'meta_update_freq': 10,
            'num_envs': 1,
            'save_replay_buffer': False,
            '_init_setup_model': True,
            'policy_kwargs': {
                'cliprange': PPO_PARAMS['cliprange'],
                'cliprange_vf': PPO_PARAMS['cliprange_vf'],
                'ent_coef': PPO_PARAMS['ent_coef'],
                'gamma': PPO_PARAMS['gamma'],
                'lam': PPO_PARAMS['lam'],
                'learning_rate': PPO_PARAMS['learning_rate'],
                'max_grad_norm': PPO_PARAMS['max_grad_norm'],
                'n_minibatches': PPO_PARAMS['n_minibatches'],
                'n_opt_epochs': PPO_PARAMS['n_opt_epochs'],
                'vf_coef': PPO_PARAMS['vf_coef'],
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'model_type': model_params["model_type"],
                    'layers': model_params["layers"],
                    'layer_norm': model_params["layer_norm"],
                    'filters': model_params["filters"],
                    'ignore_flat_channels': model_params[
                        "ignore_flat_channels"],
                    'ignore_image': model_params["ignore_image"],
                    'image_channels': model_params["image_channels"],
                    'image_height': model_params["image_height"],
                    'image_width': model_params["image_width"],
                    'kernel_sizes': model_params["kernel_sizes"],
                    'strides': model_params["strides"],
                },
            }
        })

        # =================================================================== #
        # test case 1.b                                                       #
        # =================================================================== #

        args = parse_options(
            "", "",
            args=[
                "AntMaze",
                "--alg", "PPO",
                '--evaluate',
                '--save_replay_buffer',
                '--n_training', '1',
                '--total_steps', '2',
                '--seed', '3',
                '--log_dir', 'custom_dir',
                '--log_interval', '4',
                '--eval_interval', '5',
                '--save_interval', '6',
                '--nb_train_steps', '7',
                '--nb_rollout_steps', '8',
                '--nb_eval_episodes', '9',
                '--reward_scale', '10',
                '--render',
                '--render_eval',
                '--verbose', '11',
                '--actor_update_freq', '12',
                '--meta_update_freq', '13',
                '--num_envs', '21',
                '--model_params:model_type', 'model_type',
                '--model_params:layer_norm',
                '--model_params:layers', '22', '23',
                '--cliprange', '24',
                '--cliprange_vf', '25',
                '--ent_coef', '26',
                '--gamma', '27',
                '--lam', '28',
                '--learning_rate', '29',
                '--max_grad_norm', '30',
                '--n_minibatches', '31',
                '--n_opt_epochs', '32',
                '--vf_coef', '33',
            ],
            multiagent=False,
            hierarchical=False,
        )
        self.assertDictEqual(vars(args), {
            'actor_update_freq': 12,
            'alg': 'PPO',
            'env_name': 'AntMaze',
            'eval_interval': 5,
            'evaluate': True,
            'initial_exploration_steps': 10000,
            'log_dir': 'custom_dir',
            'log_interval': 4,
            'meta_update_freq': 13,
            'model_params:layers': [22, 23],
            'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
            'model_params:filters': None,
            'model_params:ignore_flat_channels': None,
            'model_params:ignore_image': False,
            'model_params:image_channels': 3,
            'model_params:image_height': 32,
            'model_params:image_width': 32,
            'model_params:kernel_sizes': None,
            'model_params:layer_norm': True,
            'model_params:model_type': 'model_type',
            'model_params:strides': None,
            'n_training': 1,
            'nb_eval_episodes': 9,
            'nb_rollout_steps': 8,
            'nb_train_steps': 7,
            'num_envs': 21,
            'render': True,
            'render_eval': True,
            'reward_scale': 10.0,
            'save_interval': 6,
            'save_replay_buffer': True,
            'seed': 3,
            'total_steps': 2,
            'verbose': 11,
            'cliprange': 24.0,
            'cliprange_vf': 25.0,
            'ent_coef': 26.0,
            'gamma': 27.0,
            'lam': 28.0,
            'learning_rate': 29.0,
            'max_grad_norm': 30.0,
            'n_minibatches': 31,
            'n_opt_epochs': 32,
            'vf_coef': 33.0,
        })

        hp = get_hyperparameters(args, PPOFeedForwardPolicy)
        self.assertDictEqual(hp, {
            'total_steps': 2,
            '_init_setup_model': True,
            'render': True,
            'render_eval': True,
            'reward_scale': 10.0,
            'save_replay_buffer': True,
            'verbose': 11,
            'actor_update_freq': 12,
            'meta_update_freq': 13,
            'nb_eval_episodes': 9,
            'nb_rollout_steps': 8,
            'nb_train_steps': 7,
            'num_envs': 21,
            'policy_kwargs': {
                'cliprange': 24,
                'cliprange_vf': 25,
                'ent_coef': 26,
                'gamma': 27,
                'lam': 28,
                'learning_rate': 29,
                'max_grad_norm': 30,
                'n_minibatches': 31,
                'n_opt_epochs': 32,
                'vf_coef': 33,
                'l2_penalty': FEEDFORWARD_PARAMS["l2_penalty"],
                'model_params': {
                    'layers': [22, 23],
                    'filters': [16, 16, 16],
                    'ignore_flat_channels': [],
                    'ignore_image': False,
                    'image_channels': 3,
                    'image_height': 32,
                    'image_width': 32,
                    'kernel_sizes': [5, 5, 5],
                    'layer_norm': True,
                    'model_type': 'model_type',
                    'strides': [2, 2, 2]
                },
            },
        })


class TestRewardFns(unittest.TestCase):
    """Test the reward_fns method."""

    def test_negative_distance(self):
        a = np.array([1, 2, 10])
        b = np.array([1, 2])
        c = negative_distance(b, b, a, goal_indices=[1, 2])
        self.assertEqual(c, -8.062257748304752)


class TestEnvUtil(unittest.TestCase):
    """Test the environment utility methods."""

    def test_meta_ac_space(self):
        # non-relevant parameters for most tests
        params = dict(
            ob_space=None,
            relative_goals=False,
        )
        rel_params = params.copy()
        rel_params.update({"relative_goals": True})

        # test for AntMaze
        ac_space = get_meta_ac_space(env_name="AntMaze", **params)
        test_space(
            ac_space,
            expected_min=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3,
                                   -0.5, -0.3, -0.5, -0.3, -0.5, -0.3]),
            expected_max=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3,
                                   0.5, 0.3, 0.5, 0.3]),
            expected_size=15,
        )

        # test for AntGather
        ac_space = get_meta_ac_space(env_name="AntGather", **params)
        test_space(
            ac_space,
            expected_min=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3,
                                   -0.5, -0.3, -0.5, -0.3, -0.5, -0.3]),
            expected_max=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3,
                                   0.5, 0.3, 0.5, 0.3]),
            expected_size=15,
        )

        # test for AntPush
        ac_space = get_meta_ac_space(env_name="AntPush", **params)
        test_space(
            ac_space,
            expected_min=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3,
                                   -0.5, -0.3, -0.5, -0.3, -0.5, -0.3]),
            expected_max=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3,
                                   0.5, 0.3, 0.5, 0.3]),
            expected_size=15,
        )

        # test for AntFall
        ac_space = get_meta_ac_space(env_name="AntFall", **params)
        test_space(
            ac_space,
            expected_min=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3,
                                   -0.5, -0.3, -0.5, -0.3, -0.5, -0.3]),
            expected_max=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3,
                                   0.5, 0.3, 0.5, 0.3]),
            expected_size=15,
        )

        # test for UR5
        ac_space = get_meta_ac_space(env_name="UR5", **params)
        test_space(
            ac_space,
            expected_min=np.array([-2*np.pi, -2*np.pi, -2*np.pi, -4, -4, -4]),
            expected_max=np.array([2*np.pi, 2*np.pi, 2*np.pi, 4, 4, 4]),
            expected_size=6,
        )

        # test for Pendulum
        ac_space = get_meta_ac_space(env_name="Pendulum", **params)
        test_space(
            ac_space,
            expected_min=np.array([-np.pi, -15]),
            expected_max=np.array([np.pi, 15]),
            expected_size=2,
        )

        # test for ring-v0
        ac_space = get_meta_ac_space(env_name="ring-v0", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(5)]),
            expected_max=np.array([10 for _ in range(5)]),
            expected_size=5,
        )
        ac_space = get_meta_ac_space(env_name="ring-v0", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(5)]),
            expected_max=np.array([5 for _ in range(5)]),
            expected_size=5,
        )

        # test for ring-v0-fast
        ac_space = get_meta_ac_space(env_name="ring-v0-fast", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(1)]),
            expected_max=np.array([10 for _ in range(1)]),
            expected_size=1,
        )
        ac_space = get_meta_ac_space(env_name="ring-v0-fast", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(1)]),
            expected_max=np.array([5 for _ in range(1)]),
            expected_size=1,
        )

        # test for ring-v1-fast
        ac_space = get_meta_ac_space(env_name="ring-v1-fast", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(2)]),
            expected_max=np.array([10 for _ in range(2)]),
            expected_size=2,
        )
        ac_space = get_meta_ac_space(env_name="ring-v1-fast", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(2)]),
            expected_max=np.array([5 for _ in range(2)]),
            expected_size=2,
        )

        # test for ring-v2-fast
        ac_space = get_meta_ac_space(env_name="ring-v2-fast", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(3)]),
            expected_max=np.array([10 for _ in range(3)]),
            expected_size=3,
        )
        ac_space = get_meta_ac_space(env_name="ring-v2-fast", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(3)]),
            expected_max=np.array([5 for _ in range(3)]),
            expected_size=3,
        )

        # test for ring-v3-fast
        ac_space = get_meta_ac_space(env_name="ring-v3-fast", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(4)]),
            expected_max=np.array([10 for _ in range(4)]),
            expected_size=4,
        )
        ac_space = get_meta_ac_space(env_name="ring-v3-fast", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(4)]),
            expected_max=np.array([5 for _ in range(4)]),
            expected_size=4,
        )

        # test for ring-v4-fast
        ac_space = get_meta_ac_space(env_name="ring-v4-fast", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(5)]),
            expected_max=np.array([10 for _ in range(5)]),
            expected_size=5,
        )
        ac_space = get_meta_ac_space(env_name="ring-v4-fast", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(5)]),
            expected_max=np.array([5 for _ in range(5)]),
            expected_size=5,
        )

        # test for ring-imitation
        ac_space = get_meta_ac_space(env_name="ring-imitation", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(5)]),
            expected_max=np.array([1 for _ in range(5)]),
            expected_size=5,
        )

        # test for merge-v0
        ac_space = get_meta_ac_space(env_name="merge-v0", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(5)]),
            expected_max=np.array([1 for _ in range(5)]),
            expected_size=5,
        )
        ac_space = get_meta_ac_space(env_name="merge-v0", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(5)]),
            expected_max=np.array([0.5 for _ in range(5)]),
            expected_size=5,
        )

        # test for merge-v1
        ac_space = get_meta_ac_space(env_name="merge-v1", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(13)]),
            expected_max=np.array([1 for _ in range(13)]),
            expected_size=13,
        )
        ac_space = get_meta_ac_space(env_name="merge-v1", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(13)]),
            expected_max=np.array([0.5 for _ in range(13)]),
            expected_size=13,
        )

        # test for merge-v2
        ac_space = get_meta_ac_space(env_name="merge-v2", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(17)]),
            expected_max=np.array([1 for _ in range(17)]),
            expected_size=17,
        )
        ac_space = get_meta_ac_space(env_name="merge-v2", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-0.5 for _ in range(17)]),
            expected_max=np.array([0.5 for _ in range(17)]),
            expected_size=17,
        )

        # test for highway-v0
        ac_space = get_meta_ac_space(env_name="highway-v0", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(10)]),
            expected_max=np.array([20 for _ in range(10)]),
            expected_size=10,
        )
        ac_space = get_meta_ac_space(env_name="highway-v0", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(10)]),
            expected_max=np.array([5 for _ in range(10)]),
            expected_size=10,
        )

        # test for highway-v1
        ac_space = get_meta_ac_space(env_name="highway-v1", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(10)]),
            expected_max=np.array([20 for _ in range(10)]),
            expected_size=10,
        )
        ac_space = get_meta_ac_space(env_name="highway-v1", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(10)]),
            expected_max=np.array([5 for _ in range(10)]),
            expected_size=10,
        )

        # test for highway-v2
        ac_space = get_meta_ac_space(env_name="highway-v2", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(10)]),
            expected_max=np.array([20 for _ in range(10)]),
            expected_size=10,
        )
        ac_space = get_meta_ac_space(env_name="highway-v2", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(10)]),
            expected_max=np.array([5 for _ in range(10)]),
            expected_size=10,
        )

        # test for highway-v3
        ac_space = get_meta_ac_space(env_name="highway-v3", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(10)]),
            expected_max=np.array([20 for _ in range(10)]),
            expected_size=10,
        )
        ac_space = get_meta_ac_space(env_name="highway-v3", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(10)]),
            expected_max=np.array([5 for _ in range(10)]),
            expected_size=10,
        )

        # test for highway-imitation
        ac_space = get_meta_ac_space(env_name="highway-imitation", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(10)]),
            expected_max=np.array([1 for _ in range(10)]),
            expected_size=10,
        )

        # test for i210-v0
        ac_space = get_meta_ac_space(env_name="i210-v0", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(50)]),
            expected_max=np.array([20 for _ in range(50)]),
            expected_size=50,
        )
        ac_space = get_meta_ac_space(env_name="i210-v0", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(50)]),
            expected_max=np.array([5 for _ in range(50)]),
            expected_size=50,
        )

        # test for i210-v1
        ac_space = get_meta_ac_space(env_name="i210-v1", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(50)]),
            expected_max=np.array([20 for _ in range(50)]),
            expected_size=50,
        )
        ac_space = get_meta_ac_space(env_name="i210-v1", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(50)]),
            expected_max=np.array([5 for _ in range(50)]),
            expected_size=50,
        )

        # test for i210-v2
        ac_space = get_meta_ac_space(env_name="i210-v2", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(50)]),
            expected_max=np.array([20 for _ in range(50)]),
            expected_size=50,
        )
        ac_space = get_meta_ac_space(env_name="i210-v2", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(50)]),
            expected_max=np.array([5 for _ in range(50)]),
            expected_size=50,
        )

        # test for i210-v3
        ac_space = get_meta_ac_space(env_name="i210-v3", **params)
        test_space(
            ac_space,
            expected_min=np.array([0 for _ in range(50)]),
            expected_max=np.array([20 for _ in range(50)]),
            expected_size=50,
        )
        ac_space = get_meta_ac_space(env_name="i210-v3", **rel_params)
        test_space(
            ac_space,
            expected_min=np.array([-5 for _ in range(50)]),
            expected_max=np.array([5 for _ in range(50)]),
            expected_size=50,
        )

        # test for Point2DEnv
        ac_space = get_meta_ac_space(env_name="Point2DEnv", **params)
        test_space(
            ac_space,
            expected_min=np.array([-4 for _ in range(2)]),
            expected_max=np.array([4 for _ in range(2)]),
            expected_size=2,
        )

        # test for Point2DImageEnv
        ac_space = get_meta_ac_space(env_name="Point2DImageEnv", **params)
        test_space(
            ac_space,
            expected_min=np.array([-4 for _ in range(2)]),
            expected_max=np.array([4 for _ in range(2)]),
            expected_size=2,
        )

    def test_state_indices(self):
        # non-relevant parameters for most tests
        params = dict(
            ob_space=Box(-1, 1, shape=(2,)),
        )

        # test for AntMaze
        self.assertListEqual(
            get_state_indices(env_name="AntMaze", **params),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )

        # test for AntGather
        self.assertListEqual(
            get_state_indices(env_name="AntGather", **params),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )

        # test for AntPush
        self.assertListEqual(
            get_state_indices(env_name="AntPush", **params),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )

        # test for AntFall
        self.assertListEqual(
            get_state_indices(env_name="AntFall", **params),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )

        # test for UR5
        self.assertIsNone(get_state_indices(env_name="UR5", **params))

        # test for Pendulum
        self.assertListEqual(
            get_state_indices(env_name="Pendulum", **params),
            [0, 2]
        )

        # test for ring-v0
        self.assertListEqual(
            get_state_indices(env_name="ring-v0", **params),
            [0]
        )

        # test for ring-v0-fast
        self.assertListEqual(
            get_state_indices(env_name="ring-v0-fast", **params),
            [0]
        )

        # test for ring-v1-fast
        self.assertListEqual(
            get_state_indices(env_name="ring-v1-fast", **params),
            [0, 5]
        )

        # test for ring-v2-fast
        self.assertListEqual(
            get_state_indices(env_name="ring-v2-fast", **params),
            [0, 5, 10]
        )

        # test for ring-v3-fast
        self.assertListEqual(
            get_state_indices(env_name="ring-v3-fast", **params),
            [0, 5, 10, 15]
        )

        # test for ring-v4-fast
        self.assertListEqual(
            get_state_indices(env_name="ring-v4-fast", **params),
            [0, 5, 10, 15, 20]
        )

        # test for ring-imitation
        self.assertListEqual(
            get_state_indices(env_name="ring-imitation", **params),
            [0, 5, 10, 15, 20]
        )

        # test for merge-v0
        self.assertListEqual(
            get_state_indices(env_name="merge-v0", **params),
            [0, 5, 10, 15, 20]
        )

        # test for merge-v1
        self.assertListEqual(
            get_state_indices(env_name="merge-v1", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        )

        # test for merge-v2
        self.assertListEqual(
            get_state_indices(env_name="merge-v2", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        )

        # test for highway-v0
        self.assertListEqual(
            get_state_indices(env_name="highway-v0", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        )

        # test for highway-v1
        self.assertListEqual(
            get_state_indices(env_name="highway-v1", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        )

        # test for highway-v2
        self.assertListEqual(
            get_state_indices(env_name="highway-v2", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        )

        # test for highway-v3
        self.assertListEqual(
            get_state_indices(env_name="highway-v3", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        )

        # test for highway-imitation
        self.assertListEqual(
            get_state_indices(env_name="highway-imitation", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        )

        # test for i210-v0
        self.assertListEqual(
            get_state_indices(env_name="i210-v0", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
             85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
             155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215,
             220, 225, 230, 235, 240, 245]
        )

        # test for i210-v1
        self.assertListEqual(
            get_state_indices(env_name="i210-v1", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
             85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
             155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215,
             220, 225, 230, 235, 240, 245]
        )

        # test for i210-v2
        self.assertListEqual(
            get_state_indices(env_name="i210-v2", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
             85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
             155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215,
             220, 225, 230, 235, 240, 245]
        )

        # test for i210-v3
        self.assertListEqual(
            get_state_indices(env_name="i210-v3", **params),
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
             85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
             155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215,
             220, 225, 230, 235, 240, 245]
        )

        # test for Point2DEnv
        self.assertListEqual(
            get_state_indices(env_name="Point2DEnv", **params),
            [0, 1]
        )

        # test for Point2DImageEnv
        self.assertListEqual(
            get_state_indices(env_name="Point2DImageEnv", **params),
            [3072, 3073]
        )

    def test_import_flow_env(self):
        """Validate the functionality of the import_flow_env() method.

        This is done for the following 3 cases:

        1. "singleagent_ring"
        2. "multiagent_ring"
        3. "flow:sdfn" --> returns ValueError
        """
        # =================================================================== #
        # test case 1                                                         #
        # =================================================================== #
        env = import_flow_env(
            "flow:singleagent_ring", False, False, False, False)

        # check the spaces
        test_space(
            gym_space=env.action_space,
            expected_min=np.array([-1.]),
            expected_max=np.array([1.]),
            expected_size=1,
        )
        test_space(
            gym_space=env.observation_space,
            expected_min=np.array([-float("inf") for _ in range(3)]),
            expected_max=np.array([float("inf") for _ in range(3)]),
            expected_size=3,
        )

        # delete the environment
        del env

        # =================================================================== #
        # test case 2                                                         #
        # =================================================================== #
        env = import_flow_env(
            "flow:multiagent_ring", False, True, False, False)

        # check the spaces
        test_space(
            gym_space=env.action_space,
            expected_min=np.array([-1.]),
            expected_max=np.array([1.]),
            expected_size=1,
        )
        test_space(
            gym_space=env.observation_space,
            expected_min=np.array([-5 for _ in range(3)]),
            expected_max=np.array([5 for _ in range(3)]),
            expected_size=3,
        )

        # delete the environment
        del env

        # =================================================================== #
        # test case 3                                                         #
        # =================================================================== #
        self.assertRaises(
            ValueError,
            import_flow_env,
            env_name="flow:sdfn",
            render=False,
            shared=False,
            maddpg=False,
            evaluate=False,
        )


class TestTFUtil(unittest.TestCase):

    def setUp(self):
        self.sess = tf.compat.v1.Session()

    def tearDown(self):
        self.sess.close()

    def test_layer(self):
        """Check the functionality of the layer() method.

        This method is tested for the following features:

        1. the number of outputs from the layer equals num_outputs
        2. the name is properly used
        3. the proper activation function applied if requested
        4. layer_norm is applied if requested
        """
        # =================================================================== #
        # test case 1                                                         #
        # =================================================================== #

        # Choose a random number of outputs.
        num_outputs = random.randint(1, 10)

        # Create the layer.
        out_val = layer(
            val=tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='input_test1',
            ),
            num_outputs=num_outputs,
            name="test1",
        )

        # Test the number of outputs.
        self.assertEqual(out_val.shape[-1], num_outputs)

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 2                                                         #
        # =================================================================== #

        # Create the layer.
        out_val = layer(
            val=tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='input_test2',
            ),
            num_outputs=num_outputs,
            name="test2",
        )

        # Test the name matches what is expected.
        self.assertEqual(out_val.name, "test2/BiasAdd:0")

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 3                                                         #
        # =================================================================== #

        # Create the layer.
        out_val = layer(
            val=tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='input_test3',
            ),
            act_fun=tf.nn.relu,
            num_outputs=num_outputs,
            name="test3",
        )

        # Test that the name matches the activation function that was added.
        self.assertEqual(out_val.name, "Relu:0")

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 4                                                         #
        # =================================================================== #

        # Create the layer.
        _ = layer(
            val=tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='input_test4',
            ),
            layer_norm=True,
            num_outputs=num_outputs,
            name="test4",
        )

        # Test that the LayerNorm layer was added.
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['LayerNorm/beta:0',
             'LayerNorm/gamma:0',
             'test4/bias:0',
             'test4/kernel:0']
        )

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_conv_layer(self):
        """Check the functionality of the conv_layer() method.

        This method is tested for the following features:

        1. that the filters, kernel_size, and strides features work as expected
        2. the name is properly used
        3. the proper activation function applied if requested
        4. layer_norm is applied if requested
        """
        # =================================================================== #
        # test case 1                                                         #
        # =================================================================== #

        # Create the input variable.
        in_val = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, 32, 32, 3),
            name='input_test2',
        )

        # Create the layer.
        out_val = conv_layer(
            val=in_val,
            filters=16,
            kernel_size=5,
            strides=2,
            name="test2",
            act_fun=None,
            layer_norm=False,
        )

        # Test the shape of the output.
        self.assertEqual(out_val.shape[-1], 16)
        self.assertEqual(out_val.shape[-2], 16)
        self.assertEqual(out_val.shape[-3], 16)

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 2                                                         #
        # =================================================================== #

        # Create the input variable.
        in_val = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, 32, 32, 3),
            name='input_test2',
        )

        # Create the layer.
        out_val = conv_layer(
            val=in_val,
            filters=16,
            kernel_size=5,
            strides=2,
            name="test2",
            act_fun=None,
            layer_norm=False,
        )

        # Test the name matches what is expected.
        self.assertEqual(out_val.name, "test2/BiasAdd:0")

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 3                                                         #
        # =================================================================== #

        # Create the input variable.
        in_val = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, 32, 32, 3),
            name='input_test3',
        )

        # Create the layer.
        out_val = conv_layer(
            val=in_val,
            filters=16,
            kernel_size=5,
            strides=2,
            name="test3",
            act_fun=tf.nn.tanh,
            layer_norm=False,
        )

        # Test that the name matches the activation function that was added.
        self.assertEqual(out_val.name, "Tanh:0")

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 4                                                         #
        # =================================================================== #

        # Create the input variable.
        in_val = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, 32, 32, 3),
            name='input_test4',
        )

        # Create the layer.
        _ = conv_layer(
            val=in_val,
            filters=16,
            kernel_size=5,
            strides=2,
            name="test4",
            act_fun=None,
            layer_norm=True,
        )

        # Test that the LayerNorm layer was added.
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['LayerNorm/beta:0',
             'LayerNorm/gamma:0',
             'test4/bias:0',
             'test4/kernel:0']
        )

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_gaussian_likelihood(self):
        """Check the functionality of the gaussian_likelihood() method."""
        input_ = tf.constant([[0, 1, 2]], dtype=tf.float32)
        mu_ = tf.constant([[0, 0, 0]], dtype=tf.float32)
        log_std = tf.constant([[-4, -3, -2]], dtype=tf.float32)
        val = gaussian_likelihood(input_, mu_, log_std)
        expected = -304.65784

        self.assertAlmostEqual(self.sess.run(val)[0], expected, places=4)

    def test_apply_squashing(self):
        """Check the functionality of the apply_squashing() method."""
        # Some inputs
        mu_ = tf.constant([[0, 0.5, 1, 2]], dtype=tf.float32)
        pi_ = tf.constant([[0, 0.5, 1, 2]], dtype=tf.float32)
        logp_pi = tf.constant([[0, 0.5, 1, 2]], dtype=tf.float32)

        # Run the function.
        det_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)

        # Initialize everything.
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        # Test the output from the deterministic squashed output.
        np.testing.assert_almost_equal(
            sess.run(det_policy), [[0., 0.4621172, 0.7615942, 0.9640276]])

        # Clear the graph.
        tf.compat.v1.reset_default_graph()


def test_space(gym_space, expected_size, expected_min, expected_max):
    """Test the shape and bounds of an action or observation space.

    Parameters
    ----------
    gym_space : gym.spaces.Box
        gym space object to be tested
    expected_size : int
        expected size
    expected_min : float or array_like
        expected minimum value(s)
    expected_max : float or array_like
        expected maximum value(s)
    """
    assert gym_space.shape[0] == expected_size, \
        "{}, {}".format(gym_space.shape[0], expected_size)
    np.testing.assert_almost_equal(gym_space.high, expected_max, decimal=4)
    np.testing.assert_almost_equal(gym_space.low, expected_min, decimal=4)


if __name__ == '__main__':
    unittest.main()
