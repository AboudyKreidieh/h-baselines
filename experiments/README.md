# Benchmarking HRL Models and Algorithms

TODO

## Contents

* [Running Existing Models and Algorithms](#running-existing-models-and-algorithms)
* [Visualizing Pre-trained Results](#visualizing-pre-trained-results)

## Running Existing Models and Algorithms

These are three existing models, using policies: the feed-forward policy, the goal-conditioned policy, and the multi-agent feed-forward policy.

To run these models, use command:
```shell
python MODEL.py ENV_NAME
```
with `run_fcnet.py` for feed-forward policy, `run_hrl.py` for goal-conditioned policy, `run_multi_fcnet.py` for multi-agent feed-forward policy in place of `MODEL.py`.

The following optional command-line arguments may be passed in to adjust the choice of algorithm:

* `--alg` (*str*): The algorithm to use. Must be one of [TD3, SAC]. Defaults to
  'TD3'.
* `--evaluate` (*store_true*): whether to add an evaluation environment. The 
  evaluation environment is similar to the training environment, but with 
  `env_params.evaluate` set to True.
* `--n_training` (*int*): Number of training operations to perform. Each 
  training operation is performed on a new seed. Defaults to 1.
* `--total_steps` (*int*): Total number of timesteps used during training. 
  Defaults to 1000000.
* `--seed` (*int*): Sets the seed for numpy, tensorflow, and random. Defaults 
  to 1.
* `--log_interval` (*int*): the number of training steps before logging 
  training results. Defaults to 2000.
* `--eval_interval` (*int*): number of simulation steps in the training 
  environment before an evaluation is performed. Only relevant if `--evaluate` 
  is called. Defaults to 50000.
* `--save_interval` (int): number of simulation steps in the training 
  environment before the model is saved. Defaults to 50000.
* `--initial_exploration_steps` (*int*): number of timesteps that the policy is
  run before training to initialize the replay buffer with samples. Defaults to
  10000.

The following optional command-line arguments may be passed in to adjust variable hyperparameters of the algorithms:

* `--nb_train_steps` (*int*): the number of training steps. Defaults to 1.
* `--nb_rollout_steps` (*int*): the number of rollout steps. Defaults to 1.
* `--nb_eval_episodes` (*int*): the number of evaluation episodes. Only 
  relevant if `--evaluate` is called. Defaults to 50.
* `--reward_scale` (*float*): the value the reward should be scaled by. 
  Defaults to 1.
* `--render`: enable rendering of the environment.
* `--render_eval`: enable rendering of the evaluation environment.
* `--verbose` (*int*): the verbosity level: 0 none, 1 training information, 2 tensorflow debug. Defaults to 2.
* `--actor_update_freq` (*int*): the number of training steps per actor policy update step. The critic policy is updated every training step. Only used when 
  the algorithm is set to "TD3". Defaults to 2.
* `--meta_update_freq` (*int*): the number of training steps per meta policy update step. Defaults to 10.

Additionally, each model can take optional arguments specifically for respective policies.

### Fcnet Model with Feed-forward Policy

* `--buffer_size` (*int*): the max number of transitions to store. Defaults to 
  200000.
* `--batch_size` (*int*): the size of the batch for learning the policy. 
  Defaults to 128.
* `--actor_lr` (*float*): the actor learning rate. Defaults to 3e-4.
* `--critic_lr` (*float*): the critic learning rate. Defaults to 3e-4.
* `--tau` (*float*): the soft update coefficient (keep old values, between 0 
  and 1). Defatuls to 0.005.
* `--gamma` (*float*): the discount rate. Defaults to 0.99.
* `--layer_norm` (*store_true*): enable layer normalisation
* `--use_huber` (*store_true*): specifies whether to use the huber distance 
  function as the loss for the critic. If set to False, the mean-squared error 
  metric is used instead.

### Hierarchical RL Model with Goal-conditioned Policy

* `--num_levels` (*int*): the number of levels within the hierarchy. Must be greater than 1. Defaults to 2.
* `--meta_period` (*int*): the meta-policy action period. Defaults to 10.
* `--intrinsic_reward_scale` (*int*): the value that the intrinsic reward should be scaled by. Defaults to 1.
* `--relative_goals` (*store_true*): whether the goal issued by the higher-level policies is meant to be a relative or absolute goal. 
* `--off_policy_corrections` (*store_true*): whether to use off-policy corrections during the update procedure. See: https://arxiv.org/abs/1805.08296.
* `--hindsight` (*store_true*): whether to include hindsight action and goal transitions in the replay buffer. See: https://arxiv.org/abs/1712.00948
* `--subgoal_testing_rate` (*float*): the rate at which the original (non-hindsight) sample is stored in the replay buffer as well. Used only if `hindsight` is set to True. Defaults to 0.3.
* `--connected_gradients` (*store_true*): whether to use the connected gradient update actor update procedure to the higher-level policies. See: https://arxiv.org/abs/1912.02368v1
* `--cg_weights` (*float*): weights for the gradients of the loss of the lower-level policies with respect to the parameters of the higher-level policies. Only used if `connected_gradients` is set to True. Defaults to 0.0005.
* `--use_fingerprints` (*store_true*): whether to add a time-dependent fingerprint to the observations. 
* `--centralized_value_functions` (*store_true*): whether to use centralized value functions. 

### Fcnet Model with Multi-agent Feed-forward Policy

All optional arguments the same as in regular feed-forward policy, with two extra optional arguments:

* `--shared` (*store_true*): whether to use a shared policy for all agents
* `--maddpg` (*store_true*): whether to use an algorithm-specific variant of 
  the MADDPG algorithm

### Evaluator Script

An evaluator script is written to run evaluation episodes of a given checkpoint using pre-trained policies. Run with the following command:

```shell
python run_eval.py DIR_NAME
```
with `DIR_NAME` as path to the checkpoints folder.

Some optional arguments to be passed in are:
* `--ckpt_num` (*int*): the checkpoint number. If not specified, the last checkpoint is used.
* `--num_rollouts` (*int*): the number of eval episodes. Defaults to 1.
* `--no_render` (*store_true*): shuts off rendering.

## Visualizing Pre-trained Results

TODO