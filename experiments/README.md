# Benchmarking HRL Models and Algorithms

We provide a sequence of runner and evaluation scripts to validate the 
performance of the various algorithms provided within this repository. The 
environments that are currently supported for training can be found 
[here](https://github.com/AboudyKreidieh/h-baselines#3-environments). If you 
would like to test these algorithms on custom environments, refer to this 
[section](#3-training-on-custom-environments) on incorporating custom 
environments.

If you are attempting to recreate our results from the paper titled 
"Inter-Level Cooperation in Hierarchical Reinforcement Learning", refer to this
[section](#4-performance-of-the-cher-algorithm).

## Contents

1. [Running Existing Models and Algorithms](#1-running-existing-models-and-algorithms)
2. [Visualizing Pre-trained Results](#2-visualizing-pre-trained-results)  
    2.1 [Plotting Learning Curves](#21-plotting-learning-curves)  
    2.2 [Visualizing Pr-trained Models](#22-visualizing-pr-trained-models)  
3. [Training on Custom Environments](#3-training-on-custom-environments)
4. [Performance of the CHER Algorithm](#4-performance-of-the-cher-algorithm)  
   4.1. [Rerunning Experiments](#41-rerunning-experiments)  
   4.2. [Downloading and Replaying Pre-trained Models](#42-downloading-and-replaying-pre-trained-models)  

## 1. Running Existing Models and Algorithms

These are three existing models, using policies: the feed-forward policy, the
goal-conditioned policy, and the multi-agent feed-forward policy.

To run these models, use command:
```shell script
python MODEL.py ENV_NAME
```
with `run_fcnet.py` for feed-forward policy, `run_hrl.py` for goal-conditioned
policy, `run_multi_fcnet.py` for multi-agent feed-forward policy in place of
`MODEL.py`.

The following optional command-line arguments may be passed in to adjust the
choice of algorithm:

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

The following optional command-line arguments may be passed in to adjust
variable hyperparameters of the algorithms:

* `--nb_train_steps` (*int*): the number of training steps. Defaults to 1.
* `--nb_rollout_steps` (*int*): the number of rollout steps. Defaults to 1.
* `--nb_eval_episodes` (*int*): the number of evaluation episodes. Only 
  relevant if `--evaluate` is called. Defaults to 50.
* `--reward_scale` (*float*): the value the reward should be scaled by. 
  Defaults to 1.
* `--render`: enable rendering of the environment.
* `--render_eval`: enable rendering of the evaluation environment.
* `--verbose` (*int*): the verbosity level: 0 none, 1 training information, 2
  tensorflow debug. Defaults to 2.
* `--actor_update_freq` (*int*): the number of training steps per actor policy
  update step. The critic policy is updated every training step. Only used when 
  the algorithm is set to "TD3". Defaults to 2.
* `--meta_update_freq` (*int*): the number of training steps per meta policy
  update step. Defaults to 10.

Additionally, each model can take optional arguments specifically for
respective policies.

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

* `--num_levels` (*int*): the number of levels within the hierarchy. Must be
  greater than 1. Defaults to 2.
* `--meta_period` (*int*): the meta-policy action period. Defaults to 10.
* `--intrinsic_reward_type` (*str*): the reward function to be used by the 
  lower-level policies. See the base goal-conditioned policy for a description.
  Defaults to "negative_distance".
* `--intrinsic_reward_scale` (*int*): the value that the intrinsic reward
  should be scaled by. Defaults to 1.
* `--relative_goals` (*store_true*): whether the goal issued by the
  higher-level policies is meant to be a relative or absolute goal. 
* `--off_policy_corrections` (*store_true*): whether to use off-policy
  corrections during the update procedure. See: 
  https://arxiv.org/abs/1805.08296.
* `--hindsight` (*store_true*): whether to include hindsight action and goal
  transitions in the replay buffer. See: https://arxiv.org/abs/1712.00948
* `--subgoal_testing_rate` (*float*): the rate at which the original
  (non-hindsight) sample is stored in the replay buffer as well. Used only if
  `hindsight` is set to True. Defaults to 0.3.
* `--cooperative_gradients` (*store_true*): whether to use the cooperative
  gradient update procedure for the higher-level policies. See:
  https://arxiv.org/abs/1912.02368v1
* `--cg_weights` (*float*): weights for the gradients of the loss of the
  lower-level policies with respect to the parameters of the higher-level
  policies. Only used if `cooperative_gradients` is set to True. Defaults to
  0.0005.
* `--use_fingerprints` (*store_true*): whether to add a time-dependent
  fingerprint to the observations. 
* `--centralized_value_functions` (*store_true*): whether to use centralized
  value functions. 

### Fcnet Model with Multi-agent Feed-forward Policy

All optional arguments the same as in regular feed-forward policy, with two
extra optional arguments:

* `--shared` (*store_true*): whether to use a shared policy for all agents
* `--maddpg` (*store_true*): whether to use an algorithm-specific variant of 
  the MADDPG algorithm

## 2. Visualizing Pre-trained Results

### 2.1 Plotting Learning Curves

Results from a sequence of completed training operations using the following 
command:

```shell script
python plot.py FOLDERS
```

The method take as a required input the path to where the model/algorithm 
results are located. The results from each unique model/algorithm must be 
separated by a folder. For example, if you were to have two results when 
utilizing TD3, and one when utilizing SAC, the file structure should look 
something similar to the following:

```
experiments -----|
                 |
                 |------ plot.py
                 |
                 |------ TD3
                 |        |--- 0
                 |        |    |--- train.csv
                 |        |
                 |        |--- 1
                 |             |--- train.csv
                 |
                 |------ SAC
                          |--- 0
                               |--- train.csv
```

The command under this situation may be filled as follows:

```shell script
python plot.py "TD3" "SAC"
```

Additional command-line arguments are:

* `--names` (*list of str*) : The names to be assigned for each result. Must be
  equal in size to the number of folder specified. If not assigned, no legend 
  is added.
* `--out` (*str*) : the path where the figure should be saved. Append with a 
  .svg file if you would like to generate SVG formatted graphs.
* `--use_eval` (*store_true*) : whether to use the eval_*.csv or train.csv 
  files to generate the plots
* `--y` (*str*) : the column to use for the y-coordinates
* `--x` (*str*) : the column to use for the x-coordinates
* `--ylabel` (*str*) : the label to use for the y-axis. If set to None, the 
  name of the column used for the y-coordinates is used.
* `--xlabel` (*str*) : the label to use for the x-axis. If set to None, the 
  name of the column used for the x-coordinates is used.
* `--show` (*store_true*) : whether to show the figure that was saved

### 2.2 Visualizing Pr-trained Models

An evaluator script is written to run evaluation episodes of a given checkpoint
using pre-trained policies. Run with the following command:

```shell script
python run_eval.py DIR_NAME
```
with `DIR_NAME` as path to the checkpoints folder.

Some optional arguments to be passed in are:
* `--ckpt_num` (*int*): the checkpoint number. If not specified, the last
  checkpoint is used.
* `--num_rollouts` (*int*): the number of eval episodes. Defaults to 1.
* `--no_render` (*store_true*): shuts off rendering.
* `--random_seed` (*store_true*): whether to run the simulation on a random 
  seed. If not added, the original seed is used.

## 3. Training on Custom Environments

In addition to typical environments registered within `gym` or provided by the
Flow examples folder, any gym-compatible environment can be made to run via 
the above commands. In order to include support for custom environments, you 
will need to add said environment to the `ENV_ATTRIBUTES` dictionary object 
located in 
[hbaselines/utils/env_util.py](https://github.com/AboudyKreidieh/h-baselines/blob/master/hbaselines/utils/env_util.py).
Via this dict, new environments can be specified as individual elements whose
keys represents the name of the environment when being run. For example, if you
would like to incorporate a new environment called "myEnv", the environment 
would be included to this dict as follows:

```python
ENV_ATTRIBUTES = {
    # do not delete existing environments
    # ...
    # create a new environment named "myEnv"
    "myEnv": {
        "meta_ac_space": None,
        "state_indices": None,
        "env": None,
    },
}
```

The components of any key within the dictionary are used to specify the method
for instantiating and returning the environment, as well as the indices within
the observation that are assigned goals by meta-policies and the bounds of 
these goals. This is broken down into three subcategories, defined as follows:

* **meta_ac_space:** a lambda function that takes an input whether the higher 
  level policies are assigning relative goals and returns the action space of 
  the higher level policies
* **state_indices:** a list that assigns the indices that correspond to goals 
  in the Worker's state space
* **env:** a lambda term that takes an input (evaluate, render, multiagent, 
  shared, maddpg) and return an environment or list of environments

For example, taking the AntGather environment located within the original 
dictionary as an example, and described within the main directory's README, the
inclusion of said environment via the "myEnv" key can be performed as follows:

```python
import numpy as np
from gym.spaces import Box
from hbaselines.envs.snn4hrl.envs import AntGatherEnv

ENV_ATTRIBUTES = {
    # do not delete existing environments
    # ...
    # create a new environment named "myEnv"
    "myEnv": {
        "meta_ac_space": lambda relative_goals: Box(
            low=np.array([-10, -10, -0.5, -1, -1, -1, -1, -0.5, -0.3, -0.5,
                          -0.3, -0.5, -0.3, -0.5, -0.3]),
            high=np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5,
                           0.3, 0.5, 0.3]),
            dtype=np.float32,
        ),
        "state_indices": [i for i in range(15)],
        "env": lambda evaluate, render, multiagent, shared, maddpg:
        AntGatherEnv(),
    },
}
```

Finally, the environment can now be run by any of the commands provided within
this file. For example, if you would like to train "myEnv" with a fcnet TD3 
policy, run the following command:

```shell script
python run_fcnet.py "myEnv"
```

## 4. Performance of the CHER Algorithm

We explore the potential benefits of incentivizing cooperation between levels 
of a hierarchy on the training performance of goal-conditioned hierarchies. 
This is presented in the following [paper](), with the implementation of the 
resultant algorithm, called CHER, being provided in this repository. In the 
following subsections, we describe how the results from this paper can be 
recreated, and provide a set of pre-trained models for visualization purposes.

### 4.1 Rerunning Experiments

To recreate any of the results for the given environment/algorithm pairs from 
the original paper, run the appropriate command below.

* **TD3:**

  * AntGather
    ```shell script
    python run_fcnet.py "AntGather" --reward_scale 10 --use_huber
    ```
  * AntMaze
    ```shell script
    python run_fcnet.py "AntMaze" --use_huber --evaluate --eval_interval 50000 \
        --nb_eval_episodes 50 --total_steps 3000000
    ```
  * BipedalSoccer
    ```shell script
    python run_fcnet.py "BipedalSoccer" --use_huber --total_steps 3000000
    ```
  * highway-v1
    ```shell script
    python run_fcnet.py "highway-v1" --use_huber --nb_rollout_steps 10 \
        --nb_train_steps 10  --log_interval 15000 --total_steps 1500000
    ```

* **HRL:**

  * AntGather
    ```shell script
    python run_hrl.py "AntGather" --reward_scale 10 --use_huber --relative_goals
    ```
  * AntMaze
    ```shell script
    python run_hrl.py "AntMaze" --use_huber --evaluate --eval_interval 50000 \
        --nb_eval_episodes 50 --total_steps 3000000 --relative_goals
    ```
  * BipedalSoccer
    ```shell script
    python run_hrl.py "BipedalSoccer" --use_huber --total_steps 3000000 \
        --relative_goals
    ```
  * highway-v1
    ```shell script
    python run_hrl.py "highway-v1" --use_huber --nb_rollout_steps 10 \
        --nb_train_steps 10  --log_interval 15000 --total_steps 1500000
    ```

* **HIRO:**

  * AntGather
    ```shell script
    python run_hrl.py "AntGather" --reward_scale 10 --use_huber --relative_goals \
        --off_policy_corrections
    ```
  * AntMaze
    ```shell script
    python run_hrl.py "AntMaze" --use_huber --evaluate --eval_interval 50000 \
        --nb_eval_episodes 50 --total_steps 3000000 --relative_goals \
        --off_policy_corrections
    ```
  * BipedalSoccer
    ```shell script
    python run_hrl.py "BipedalSoccer" --use_huber --total_steps 3000000 \
        --relative_goals --off_policy_corrections
    ```
  * highway-v1
    ```shell script
    python run_hrl.py "highway-v1" --use_huber --nb_rollout_steps 10 \
        --nb_train_steps 10  --log_interval 15000 --total_steps 1500000 \
        --off_policy_corrections
    ```

* **HAC:**

  * AntGather
    ```shell script
    python run_hrl.py "AntGather" --reward_scale 10 --use_huber --relative_goals \
        --hindsight
    ```
  * AntMaze
    ```shell script
    python run_hrl.py "AntMaze" --use_huber --evaluate --eval_interval 50000 \
        --nb_eval_episodes 50 --total_steps 3000000 --relative_goals --hindsight
    ```
  * BipedalSoccer
    ```shell script
    python run_hrl.py "BipedalSoccer" --use_huber --total_steps 3000000 \
        --relative_goals --hindsight
    ```
  * highway-v1
    ```shell script
    python run_hrl.py "highway-v1" --use_huber --nb_rollout_steps 10 \
        --nb_train_steps 10  --log_interval 15000 --total_steps 1500000 \
        --hindsight
    ```

* **CHER:**

  * AntGather
    ```shell script
    python run_hrl.py "AntGather" --reward_scale 10 --use_huber --relative_goals \
        --cooperative_gradients --cg_weights 0.01
    ```
  * AntMaze
    ```shell script
    python run_hrl.py "AntMaze" --use_huber --evaluate --eval_interval 50000 \
        --nb_eval_episodes 50 --total_steps 3000000 --relative_goals \
        --cooperative_gradients --cg_weights 0.005
    ```
  * BipedalSoccer
    ```shell script
    python run_hrl.py "BipedalSoccer" --use_huber --total_steps 3000000 \
        --relative_goals --cooperative_gradients --cg_weights 0.01
    ```
  * highway-v1
    ```shell script
    python run_hrl.py "highway-v1" --use_huber --nb_rollout_steps 10 \
        --nb_train_steps 10  --log_interval 15000 --total_steps 1500000 \
        --cooperative_gradients --cg_weights 0.02
    ```

### 4.2 Downloading and Replaying Pre-trained Models

We provide an example of the final policy generated from each of the above 
described algorithm/environment pairs. Each of these policies were generated 
when utilizing a equivalent seed in order to ensure a somewhat fair comparison.
In order to download all of these policies, run the following command:

```shell script
scripts/import_cher.sh
```

One the above command is completed, a new file in the experiments folder will 
be created titled "pretrained". The file structure for this directory will look
as follows:

```
h-baselines/experiments/pretrained
                            |
                            |-- CHER
                            |    |-- AntGather
                            |    |-- AntMaze
                            |    └-- highway-v1
                            |
                            |-- HAC
                            |    |-- AntGather
                            |    |-- AntMaze
                            |    └-- highway-v1
                            |
                            |-- HIRO
                            |    |-- AntGather
                            |    |-- AntMaze
                            |    └-- highway-v1
                            |
                            |-- HRL
                            |    |-- AntGather
                            |    |-- AntMaze
                            |    └-- highway-v1
                            |
                            └-- TD3
                                 |-- AntGather
                                 |-- AntMaze
                                 └-- highway-v1
```

Each of these final directories will contain the data generated by any of the 
runner scripts provided in the folder. The policy provided within these 
directories can accordingly be replayed via the `run_eval.py` scripts. For 
example, if you would like to replay the AntGather policy from the CHER 
algorithm, you can do so by running the following command:

```shell script
python run_eval.py "pretrained/CHER/AntGather" --random_seed
```

Note that we add the `--random_seed` attribute so that every replay produces a 
different behavior.
